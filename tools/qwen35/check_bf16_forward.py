"""Layer-streamed official Qwen HF oracle for one complete native causal trace.

Holds one dense decoder layer and one expert at a time; no full expert bank.
The output head streams to GPU. Failed tolerances remain failures, exit 1.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from test_bf16_layers import read_tensor


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('index','config','trace','state','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    from transformers import DynamicCache
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as hf
    ix=json.loads(a.index.read_text());assert ix['complete'] and ix['scope']=='full-text-model'
    root=Path(ix['model_root']);cfg=json.loads(a.config.read_text());ids=cfg['input_tokens'];t=len(ids);assert 0<t<=32
    c=Qwen3_5MoeTextConfig(**ix['config']);c._attn_implementation='sdpa'
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    report={'scope':'complete 40-layer Qwen causal prefill, layer-streamed HF oracle','revision':ix['revision'],'sdpa_backend':'math','first_router_set_difference':None,'layers':[],'failures':[]}

    def native(name):
        p=Path(name);out=torch.from_numpy(np.frombuffer(p.read_bytes(),dtype=np.float32 if p.suffix=='.f32' else np.uint16).copy())
        return out if p.suffix=='.f32' else out.view(torch.bfloat16)

    def error(ref,actual):
        ref=ref.detach().float().cpu().flatten();actual=actual.float().cpu().flatten();assert ref.shape==actual.shape
        d=(ref-actual).abs();return {'max_abs_error':float(d.max()),'mean_abs_error':float(d.mean()),'relative_rms_error':float(d.square().mean().sqrt()/ref.square().mean().sqrt().clamp_min(1e-12)),'different_components':int((ref!=actual).sum())}

    emb=ix['tensors']['model.embed_tokens.weight'];xs=[]
    for token in ids:xs.append(read_tensor(root,{'file':emb['file'],'offset':emb['offset']+token*4096,'bytes':4096},(2048,)))
    x=torch.stack(xs)[None].cuda();assert torch.equal(x.cpu().flatten(),native(a.trace/'embedding.bf16'))
    cache=DynamicCache(config=c);positions=torch.arange(t,device='cuda')[None].expand(3,1,t);rotary=hf.Qwen3_5MoeTextRotaryEmbedding(c).cuda()
    mask=(torch.arange(t,device='cuda')[None,:]<=torch.arange(t,device='cuda')[:,None])[None,None]
    with torch.inference_mode(),torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
        for layer_id in range(40):
            with torch.device('meta'):layer=hf.Qwen3_5MoeDecoderLayer(c,layer_id)
            for name,param in list(layer.named_parameters()):
                if name.startswith('mlp.experts.'):continue
                parent,leaf=name.rsplit('.',1);value=read_tensor(root,ix['tensors'][f'model.layers.{layer_id}.'+name]).cuda()
                layer.get_submodule(parent).register_parameter(leaf,torch.nn.Parameter(value,requires_grad=False))
            routing={}

            class StreamingExperts(torch.nn.Module):
                def forward(self,activations,selected,weights):
                    routing['ids']=selected.cpu();routing['weights']=weights.float().cpu();out=torch.zeros_like(activations)
                    for e in selected.unique(sorted=True).tolist():
                        token,slot=torch.where(selected==e);entry=ix['experts'][layer_id*256+e]
                        gu=read_tensor(root,entry['gate_up'],(1024,2048)).cuda();down=read_tensor(root,entry['down'],(2048,512)).cuda()
                        gate,up=F.linear(activations[token],gu).chunk(2,-1);y=F.linear(F.silu(gate)*up,down)*weights[token,slot,None]
                        out.index_add_(0,token,y.to(out.dtype))
                    return out

            layer.mlp.experts=StreamingExperts();layer.eval()
            original=native(a.trace/f'layer{layer_id}-input.bf16').reshape(1,t,2048)
            y=layer(x,position_embeddings=rotary(x,positions),attention_mask=mask if c.layer_types[layer_id]=='full_attention' else None,past_key_values=cache)
            nr=json.loads((a.trace/f'layer{layer_id}-routes.json').read_text());ni=torch.tensor(nr['ids']).reshape(t,8)
            different=int((ni.sort(-1).values!=routing['ids'].sort(-1).values).any(-1).sum())
            row={'layer':layer_id,'kind':c.layer_types[layer_id],'input_error':error(x,original),'output_error':error(y,native(a.trace/f'layer{layer_id}-output.bf16')),'different_expert_set_rows':different}
            if different and report['first_router_set_difference'] is None:
                report['first_router_set_difference']=layer_id
                local=layer(original.cuda(),position_embeddings=rotary(original.cuda(),positions),attention_mask=mask if c.layer_types[layer_id]=='full_attention' else None,past_key_values=DynamicCache(config=c))
                row['isolated_native_input_output_error']=error(local,native(a.trace/f'layer{layer_id}-output.bf16'))
                row['isolated_router_set_rows']=int((ni.sort(-1).values!=routing['ids'].sort(-1).values).any(-1).sum())
            if c.layer_types[layer_id]=='linear_attention':
                row['conv_state_error']=error(cache.get_conv_state(layer_id,0)[...,-3:],native(a.state/f'layer{layer_id}-conv.bf16'))
                row['ssm_state_error']=error(cache.get_recurrent_state(layer_id,0).transpose(-1,-2),native(a.state/f'layer{layer_id}-ssm.f32'))
            else:
                row['key_state_error']=error(cache.layers[layer_id].keys.transpose(1,2),native(a.state/f'layer{layer_id}-key.bf16'))
                row['value_state_error']=error(cache.layers[layer_id].values.transpose(1,2),native(a.state/f'layer{layer_id}-value.bf16'))
            if row['output_error']['relative_rms_error']>=0.004:report['failures'].append({'layer':layer_id,'output_error':row['output_error']})
            report['layers'].append(row);print(json.dumps({'layer':layer_id,'output_rms':row['output_error']['relative_rms_error'],'route_set_rows':different}),flush=True)
            x=y;del layer;torch.cuda.empty_cache()
        norm=hf.Qwen3_5MoeRMSNorm(c.hidden_size,c.rms_norm_eps).bfloat16().cuda();norm.weight.data.copy_(read_tensor(root,ix['tensors']['model.norm.weight']).cuda());x=norm(x)
        report['final_norm_error']=error(x,native(a.trace/'final_norm.bf16'))
        desc=ix['tensors']['lm_head.weight'];head=torch.empty(desc['shape'],device='cuda',dtype=torch.bfloat16);flat=head.view(-1)
        with (root/desc['file']).open('rb') as f:
            f.seek(desc['offset']);at=0
            while at<desc['bytes']:
                raw=bytearray(f.read(min(4*1024**2,desc['bytes']-at)));assert raw
                part=torch.frombuffer(raw,dtype=torch.bfloat16);flat[at//2:at//2+part.numel()].copy_(part);at+=len(raw)
        logits=F.linear(x,head).float().cpu().reshape(t,c.vocab_size);actual=native(a.trace/'logits.bf16').float().reshape_as(logits)
        report['logits_error']=error(logits,actual);report['native_argmax']=actual.argmax(-1).tolist();report['reference_argmax']=logits.argmax(-1).tolist();report['last_argmax_equal']=report['native_argmax'][-1]==report['reference_argmax'][-1]
    report['passed']=not report['failures'] and report['first_router_set_difference'] is None and report['logits_error']['relative_rms_error']<0.004
    (a.output/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ('layers','failures')},indent=2));assert report['passed']


if __name__=='__main__':main()
