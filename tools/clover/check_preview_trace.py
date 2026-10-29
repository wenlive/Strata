"""Full Clover preview oracle with identical, layer-streamed committed history.

This conditions HF and native preview on the SAME saved native history. It does
not claim independent end-to-end HF generation parity or benchmark performance.
Only one decoder layer and one routed expert are resident for the reference.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from test_official_layers import oracle_namespace, read_tensor


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('index','trace','state','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--tokens',type=int,nargs='+',help='Preview IDs; default all mask, inferred width from trace embedding')
    p.add_argument('--isolate-each-layer',action='store_true')
    p.add_argument('--pad-expert-rows',action='store_true',help='Diagnostic only: zero-pad expert GEMMs to native fixed32 rows, then slice actual rows')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    ix=json.loads(a.index.read_text());assert ix['complete'] and ix['scope']=='full-model'
    c=SimpleNamespace(**ix['config']);root=Path(ix['model_root']);ns=oracle_namespace(root/'modeling_clover_moe.py')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    t=(a.trace/'embedding.bf16').stat().st_size//(c.hidden_size*2);ids=a.tokens or [c.mask_token_id]*t;assert len(ids)==t
    manifest=json.loads((a.state/'manifest.json').read_text());prefix=manifest['prefix_tokens']
    report={'scope':__doc__,'revision':ix['revision'],'width':t,'prefix_tokens':prefix,'tokens':ids,'sdpa_backend':'math',
            'history_policy':'identical saved native history in both preview paths; not independent HF history','diagnostic_padded_expert_rows':a.pad_expert_rows,'layers':[],'failures':[],'first_expert_set_difference':None}

    def native(path):
        x=np.fromfile(path,dtype='<f4' if path.suffix=='.f32' else '<u2').copy();v=torch.from_numpy(x)
        return v if path.suffix=='.f32' else v.view(torch.bfloat16)

    def error(ref,actual):
        x=ref.detach().float().cpu().flatten();y=actual.detach().float().cpu().flatten();assert x.shape==y.shape
        d=(x-y).abs();return {'max_abs_error':float(d.max()),'mean_abs_error':float(d.mean()),'relative_rms_error':float(d.square().mean().sqrt()/x.square().mean().sqrt().clamp_min(1e-12)),'different_components':int((x!=y).sum())}

    emb=ix['tensors']['model.embed_tokens.weight'];pieces=[]
    for token in ids:pieces.append(read_tensor(root,{'file':emb['file'],'offset':emb['offset']+token*c.hidden_size*2,'bytes':c.hidden_size*2},(c.hidden_size,)))
    x=torch.stack(pieces)[None].cuda();assert torch.equal(x.cpu().flatten(),native(a.trace/'embedding.bf16'))
    valid=torch.ones(1,t,device='cuda',dtype=torch.bool);prefix_valid=torch.ones(1,prefix,device='cuda',dtype=torch.bool)
    positions=torch.arange(prefix,prefix+t,device='cuda')[None]
    with torch.inference_mode(),torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
        for l in range(c.num_hidden_layers):
            with torch.device('meta'):layer=ns['CloverDecoderLayer'](c,l)
            for name,param in list(layer.named_parameters()):
                if name.startswith('mlp.experts.'):continue
                parent,leaf=name.rsplit('.',1);value=read_tensor(root,ix['tensors'][f'model.layers.{l}.'+name]).cuda()
                layer.get_submodule(parent).register_parameter(leaf,torch.nn.Parameter(value,requires_grad=False))
            routing={};raw_router={}

            class StreamingExperts(torch.nn.Module):
                def forward(self,activations,selected,weights):
                    routing['ids']=selected.cpu().reshape(t,8);routing['weights']=weights.float().cpu();out=torch.zeros_like(activations,dtype=torch.float32)
                    for e in selected.unique(sorted=True).tolist():
                        token,slot=torch.where(selected==e);desc=ix['experts'][l*256+e]
                        gu=read_tensor(root,desc['gate_up'],(1024,2048)).cuda();down=read_tensor(root,desc['down'],(2048,512)).cuda()
                        inputs=activations[token];count=len(inputs)
                        if a.pad_expert_rows:inputs=F.pad(inputs,(0,0,0,32-count))
                        gate,up=F.linear(inputs,gu).chunk(2,-1);hidden=(F.silu(gate.float())*up.float()).bfloat16()
                        y=ns['linear_fp32_output'](hidden,down)[:count];out.index_add_(0,token,(y*weights[token,slot,None].float()).bfloat16().float())
                    return out.bfloat16()

            layer.mlp.experts=StreamingExperts();layer.eval()
            hook=layer.mlp.gate.register_forward_hook(lambda m,inp,out:raw_router.__setitem__('value',out.detach().clone()))
            if c.layer_types[l]=='linear_attention':
                history={'conv':native(a.state/f'layer{l}-conv.bf16').reshape(1,8192,3).cuda(),
                         'ssm':native(a.state/f'layer{l}-ssm.f32').reshape(1,32,128,128).cuda()}
            else:
                history={key:native(a.state/f'layer{l}-{key}.bf16').reshape(1,prefix,2,256).transpose(1,2).cuda() for key in ('key','value')}
            y,_=layer(x,history,valid,prefix_valid,positions,True)
            nr=json.loads((a.trace/f'layer{l}-routes.json').read_text());ni=torch.tensor(nr['ids']).reshape(t,8)
            different=int((ni.sort(-1).values!=routing['ids'].sort(-1).values).any(-1).sum())
            row={'layer':l,'kind':c.layer_types[l],'output_error':error(y,native(a.trace/f'layer{l}-output.bf16')),'different_expert_set_rows':different}
            if different and report['first_expert_set_difference'] is None:report['first_expert_set_difference']=l
            allowed=ns['coreset_torch'](raw_router['value'],valid,8,c.moe_block_coreset_top_p).isfinite()[0,0].nonzero().flatten().cpu().tolist()
            row['coreset_set_equal']=set(allowed)==set(nr['coreset']);row['reference_coreset_experts']=len(allowed);row['native_coreset_experts']=len(nr['coreset'])
            if a.isolate_each_layer:
                local,_=layer(native(a.trace/f'layer{l}-input.bf16').reshape(1,t,2048).cuda(),history,valid,prefix_valid,positions,True)
                row['isolated_output_error']=error(local,native(a.trace/f'layer{l}-output.bf16'))
                row['isolated_different_expert_set_rows']=int((ni.sort(-1).values!=routing['ids'].sort(-1).values).any(-1).sum())
            report['layers'].append(row)
            if row['output_error']['relative_rms_error']>=.004:report['failures'].append(row)
            print(json.dumps({'layer':l,'output_rms':row['output_error']['relative_rms_error'],'sets':different}),flush=True)
            x=y;hook.remove();del layer,history;torch.cuda.empty_cache()
        norm=ns['CloverRMSNorm'](2048,c.rms_norm_eps).bfloat16().cuda();norm.weight.data.copy_(read_tensor(root,ix['tensors']['model.norm.weight']).cuda());x=norm(x)
        desc=ix['tensors']['lm_head.weight'];head=torch.empty(desc['shape'],device='cuda',dtype=torch.bfloat16);flat=head.view(-1)
        with (root/desc['file']).open('rb') as f:
            f.seek(desc['offset']);at=0
            while at<desc['bytes']:
                raw=bytearray(f.read(min(4*1024**2,desc['bytes']-at)));assert raw
                part=torch.frombuffer(raw,dtype=torch.bfloat16);flat[at//2:at//2+len(part)].copy_(part);at+=len(raw)
        logits=F.linear(x,head).float().cpu().reshape(t,c.vocab_size);actual=native(a.trace/'logits.bf16').float().reshape_as(logits)
        report['logits_error']=error(logits,actual);report['different_argmax_positions']=(logits.argmax(-1)!=actual.argmax(-1)).nonzero().flatten().tolist()
        ref_logp=logits.log_softmax(-1);act_logp=actual.log_softmax(-1)
        report['kl_reference_to_native_per_position']=(ref_logp.exp()*(ref_logp-act_logp)).sum(-1).tolist()
        report['reference_top1_probability']=ref_logp.exp().max(-1).values.tolist();report['native_top1_probability']=act_logp.exp().max(-1).values.tolist()
    report['within_numeric_bounds']=not report['failures'] and report['first_expert_set_difference'] is None and report['logits_error']['relative_rms_error']<.004
    report['passed']=report['within_numeric_bounds'] and not a.pad_expert_rows
    (a.output/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ('layers','failures')},indent=2))
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
