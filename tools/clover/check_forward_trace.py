"""Layer-streamed full-checkpoint HF math oracle for one short causal prefill.

Requires a real native 40-layer trace. Materializes one dense decoder layer and
one selected expert at a time; streams the BF16 output head to GPU in 4 MiB
chunks. It never creates an all-expert host or GPU bank.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from types import SimpleNamespace

from test_official_layers import oracle_namespace,read_tensor


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--trace',type=Path,required=True);p.add_argument('--state',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--linear-geometry',choices=('hf','padded32-fp32','padded32-bf16'),default='hf',
                   help='Diagnostic only: pad linear GEMMs to 32 rows with FP32 or BF16 output; default leaves HF operations unchanged')
    p.add_argument('--isolate-each-layer',action='store_true',help='Also compare each layer using the identical native input')
    p.add_argument('--native-router-diagnostic',action='store_true',help='Diagnostic only: feed saved native router logits into HF routing; cannot qualify an unmodified-reference pass')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    index=json.loads(a.index.read_text());assert index['complete'] and index['scope']=='full-model'
    config=json.loads(a.config.read_text());ids=config['input_tokens'];assert 0<len(ids)<=32
    c=SimpleNamespace(**index['config']);root=Path(index['model_root']);ns=oracle_namespace(root/'modeling_clover_moe.py')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    if a.linear_geometry!='hf':
        original_linear=F.linear

        def padded_mm(x,weight):
            flat=x.reshape(-1,x.shape[-1]);assert 0<flat.shape[0]<=32
            padded=F.pad(flat,(0,0,0,32-flat.shape[0]))
            return torch.mm(padded,weight.T,out_dtype=torch.float32)[:flat.shape[0]].reshape(*x.shape[:-1],weight.shape[0])

        def diagnostic_linear(x,weight,bias=None):
            if x.is_cuda and x.dtype==weight.dtype==torch.bfloat16:
                if a.linear_geometry=='padded32-bf16':
                    flat=x.reshape(-1,x.shape[-1]);assert 0<flat.shape[0]<=32
                    value=original_linear(F.pad(flat,(0,0,0,32-flat.shape[0])),weight,bias)
                    return value[:flat.shape[0]].reshape(*x.shape[:-1],weight.shape[0])
                result=padded_mm(x,weight).bfloat16()
                return result if bias is None else result+bias
            return original_linear(x,weight,bias)

        F.linear=diagnostic_linear
        ns['linear_fp32_output']=padded_mm
    t=len(ids);report={'kind':'full-checkpoint-causal-prefill-layer-streamed-oracle','revision':index['revision'],
        'sdpa_backend':'math','linear_geometry':a.linear_geometry,'diagnostic_modified_gemm':a.linear_geometry!='hf',
        'tokens':ids,'layers':[],'failures':[],'first_router_difference':None,'first_router_set_difference':None,
        'diagnostic_native_router_logits':a.native_router_diagnostic}

    def native(path):
        raw=path.read_bytes();fp32=path.suffix=='.f32'
        out=torch.from_numpy(np.frombuffer(raw,dtype=np.float32 if fp32 else np.uint16).copy())
        return out if fp32 else out.view(torch.bfloat16)

    def error(ref,actual):
        ref=ref.detach().float().cpu().flatten();actual=actual.float().flatten();assert ref.shape==actual.shape
        d=(ref-actual).abs();rms=((ref-actual).square().mean().sqrt()/ref.square().mean().sqrt().clamp_min(1e-12)).item() if ref.numel() else 0
        return {'max_abs_error':d.max().item() if d.numel() else 0,'mean_abs_error':d.mean().item() if d.numel() else 0,
                'relative_rms_error':rms,'different_components':int((ref!=actual).sum())}

    emb=index['tensors']['model.embed_tokens.weight'];pieces=[]
    for token in ids:pieces.append(read_tensor(root,{'file':emb['file'],'offset':emb['offset']+token*4096,'bytes':4096},(2048,)))
    x=torch.stack(pieces)[None].cuda();assert torch.equal(x.cpu().flatten(),native(a.trace/'embedding.bf16'))
    valid=torch.ones(1,t,device='cuda',dtype=torch.bool);empty=torch.ones(1,0,device='cuda',dtype=torch.bool)
    positions=torch.arange(t,device='cuda')[None]
    state_manifest=json.loads((a.state/'manifest.json').read_text());assert state_manifest['prefix_tokens']==t
    with torch.inference_mode(),torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
        for l in range(c.num_hidden_layers):
            with torch.device('meta'):layer=ns['CloverDecoderLayer'](c,l)
            for name,param in list(layer.named_parameters()):
                if name.startswith('mlp.experts.'):continue
                parent,leaf=name.rsplit('.',1);desc=index['tensors'][f'model.layers.{l}.'+name]
                layer.get_submodule(parent).register_parameter(leaf,torch.nn.Parameter(read_tensor(root,desc).cuda(),requires_grad=False))
            routing={}

            class StreamingExperts(torch.nn.Module):
                def forward(self,activations,ids,weights):
                    routing['ids']=ids.detach().cpu().flatten();routing['weights']=weights.detach().cpu().flatten()
                    out=torch.zeros_like(activations,dtype=torch.float32)
                    for e in ids.unique(sorted=True).tolist():
                        token,slot=torch.where(ids==e);entry=index['experts'][l*c.num_experts+e]
                        gu=read_tensor(root,entry['gate_up'],(1024,2048)).cuda();down=read_tensor(root,entry['down'],(2048,512)).cuda()
                        gate,up=F.linear(activations[token],gu).chunk(2,-1)
                        middle=(F.silu(gate.float())*up.float()).bfloat16()
                        y=ns['linear_fp32_output'](middle,down)
                        out.index_add_(0,token,(y*weights[token,slot,None].float()).bfloat16().float())
                    return out.bfloat16()

            layer.mlp.experts=StreamingExperts();layer.eval()
            traces={};hooks=[]
            for key,module in [('input_norm',layer.input_layernorm),('post_norm',layer.post_attention_layernorm),('router',layer.mlp.gate)]:
                hooks.append(module.register_forward_hook(lambda m,inp,out,key=key:traces.__setitem__(key,out.detach().clone())))
            if a.native_router_diagnostic:
                hooks.append(layer.mlp.gate.register_forward_hook(lambda m,inp,out,l=l:native(a.trace/f'layer{l}-router.bf16').reshape_as(out).to(out.device)))
            mixer=layer.linear_attn if hasattr(layer,'linear_attn') else layer.self_attn
            hooks.append(mixer.register_forward_hook(lambda m,inp,out:traces.__setitem__('mixer',out[0].detach().clone())))
            if hasattr(layer,'linear_attn'):
                hooks.append(mixer.in_proj_qkv.register_forward_hook(lambda m,inp,out:traces.__setitem__('gdn_raw',out.detach().clone())))
                hooks.append(mixer.norm.register_forward_pre_hook(lambda m,inp:traces.__setitem__('gdn_y',inp[0].detach().clone())))
                hooks.append(mixer.norm.register_forward_hook(lambda m,inp,out:traces.__setitem__('gdn_normalized',out.detach().clone())))
            native_input=native(a.trace/f'layer{l}-input.bf16').reshape(1,t,c.hidden_size)
            row={'layer':l,'kind':c.layer_types[l],'input_error':error(x,native_input)}
            y,cache=layer(x,None,valid,empty,positions,False)
            row['output_error']=error(y,native(a.trace/f'layer{l}-output.bf16'))
            if l==0:
                row['first_layer_operations']={key:error(value,native(a.trace/f'layer{l}-{key}.bf16')) for key,value in traces.items()}
            nr=json.loads((a.trace/f'layer{l}-routes.json').read_text());ni=torch.tensor(nr['ids'])
            row['different_route_slots']=int((ni!=routing['ids']).sum())
            ni_rows=ni.reshape(t,c.num_experts_per_tok);ri_rows=routing['ids'].reshape_as(ni_rows)
            row['different_expert_set_rows']=int((ni_rows.sort(-1).values!=ri_rows.sort(-1).values).any(-1).sum())
            if row['different_expert_set_rows'] and report['first_router_set_difference'] is None:report['first_router_set_difference']=l
            row['router_cutoff_diagnostic']=[]
            actual_router=traces['router'].float().cpu().reshape(t,c.num_experts)
            native_router=native(a.trace/f'layer{l}-router.bf16').float().reshape_as(actual_router)
            for position in range(t):
                if torch.equal(ni_rows[position].sort().values,ri_rows[position].sort().values):continue
                scores=actual_router[position].sort(descending=True).values
                row['router_cutoff_diagnostic'].append({'position':position,'native_experts':ni_rows[position].tolist(),'reference_experts':ri_rows[position].tolist(),
                    'reference_top8_top9_logit_margin':float(scores[c.num_experts_per_tok-1]-scores[c.num_experts_per_tok]),
                    'router_max_abs_error':float((native_router[position]-actual_router[position]).abs().max())})
            (a.output/f'layer{l}-reference-router.bf16').write_bytes(traces['router'].cpu().view(torch.uint16).numpy().tobytes())
            (a.output/f'layer{l}-reference-routes.json').write_text(json.dumps({'ids':routing['ids'].tolist(),'weights':routing['weights'].tolist()},indent=2))
            first_difference=row['different_route_slots'] and report['first_router_difference'] is None
            if first_difference:
                report['first_router_difference']=l
            if first_difference or a.isolate_each_layer:
                local_y,_=layer(native_input.cuda(),None,valid,empty,positions,False)
                row['isolated_native_input_output_error']=error(local_y,native(a.trace/f'layer{l}-output.bf16'))
                row['isolated_different_route_slots']=int((ni!=routing['ids']).sum())
                row['isolated_different_expert_set_rows']=int((ni_rows.sort(-1).values!=routing['ids'].reshape_as(ni_rows).sort(-1).values).any(-1).sum())
                row['isolated_operations']={key:error(value,native(a.trace/f'layer{l}-{key}.bf16')) for key,value in traces.items()}
            if c.layer_types[l]=='linear_attention':
                row['conv_state_error']=error(cache['conv'],native(a.state/f'layer{l}-conv.bf16'))
                row['ssm_state_error']=error(cache['ssm'],native(a.state/f'layer{l}-ssm.f32'))
            else:
                row['key_state_error']=error(cache['key'].transpose(1,2),native(a.state/f'layer{l}-key.bf16'))
                row['value_state_error']=error(cache['value'].transpose(1,2),native(a.state/f'layer{l}-value.bf16'))
            report['layers'].append(row)
            if row['output_error']['relative_rms_error']>=0.004:report['failures'].append({'layer':l,'output_error':row['output_error']})
            print(json.dumps({'layer':l,'output_rms':row['output_error']['relative_rms_error'],'different_route_slots':row['different_route_slots']}),flush=True)
            x=y
            for hook in hooks:hook.remove()
            del layer,traces;torch.cuda.empty_cache()
        norm=ns['CloverRMSNorm'](c.hidden_size,c.rms_norm_eps).bfloat16().cuda()
        norm.weight.data.copy_(read_tensor(root,index['tensors']['model.norm.weight']).cuda());x=norm(x)
        report['final_norm_error']=error(x,native(a.trace/'final_norm.bf16'))
        desc=index['tensors']['lm_head.weight'];head=torch.empty(desc['shape'],device='cuda',dtype=torch.bfloat16)
        flat=head.view(-1)
        with (root/desc['file']).open('rb') as f:
            f.seek(desc['offset']);at=0
            while at<desc['bytes']:
                raw=bytearray(f.read(min(4*1024**2,desc['bytes']-at)));assert raw
                part=torch.frombuffer(raw,dtype=torch.bfloat16);flat[at//2:at//2+part.numel()].copy_(part);at+=len(raw)
        logits=F.linear(x,head).float().cpu().reshape(t,c.vocab_size)
        actual=native(a.trace/'logits.bf16').float().reshape_as(logits)
        report['logits_error']=error(logits,actual)
        report['centered_logits_error']=error(logits-logits.mean(-1,keepdim=True),actual-actual.mean(-1,keepdim=True))
        (a.output/'reference-logits.bf16').write_bytes(logits.bfloat16().view(torch.uint16).numpy().tobytes())
        (a.output/'native-logits.bf16').write_bytes(actual.bfloat16().view(torch.uint16).numpy().tobytes())
        ref_logp=logits.log_softmax(-1);native_logp=actual.log_softmax(-1);ref_p=ref_logp.exp();native_p=native_logp.exp()
        kl=(ref_p*(ref_logp-native_logp)).sum(-1);tv=(ref_p-native_p).abs().sum(-1)/2
        report['probability_comparison']={'kl_reference_to_native_per_position':kl.tolist(),'total_variation_per_position':tv.tolist(),
            'reference_top1_probability_per_position':ref_p.max(-1).values.tolist(),'native_top1_probability_per_position':native_p.max(-1).values.tolist()}
        report['native_argmax']=actual.argmax(-1).tolist();report['reference_argmax']=logits.argmax(-1).tolist()
        report['last_token_argmax_equal']=report['native_argmax'][-1]==report['reference_argmax'][-1]
        report['different_argmax_positions']=[i for i,(x,y) in enumerate(zip(report['native_argmax'],report['reference_argmax'])) if x!=y]
        top=logits[-1].topk(2).values;report['reference_last_top1_margin']=float(top[0]-top[1])
        top=actual[-1].topk(2).values;report['native_last_top1_margin']=float(top[0]-top[1])
    report['within_tensor_tolerances']=not report['failures']
    report['layer_output_relative_rms_limit']=0.004
    report['within_numeric_bounds']=report['within_tensor_tolerances'] and report['first_router_difference'] is None and report['logits_error']['relative_rms_error']<0.004
    report['passed']=report['within_numeric_bounds'] and not report['diagnostic_modified_gemm'] and not a.native_router_diagnostic
    report['unmodified_reference_pass']=report['passed']
    report['note']='Routing/argmax differences are reported separately from tensor error bounds; this does not test HF block denoising equivalence.'
    (a.output/'summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('layers','failures')},indent=2),flush=True)
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
