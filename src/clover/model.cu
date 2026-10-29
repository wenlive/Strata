#include "model.hpp"
#include <set>

namespace strata::clover {
Json counter_delta(const Json& after,const Json& before) {
    Json out;
    for(const char* key:{"gpu_hits","gpu_misses","ram_hits","ram_misses","logical_file_read_bytes",
                         "aligned_file_requested_bytes","aligned_file_completed_bytes","read_calls","h2d_bytes","d2h_bytes"})
        out[key]=after.at(key).get<uint64_t>()-before.at(key).get<uint64_t>();
    for(const char* key:{"read_ms","host_staging_wait_ms"}) out[key]=after.at(key).get<double>()-before.at(key).get<double>();
    return out;
}
const BF* Model::weight(const std::string& name) const {return weights_.at(name).as<BF>();}
void Model::linear(const std::string& name,const BF* x,BF* y,int t) {
    const auto& shape=store_.index.at("tensors").at(name).at("shape");
    require(shape.size()==2,"Linear weight is not a matrix: "+name);
    int n=shape.at(0),k=shape.at(1);
    require(size_t(max_tokens_)*n*4<=fp32_.bytes && size_t(max_tokens_)*k*2<=dense_input_.bytes,"Dense GEMM workspace overflow");
    // Keep cuBLAS's row geometry fixed across prefill/verify/replay/decode.
    // BF16 rounding at router ties must not depend on a speculative batch's
    // acceptance length. Padding is never passed to attention, routing or state.
    const BF* input=x;
    if(t<max_tokens_) {
        CLOVER_CUDA(cudaMemcpyAsync(dense_input_.p,x,size_t(t)*k*2,cudaMemcpyDeviceToDevice,stream_));
        CLOVER_CUDA(cudaMemsetAsync(dense_input_.as<BF>()+size_t(t)*k,0,size_t(max_tokens_-t)*k*2,stream_));
        input=dense_input_.as<BF>();
    }
    dense_.bf16(reinterpret_cast<const uint16_t*>(input),reinterpret_cast<const uint16_t*>(weight(name)),fp32_.as<float>(),max_tokens_,n,k);
    ops::cast(fp32_.as<float>(),y,t*n,stream_);
}
Model::Model(ExpertStore& source,int context,int max_tokens,cudaStream_t stream,bool layer_test)
    :store_(source),context_(context),max_tokens_(max_tokens),stream_(stream),experts_(source,max_tokens,stream) {
    require(source.index.at("scope")=="full-model"||(layer_test&&source.index.at("scope")=="selected-official-layer-test"),"Full model requires a complete official model index");
    config=source.index.at("config");
    h_=config.at("hidden_size");ff_=config.at("moe_intermediate_size");
    kh_=config.at("linear_num_key_heads");vh_=config.at("linear_num_value_heads");
    kd_=config.at("linear_key_head_dim");vd_=config.at("linear_value_head_dim");
    nh_=config.at("num_attention_heads");nkv_=config.at("num_key_value_heads");hd_=config.at("head_dim");
    sf_=config.at("shared_expert_intermediate_size");top_k_=config.at("num_experts_per_tok");
    vocab=config.at("vocab_size");block_size=config.at("block_size");ch_=2*kh_*kd_+vh_*vd_;
    require(h_==2048&&ff_==512&&kh_==16&&vh_==32&&kd_==128&&vd_==128&&nh_==16&&nkv_==2&&hd_==256&&sf_==512&&block_size==32&&(store_.layers==40||(layer_test&&store_.layers==1))&&store_.experts==256,"Unsupported Clover checkpoint geometry");
    require(!config.at("attention_bias").get<bool>()&&config.at("attn_output_gate").get<bool>()&&config.at("num_shared_experts")==1&&!config.at("tie_word_embeddings").get<bool>(),"Unsupported Clover architecture option");
    require(config.at("linear_conv_kernel_dim")==4 && config.at("rope_parameters").at("rope_type")=="default","Unsupported convolution/RoPE variant");
    require(context>0&&context<=8192&&max_tokens>0&&max_tokens<=32,"First Clover driver supports context <=8192, batch <=32");
    uint64_t dense_bytes=0;
    for(const auto& item:source.index.at("tensors").items()) if(item.key().find(".experts.")==std::string::npos) dense_bytes+=item.value().at("bytes").get<uint64_t>();
    size_t free=0,total=0;CLOVER_CUDA(cudaMemGetInfo(&free,&total));
    const uint64_t state_bytes=30ull*(uint64_t(ch_)*3*2+uint64_t(vh_)*vd_*kd_*4)+10ull*uint64_t(context)*nkv_*hd_*4;
    require(dense_bytes+state_bytes+(512ull<<20)+(1024ull<<20)<free,"Insufficient VRAM for complete dense BF16 weights, states, workspaces and 1 GiB headroom; reduce expert VRAM cache/context");
    for(const auto& item:source.index.at("tensors").items()) if(item.key().find(".experts.")==std::string::npos)
        weights_.emplace(item.key(),source.load_tensor(item.key()));
    std::string error;require(dense_.init(stream,0,error),error);
    auto alloc=[&](DeviceBuffer& b,uint64_t elements,int bytes=2){b.allocate(size_t(elements)*bytes);};
    const uint64_t t=uint64_t(max_tokens);
    for(DeviceBuffer* b:{&x_,&xn_,&mixer_,&expert_out_,&shared_out_}) alloc(*b,t*h_);
    for(DeviceBuffer* b:{&shared_gate_,&shared_up_,&shared_mid_}) alloc(*b,t*sf_);
    alloc(shared_scale_,t);alloc(router_,t*store_.experts);alloc(ids_,t,4);
    alloc(fp32_,t*std::max({vocab,ch_,2*nh_*hd_}),4);alloc(logits_,t*vocab);
    alloc(dense_input_,t*std::max({h_,vh_*vd_,nh_*hd_,sf_}));
    alloc(raw_,t*ch_);alloc(mixed_,t*ch_);alloc(qf_,t*kh_*kd_,4);alloc(kf_,t*kh_*kd_,4);
    for(DeviceBuffer* b:{&z_,&features_,&features_norm_,&qatt_}) alloc(*b,t*vh_*vd_);
    for(DeviceBuffer* b:{&a_,&b_,&beta_}) alloc(*b,t*vh_);alloc(g_,t*vh_,4);
    for(DeviceBuffer* b:{&knew_,&katt_,&vnew_}) alloc(*b,t*nkv_*hd_);
    state_.resize(size_t(store_.layers));
    for(int l=0;l<store_.layers;++l) {
        auto& s=state_[size_t(l)];s.linear=config.at("layer_types").at(l)=="linear_attention";
        if(s.linear) {alloc(s.conv,uint64_t(ch_)*3);alloc(s.ssm,uint64_t(vh_)*vd_*kd_,4);}
        else {alloc(s.key,uint64_t(context)*nkv_*hd_);alloc(s.value,uint64_t(context)*nkv_*hd_);}
    }
    reset();CLOVER_CUDA(cudaStreamSynchronize(stream_));
}
void Model::reset() {
    prefix=0;
    for(auto& s:state_) for(DeviceBuffer* b:{&s.conv,&s.ssm,&s.key,&s.value})
        if(b->bytes) CLOVER_CUDA(cudaMemsetAsync(b->p,0,b->bytes,stream_));
    CLOVER_CUDA(cudaStreamSynchronize(stream_));
}
ForwardResult Model::forward(const std::vector<int>& tokens,bool preview,bool commit,const std::string& phase) {
    const int t=int(tokens.size());require(t>0&&t<=max_tokens_&&prefix+t<=context_,"Forward exceeds batch/context budget");
    require(!preview||(!commit&&t<=block_size&&block_size%t==0),"Preview must be read-only and width must divide checkpoint block size");
    for(int token:tokens) require(token>=0&&token<vocab,"Token id out of vocabulary");
    const auto before=store_.stats();const auto start=Clock::now();
    if(!trace_root_.empty()) {require(!std::filesystem::exists(trace_root_),"Trace output already exists");std::filesystem::create_directories(trace_root_);}
    auto trace=[&](const std::string& name,const void* device,size_t bytes) {
        if(trace_root_.empty()) return;
        CLOVER_CUDA(cudaStreamSynchronize(stream_));std::vector<uint8_t> host(bytes);
        CLOVER_CUDA(cudaMemcpy(host.data(),device,bytes,cudaMemcpyDeviceToHost));
        std::ofstream file(trace_root_/name,std::ios::binary);file.write(reinterpret_cast<const char*>(host.data()),std::streamsize(bytes));require(bool(file),"Trace write failed");
    };
    CLOVER_CUDA(cudaMemcpyAsync(ids_.p,tokens.data(),size_t(t)*4,cudaMemcpyHostToDevice,stream_));
    ops::embedding(weight("model.embed_tokens.weight"),ids_.as<int>(),x_.as<BF>(),t,h_,stream_);
    trace("embedding.bf16",x_.p,size_t(t)*h_*2);
    Json uniques=Json::array(),coresets=Json::array();
    const float eps=config.at("rms_norm_eps");
    for(int l=0;l<store_.layers;++l) {
        const std::string p="model.layers."+std::to_string(l)+".";auto& s=state_[size_t(l)];
        trace("layer"+std::to_string(l)+"-input.bf16",x_.p,size_t(t)*h_*2);
        ops::norm(x_.as<BF>(),weight(p+"input_layernorm.weight"),nullptr,xn_.as<BF>(),t,h_,eps,true,stream_);
        const std::string trace_pre="layer"+std::to_string(l)+"-";
        trace(trace_pre+"input_norm.bf16",xn_.p,size_t(t)*h_*2);
        if(s.linear) {
            const std::string a=p+"linear_attn.";
            linear(a+"in_proj_qkv.weight",xn_.as<BF>(),raw_.as<BF>(),t);
            linear(a+"in_proj_z.weight",xn_.as<BF>(),z_.as<BF>(),t);
            linear(a+"in_proj_a.weight",xn_.as<BF>(),a_.as<BF>(),t);
            linear(a+"in_proj_b.weight",xn_.as<BF>(),b_.as<BF>(),t);
            ops::gdn_conv(raw_.as<BF>(),s.conv.as<BF>(),weight(a+"conv1d.weight"),weight(a+"conv_right_weight"),mixed_.as<BF>(),t,ch_,4,preview,stream_);
            ops::gdn_qk(mixed_.as<BF>(),qf_.as<float>(),kf_.as<float>(),t,kh_,vh_,kd_,vd_,stream_);
            ops::gdn_factors(a_.as<BF>(),b_.as<BF>(),weight(a+"A_log"),weight(a+"dt_bias"),g_.as<float>(),beta_.as<BF>(),t,vh_,stream_);
            if(preview) ops::gdn_preview(mixed_.as<BF>(),qf_.as<float>(),kf_.as<float>(),g_.as<float>(),weight(a+"bd_softmax_logit_scale"),weight(a+"bd_softmax_history_bias"),weight(a+"bd_softmax_history_c"),s.ssm.as<float>(),features_.as<BF>(),t,kh_,vh_,kd_,vd_,block_size,stream_);
            else ops::gdn_scan(mixed_.as<BF>(),qf_.as<float>(),kf_.as<float>(),g_.as<float>(),beta_.as<BF>(),s.ssm.as<float>(),commit?s.ssm.as<float>():nullptr,features_.as<BF>(),t,kh_,vh_,kd_,vd_,stream_);
            ops::norm(features_.as<BF>(),weight(a+"norm.weight"),z_.as<BF>(),features_norm_.as<BF>(),t*vh_,vd_,eps,false,stream_);
            trace(trace_pre+"gdn_raw.bf16",raw_.p,size_t(t)*ch_*2);
            trace(trace_pre+"gdn_mixed.bf16",mixed_.p,size_t(t)*ch_*2);
            trace(trace_pre+"gdn_y.bf16",features_.p,size_t(t)*vh_*vd_*2);
            trace(trace_pre+"gdn_normalized.bf16",features_norm_.p,size_t(t)*vh_*vd_*2);
            linear(a+"out_proj.weight",features_norm_.as<BF>(),mixer_.as<BF>(),t);
            if(commit) ops::gdn_commit_conv(raw_.as<BF>(),s.conv.as<BF>(),t,ch_,4,stream_);
        } else {
            const std::string a=p+"self_attn.";
            linear(a+"q_proj.weight",xn_.as<BF>(),raw_.as<BF>(),t);
            linear(a+"k_proj.weight",xn_.as<BF>(),knew_.as<BF>(),t);
            linear(a+"v_proj.weight",xn_.as<BF>(),vnew_.as<BF>(),t);
            const int rd=int(hd_*config.at("partial_rotary_factor").get<float>());
            const float theta=config.at("rope_parameters").at("rope_theta");
            ops::prepare_q(raw_.as<BF>(),weight(a+"q_norm.weight"),qatt_.as<BF>(),z_.as<BF>(),t,nh_,hd_,rd,prefix,theta,eps,stream_);
            ops::prepare_k(knew_.as<BF>(),weight(a+"k_norm.weight"),katt_.as<BF>(),t,nkv_,hd_,rd,prefix,theta,eps,stream_);
            ops::attention(qatt_.as<BF>(),katt_.as<BF>(),vnew_.as<BF>(),s.key.as<BF>(),s.value.as<BF>(),features_.as<BF>(),t,prefix,nh_,nkv_,hd_,preview,stream_);
            ops::sigmoid_mul(features_.as<BF>(),z_.as<BF>(),features_.as<BF>(),t,nh_*hd_,false,stream_);
            linear(a+"o_proj.weight",features_.as<BF>(),mixer_.as<BF>(),t);
            if(commit) {
                const size_t at=size_t(prefix)*nkv_*hd_,bytes=size_t(t)*nkv_*hd_*2;
                CLOVER_CUDA(cudaMemcpyAsync(s.key.as<BF>()+at,katt_.p,bytes,cudaMemcpyDeviceToDevice,stream_));
                CLOVER_CUDA(cudaMemcpyAsync(s.value.as<BF>()+at,vnew_.p,bytes,cudaMemcpyDeviceToDevice,stream_));
            }
        }
        trace(trace_pre+"mixer.bf16",mixer_.p,size_t(t)*h_*2);
        ops::residual(x_.as<BF>(),mixer_.as<BF>(),x_.as<BF>(),t*h_,stream_);
        ops::norm(x_.as<BF>(),weight(p+"post_attention_layernorm.weight"),nullptr,xn_.as<BF>(),t,h_,eps,true,stream_);
        trace(trace_pre+"post_norm.bf16",xn_.p,size_t(t)*h_*2);
        linear(p+"mlp.gate.weight",xn_.as<BF>(),router_.as<BF>(),t);
        std::vector<BF> router_bf(size_t(t)*store_.experts);
        CLOVER_CUDA(cudaMemcpyAsync(router_bf.data(),router_.p,router_bf.size()*2,cudaMemcpyDeviceToHost,stream_));
        CLOVER_CUDA(cudaStreamSynchronize(stream_));
        std::vector<float> router_float(router_bf.size());for(size_t i=0;i<router_bf.size();++i) router_float[i]=__bfloat162float(router_bf[i]);
        auto routes=route(router_float,t,store_.experts,top_k_,preview&&config.at("moe_block_coreset_enabled").get<bool>(),config.at("moe_block_coreset_top_p"));
        trace(trace_pre+"router.bf16",router_.p,size_t(t)*store_.experts*2);
        if(!trace_root_.empty()) {std::ofstream out(trace_root_/(trace_pre+"routes.json"));out<<Json{{"ids",routes.ids},{"weights",routes.weights},{"coreset",routes.coreset}}.dump(2)<<'\n';}
        std::set<int> used(routes.ids.begin(),routes.ids.end());uniques.push_back(used.size());coresets.push_back(routes.coreset.size());
        experts_.run(l,xn_.as<BF>(),expert_out_.as<BF>(),t,top_k_,routes.ids,routes.weights);
        const std::string shared=p+"mlp.shared_expert.";
        linear(shared+"gate_proj.weight",xn_.as<BF>(),shared_gate_.as<BF>(),t);
        linear(shared+"up_proj.weight",xn_.as<BF>(),shared_up_.as<BF>(),t);
        ops::swiglu(shared_gate_.as<BF>(),shared_up_.as<BF>(),shared_mid_.as<BF>(),t*sf_,stream_);
        linear(shared+"down_proj.weight",shared_mid_.as<BF>(),shared_out_.as<BF>(),t);
        linear(p+"mlp.shared_expert_gate.weight",xn_.as<BF>(),shared_scale_.as<BF>(),t);
        ops::sigmoid_mul(shared_out_.as<BF>(),shared_scale_.as<BF>(),shared_out_.as<BF>(),t,h_,true,stream_);
        ops::residual(expert_out_.as<BF>(),shared_out_.as<BF>(),expert_out_.as<BF>(),t*h_,stream_);
        ops::residual(x_.as<BF>(),expert_out_.as<BF>(),x_.as<BF>(),t*h_,stream_);
        trace(trace_pre+"output.bf16",x_.p,size_t(t)*h_*2);
    }
    ops::norm(x_.as<BF>(),weight("model.norm.weight"),nullptr,xn_.as<BF>(),t,h_,eps,true,stream_);
    trace("final_norm.bf16",xn_.p,size_t(t)*h_*2);
    linear("lm_head.weight",xn_.as<BF>(),logits_.as<BF>(),t);
    trace("logits.bf16",logits_.p,size_t(t)*vocab*2);
    std::vector<BF> logits_bf(size_t(t)*vocab);
    CLOVER_CUDA(cudaMemcpyAsync(logits_bf.data(),logits_.p,logits_bf.size()*2,cudaMemcpyDeviceToHost,stream_));
    CLOVER_CUDA(cudaGetLastError());CLOVER_CUDA(cudaStreamSynchronize(stream_));
    ForwardResult result;result.logits.resize(logits_bf.size());
    for(size_t i=0;i<logits_bf.size();++i) result.logits[i]=__bfloat162float(logits_bf[i]);
    const int prefix_before=prefix;if(commit) prefix+=t;
    const uint64_t control_h2d=uint64_t(t)*4+uint64_t(store_.layers)*t*top_k_*8;
    result.measurement={{"phase",phase},{"input_tokens",t},{"preview",preview},{"commit",commit},
        {"prefix_before",prefix_before},{"prefix_after",prefix},{"elapsed_ms",milliseconds(start)},
        {"experts_unique_per_layer",uniques},{"coreset_size_per_layer",coresets},
        {"cache_io",counter_delta(store_.stats(),before)},{"control_h2d_bytes",control_h2d},{"memory",memory_snapshot()}};
    if(!trace_root_.empty()) result.measurement["trace_directory"]=trace_root_.string();
    std::cout<<"PHASE "<<result.measurement.dump()<<std::endl;return result;
}
Json Model::allocation() const {
    uint64_t weights=0,states=0,work=0;
    for(const auto& w:weights_) weights+=w.second.bytes;
    for(const auto& s:state_) states+=s.conv.bytes+s.ssm.bytes+s.key.bytes+s.value.bytes;
    for(const DeviceBuffer* b:{&x_,&xn_,&mixer_,&expert_out_,&shared_gate_,&shared_up_,&shared_mid_,&shared_out_,&shared_scale_,&router_,&ids_,&fp32_,&logits_,&dense_input_,&raw_,&mixed_,&qf_,&kf_,&z_,&a_,&b_,&g_,&beta_,&features_,&features_norm_,&qatt_,&knew_,&katt_,&vnew_}) work+=b->bytes;
    return {{"dense_bf16_weight_bytes",weights},{"persistent_state_capacity_bytes",states},{"gemm_compute_rows",max_tokens_},
            {"model_device_workspace_bytes",work},{"dense_gemm_workspace_bytes",32u<<20},
            {"expert_workspace",experts_.workspace()},{"max_logits_host_bf16_bytes",uint64_t(max_tokens_)*vocab*2},
            {"max_logits_host_float_bytes",uint64_t(max_tokens_)*vocab*4},{"store",store_.stats()}};
}
uint64_t Model::state_hash() {
    CLOVER_CUDA(cudaStreamSynchronize(stream_));uint64_t hash=1469598103934665603ull;
    std::vector<uint8_t> scratch(1u<<20);
    for(const auto& s:state_) for(const DeviceBuffer* buffer:{&s.conv,&s.ssm,&s.key,&s.value}) {
        if(!buffer->bytes) continue;
        const size_t bytes=(buffer==&s.key||buffer==&s.value)?size_t(prefix)*nkv_*hd_*2:buffer->bytes;
        for(size_t at=0;at<bytes;) {
            const size_t n=std::min(scratch.size(),bytes-at);
            CLOVER_CUDA(cudaMemcpy(scratch.data(),buffer->as<uint8_t>()+at,n,cudaMemcpyDeviceToHost));
            for(size_t i=0;i<n;++i) {hash^=scratch[i];hash*=1099511628211ull;}at+=n;
        }
    }
    return hash;
}
void Model::save_state(const std::filesystem::path& root) {
    require(!std::filesystem::exists(root),"State output already exists");std::filesystem::create_directories(root);
    CLOVER_CUDA(cudaStreamSynchronize(stream_));Json manifest={{"prefix_tokens",prefix},{"layers",Json::array()}};
    std::vector<uint8_t> scratch(1u<<20);
    for(size_t l=0;l<state_.size();++l) {
        const auto& s=state_[l];Json layer={{"layer",l}};
        for(const auto& named:std::vector<std::pair<std::string,const DeviceBuffer*>>{{"conv.bf16",&s.conv},{"ssm.f32",&s.ssm},{"key.bf16",&s.key},{"value.bf16",&s.value}}) {
            if(!named.second->bytes) continue;
            const auto file=root/("layer"+std::to_string(l)+"-"+named.first);
            const size_t bytes=named.first=="key.bf16"||named.first=="value.bf16"?size_t(prefix)*nkv_*hd_*2:named.second->bytes;
            std::ofstream out(file,std::ios::binary);
            for(size_t at=0;at<bytes;) {const size_t n=std::min(scratch.size(),bytes-at);CLOVER_CUDA(cudaMemcpy(scratch.data(),named.second->as<uint8_t>()+at,n,cudaMemcpyDeviceToHost));out.write(reinterpret_cast<const char*>(scratch.data()),std::streamsize(n));at+=n;}
            require(bool(out),"State dump write failed");layer[named.first]={{"file",file.filename().string()},{"bytes",bytes}};
        }
        manifest["layers"].push_back(layer);
    }
    std::ofstream out(root/"manifest.json");out<<manifest.dump(2)<<'\n';
}
}
