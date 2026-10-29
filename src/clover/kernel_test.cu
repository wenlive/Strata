#include "kernels.hpp"
#include "router.hpp"
#include "speculation.hpp"
#include <cstring>

namespace {
using namespace strata::clover;
DeviceBuffer load(const Json& desc) {
    size_t bytes=desc.at("bytes");require(bytes<=128ull*1024*1024,"Test input too large");
    std::ifstream file(desc.at("file").get<std::string>(),std::ios::binary|std::ios::ate);
    require(bool(file)&&size_t(file.tellg())==bytes,"Wrong test file length");file.seekg(0);
    std::vector<uint8_t> host(bytes);file.read(reinterpret_cast<char*>(host.data()),std::streamsize(bytes));
    require(bool(file),"Cannot read test input");DeviceBuffer out(bytes);
    CLOVER_CUDA(cudaMemcpy(out.p,host.data(),bytes,cudaMemcpyHostToDevice));return out;
}
void dump(const DeviceBuffer& buffer,const std::filesystem::path& file) {
    std::vector<uint8_t> host(buffer.bytes);CLOVER_CUDA(cudaMemcpy(host.data(),buffer.p,buffer.bytes,cudaMemcpyDeviceToHost));
    std::ofstream output(file,std::ios::binary);output.write(reinterpret_cast<const char*>(host.data()),std::streamsize(host.size()));
    require(bool(output),"Cannot write test output");
}
std::vector<uint8_t> bytes(const DeviceBuffer& buffer) {
    std::vector<uint8_t> host(buffer.bytes);CLOVER_CUDA(cudaMemcpy(host.data(),buffer.p,buffer.bytes,cudaMemcpyDeviceToHost));return host;
}
}
int main(int argc,char** argv) {
    using namespace strata::clover;
    try {
        require(argc==3,"Usage: strata-clover-kernel-test JOB.json OUTPUT_DIRECTORY");
        auto job=read_json(argv[1]);std::filesystem::path root=argv[2];
        require(!std::filesystem::exists(root),"Test output directory already exists");std::filesystem::create_directories(root);
        cudaStream_t stream;CLOVER_CUDA(cudaStreamCreate(&stream));Json report;
        if(job.contains("gdn")) {
            auto c=job.at("gdn");int kh=c.at("kh"),vh=c.at("vh"),kd=128,vd=128,ch=2*kh*kd+vh*vd,taps=4;
            auto& inputs=c.at("inputs");auto history=load(inputs.at("history")),state=load(inputs.at("state"));
            auto left=load(inputs.at("left")),right=load(inputs.at("right")),scale=load(inputs.at("scale"));
            auto bias=load(inputs.at("bias")),hc=load(inputs.at("history_c")),al=load(inputs.at("A_log"));
            auto dt=load(inputs.at("dt_bias")),norm=load(inputs.at("norm"));int step_id=0;
            for(const auto& step:c.at("steps")) {
                int t=step.at("tokens");bool preview=step.at("preview"),commit=step.at("commit");
                require(!preview||!commit,"Preview must be read-only");
                auto raw=load(step.at("raw")),a=load(step.at("a")),beta_in=load(step.at("b")),z=load(step.at("z"));
                DeviceBuffer mixed(size_t(t)*ch*2),q(size_t(t)*kh*kd*4),k(size_t(t)*kh*kd*4),
                    g(size_t(t)*vh*4),beta(size_t(t)*vh*2),y(size_t(t)*vh*vd*2),normalized(size_t(t)*vh*vd*2);
                const auto previous_history=bytes(history),previous_state=bytes(state);
                ops::gdn_factors(a.as<BF>(),beta_in.as<BF>(),al.as<BF>(),dt.as<BF>(),g.as<float>(),beta.as<BF>(),t,vh,stream);
                ops::gdn_conv(raw.as<BF>(),history.as<BF>(),left.as<BF>(),right.as<BF>(),mixed.as<BF>(),t,ch,taps,preview,stream);
                ops::gdn_qk(mixed.as<BF>(),q.as<float>(),k.as<float>(),t,kh,vh,kd,vd,stream);
                if(preview) ops::gdn_preview(mixed.as<BF>(),q.as<float>(),k.as<float>(),g.as<float>(),scale.as<BF>(),bias.as<BF>(),hc.as<BF>(),state.as<float>(),y.as<BF>(),t,kh,vh,kd,vd,32,stream);
                else ops::gdn_scan(mixed.as<BF>(),q.as<float>(),k.as<float>(),g.as<float>(),beta.as<BF>(),state.as<float>(),commit?state.as<float>():nullptr,y.as<BF>(),t,kh,vh,kd,vd,stream);
                ops::norm(y.as<BF>(),norm.as<BF>(),z.as<BF>(),normalized.as<BF>(),t*vh,vd,1e-6f,false,stream);
                if(commit) ops::gdn_commit_conv(raw.as<BF>(),history.as<BF>(),t,ch,taps,stream);
                CLOVER_CUDA(cudaGetLastError());CLOVER_CUDA(cudaStreamSynchronize(stream));
                const auto dir=root/("gdn-"+std::to_string(step_id++));std::filesystem::create_directories(dir);
                dump(mixed,dir/"mixed.bf16");dump(q,dir/"q.f32");dump(k,dir/"k.f32");dump(g,dir/"g.f32");dump(beta,dir/"beta.bf16");
                dump(y,dir/"y.bf16");dump(normalized,dir/"normalized.bf16");dump(history,dir/"history.bf16");dump(state,dir/"state.f32");
                bool preserved=commit||(previous_history==bytes(history)&&previous_state==bytes(state));
                require(preserved,"Preview/verify polluted persistent state");
                report["gdn"].push_back({{"tokens",t},{"preview",preview},{"commit",commit},{"read_only_preserved",preserved},{"directory",dir.string()}});
            }
        }
        if(job.contains("attention")) {
            int id=0;
            for(const auto& c:job.at("attention")) {
                int t=c.at("tokens"),prefix=c.at("prefix"),nh=c.at("heads"),nkv=c.at("kv_heads"),dim=256;
                auto& inputs=c.at("inputs");auto qp=load(inputs.at("q_proj")),kp=load(inputs.at("k_proj")),nv=load(inputs.at("v"));
                auto qw=load(inputs.at("q_norm")),kw=load(inputs.at("k_norm")),pk=load(inputs.at("prefix_key")),pv=load(inputs.at("prefix_value"));
                const auto saved_key=bytes(pk),saved_value=bytes(pv);
                DeviceBuffer q(size_t(t)*nh*dim*2),k(size_t(t)*nkv*dim*2),gate(size_t(t)*nh*dim*2),y(size_t(t)*nh*dim*2),gated(size_t(t)*nh*dim*2);
                ops::prepare_q(qp.as<BF>(),qw.as<BF>(),q.as<BF>(),gate.as<BF>(),t,nh,dim,64,prefix,10000000.0f,1e-6f,stream);
                ops::prepare_k(kp.as<BF>(),kw.as<BF>(),k.as<BF>(),t,nkv,dim,64,prefix,10000000.0f,1e-6f,stream);
                const auto dir=root/("attention-"+std::to_string(id++));std::filesystem::create_directories(dir);
                for(bool preview:{false,true}) {
                    ops::attention(q.as<BF>(),k.as<BF>(),nv.as<BF>(),pk.as<BF>(),pv.as<BF>(),y.as<BF>(),t,prefix,nh,nkv,dim,preview,stream);
                    ops::sigmoid_mul(y.as<BF>(),gate.as<BF>(),gated.as<BF>(),t,nh*dim,false,stream);
                    CLOVER_CUDA(cudaGetLastError());CLOVER_CUDA(cudaStreamSynchronize(stream));
                    dump(y,dir/(preview?"preview.bf16":"causal.bf16"));dump(gated,dir/(preview?"preview_gated.bf16":"causal_gated.bf16"));
                }
                dump(q,dir/"q.bf16");dump(k,dir/"k.bf16");dump(gate,dir/"gate.bf16");
                require(saved_key==bytes(pk)&&saved_value==bytes(pv),"Attention polluted prefix cache");
                report["attention"].push_back({{"tokens",t},{"prefix",prefix},{"prefix_preserved",true},{"directory",dir.string()}});
            }
        }
        if(job.contains("router")) for(const auto& c:job.at("router")) {
            auto r=route(c.at("logits").get<std::vector<float>>(),c.at("tokens"),c.at("experts"),c.at("top_k"),c.at("preview"),c.at("top_p"));
            report["router"].push_back({{"ids",r.ids},{"weights",r.weights},{"coreset",r.coreset}});
        }
        if(job.contains("router_device")) {
            int i=0;
            for(const auto& c:job.at("router_device")) {
                const int rows=c.at("rows");auto x=load(c.at("logits")),ids=load(c.at("ids")),allowed=load(c.at("allowed"));
                DeviceBuffer weights(size_t(rows)*8*4);
                ops::router_weights(x.as<BF>(),ids.as<int>(),allowed.as<int>(),weights.as<float>(),rows,stream);
                CLOVER_CUDA(cudaStreamSynchronize(stream));dump(weights,root/("router-weights"+std::to_string(i++)+".f32"));
            }
        }
        if(job.contains("speculation")) for(const auto& c:job.at("speculation")) {
            const auto stops=c.at("eos").get<std::vector<int>>();
            auto r=decide_exact(c.at("candidates").get<std::vector<int>>(),c.at("logits").get<std::vector<float>>(),c.at("vocab"),c.at("remaining"),std::set<int>(stops.begin(),stops.end()));
            report["speculation"].push_back(r.json());
        }
        std::ofstream result(root/"result.json");result<<report.dump(2)<<'\n';
        CLOVER_CUDA(cudaStreamDestroy(stream));std::cout<<"PASS kernel harness (oracle comparison still required)\n";return 0;
    } catch(const std::exception& e) {std::cerr<<"strata-clover-kernel-test: "<<e.what()<<'\n';return 1;}
}
