#include "expert.hpp"
#include <cstring>

// Explicit test executable. It does not claim complete Clover inference.
int main(int argc,char** argv) {
    using namespace strata::clover;
    try {
        require(argc==3,"Usage: strata-clover-expert-test JOB.json RESULT.json");
        auto job=read_json(argv[1]);
        cudaStream_t stream;CLOVER_CUDA(cudaStreamCreate(&stream));
        Json results=Json::array();
        {
            ExpertStore store(job.at("index").get<std::string>(),job.at("ram_bytes"),job.at("vram_bytes"),job.at("staging_bytes"),stream,true);
            const int t=job.at("tokens"),k=job.at("top_k");
            ExpertExecutor executor(store,t,stream);
            auto xf=job.at("input").get<std::vector<float>>();
            require(xf.size()==size_t(t)*store.hidden,"Wrong input length");
            std::vector<BF> xb(xf.size());for(size_t i=0;i<xf.size();++i) xb[i]=__float2bfloat16_rn(xf[i]);
            DeviceBuffer x(xb.size()*2),out(xb.size()*2);
            CLOVER_CUDA(cudaMemcpyAsync(x.p,xb.data(),x.bytes,cudaMemcpyHostToDevice,stream));
            for(const auto& step:job.at("steps")) {
                const int layer=step.at("layer");auto ids=step.at("ids").get<std::vector<int>>();
                auto weights=step.at("weights").get<std::vector<float>>();
                if(step.value("check_bytes",false)) {
                    for(int e:ids) {
                        const BF* blob=store.acquire(layer,e);
                        std::vector<uint8_t> device(size_t(store.blob_bytes)),expected(size_t(store.blob_bytes));
                        CLOVER_CUDA(cudaMemcpyAsync(device.data(),blob,device.size(),cudaMemcpyDeviceToHost,stream));
                        CLOVER_CUDA(cudaStreamSynchronize(stream));store.used(layer,e);
                        const auto& entry=store.index.at("experts").at(size_t(layer)*store.experts+e);
                        store.read_bytes(entry.at("gate_up"),expected.data(),store.gu_bytes);
                        store.read_bytes(entry.at("down"),expected.data()+store.gu_bytes,store.blob_bytes-store.gu_bytes);
                        require(device==expected,"Expert byte mismatch");
                    }
                }
                auto start=Clock::now();executor.run(layer,x.as<BF>(),out.as<BF>(),t,k,ids,weights);
                CLOVER_CUDA(cudaStreamSynchronize(stream));double elapsed=milliseconds(start);
                std::vector<BF> y(xf.size());CLOVER_CUDA(cudaMemcpy(y.data(),out.p,out.bytes,cudaMemcpyDeviceToHost));
                std::vector<float> yf(y.size());for(size_t i=0;i<y.size();++i) yf[i]=__bfloat162float(y[i]);
                results.push_back({{"output",yf},{"elapsed_ms",elapsed},{"stats",store.stats()},{"memory",memory_snapshot()},{"workspace",executor.workspace()}});
            }
        }
        CLOVER_CUDA(cudaStreamDestroy(stream));
        std::ofstream output(argv[2]);output<<Json{{"kind","synthetic-or-selected-expert-test"},{"results",results}}.dump(2)<<'\n';
        std::cout<<"PASS expert execution steps="<<results.size()<<'\n';return 0;
    } catch(const std::exception& e) {std::cerr<<"strata-clover-expert-test: "<<e.what()<<'\n';return 1;}
}
