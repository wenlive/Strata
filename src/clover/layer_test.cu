#include "model.hpp"

// Test-only: a selected actual layer can be inspected with bounded experts.
// The generation CLI cannot accept this scope. No reduced model is a delivery.
int main(int argc,char** argv) {
    using namespace strata::clover;
    try {
        require(argc==3,"Usage: strata-clover-layer-test JOB.json NEW_OUTPUT_DIRECTORY");
        auto job=read_json(argv[1]);std::filesystem::path root=argv[2];
        require(!std::filesystem::exists(root),"Output already exists");std::filesystem::create_directories(root);
        cudaStream_t stream;CLOVER_CUDA(cudaStreamCreate(&stream));Json report;
        {
            ExpertStore store(job.at("index").get<std::string>(),128ull<<20,768ull<<20,16ull<<20,stream,true);
            require(store.index.at("scope")=="selected-official-layer-test","Layer harness requires explicit selected-test scope");
            Model model(store,256,32,stream,true);int step_id=0;
            for(const auto& step:job.at("steps")) {
                const auto dir=root/("step"+std::to_string(step_id++));model.trace_to(dir/"trace");
                const bool preview=step.at("preview"),commit=step.at("commit");
                const auto before=model.state_hash();
                auto result=model.forward(step.at("tokens").get<std::vector<int>>(),preview,commit,"selected_layer_test");
                if(!commit) require(before==model.state_hash(),"Selected layer preview/verify changed state");
                model.save_state(dir/"state");report["steps"].push_back(result.measurement);
            }
            report["source_layer"]=store.index.at("selected_source_layer");report["scope"]=store.index.at("scope");
        }
        CLOVER_CUDA(cudaStreamDestroy(stream));std::ofstream out(root/"result.json");out<<report.dump(2)<<'\n';return 0;
    } catch(const std::exception& e) {std::cerr<<"strata-clover-layer-test: "<<e.what()<<'\n';return 1;}
}
