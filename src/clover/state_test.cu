#include "model.hpp"
#include "speculation.hpp"

// Full official weights only. Construct controlled verify chains to exercise
// every acceptance length category, while production draft generation remains
// the unmodified preview path in main.cu. No weights/router/layers are changed.
int main(int argc,char** argv) {
    using namespace strata::clover;
    try {
        require(argc==3,"Usage: strata-clover-state-test JOB.json NEW_OUTPUT_DIRECTORY");
        auto job=read_json(argv[1]);std::filesystem::path root=argv[2];
        require(!std::filesystem::exists(root),"Output already exists");std::filesystem::create_directories(root);
        const auto prompt=job.at("input_tokens").get<std::vector<int>>();
        require(!prompt.empty()&&prompt.size()<=32,"Controlled state test requires 1..32 prompt tokens");
        const int width=8;cudaStream_t stream;CLOVER_CUDA(cudaStreamCreate(&stream));Json report;
        {
            ExpertStore store(job.at("index").get<std::string>(),0,6144ull<<20,16ull<<20,stream);
            Model model(store,256,32,stream);
            auto restore=[&]() {model.reset();return model.forward(prompt,false,true,"test_restore_prefill");};
            auto initial=restore();
            std::vector<int> truth(size_t(width),0);
            truth[0]=greedy_argmax(initial.logits,int(prompt.size())-1,model.vocab);
            for(int i=1;i<width;++i) {
                auto next=model.forward({truth[size_t(i-1)]},false,true,"test_truth_causal");
                truth[size_t(i)]=greedy_argmax(next.logits,0,model.vocab);
            }
            for(int desired:{7,3,0}) {
                restore();const uint64_t clean=model.state_hash();
                std::vector<int> block(size_t(width),model.config.at("mask_token_id").get<int>());block[0]=truth[0];
                model.forward(block,true,false,"test_preview");require(clean==model.state_hash(),"Full preview changed persistent state");
                auto candidates=truth;
                if(desired<width-1) candidates[size_t(desired+1)]=(candidates[size_t(desired+1)]+1)%model.vocab;
                auto verify=model.forward(candidates,false,false,"test_verify");
                require(clean==model.state_hash(),"Full verify changed persistent state");
                auto decision=decide_exact(candidates,verify.logits,model.vocab,width,{});
                require(decision.matched_drafts==desired,"Controlled acceptance count differs: expected "+std::to_string(desired)+", got "+std::to_string(decision.matched_drafts));
                model.forward(decision.commit_inputs,false,true,"test_commit_replay");
                const uint64_t committed=model.state_hash();const int committed_prefix=model.prefix;
                const auto dir=root/("accepted"+std::to_string(desired));model.save_state(dir/"after-speculation");
                restore();model.forward(decision.commit_inputs,false,true,"test_direct_causal");
                require(model.prefix==committed_prefix&&model.state_hash()==committed,"Accepted-prefix state differs from direct causal replay");
                model.save_state(dir/"direct-causal");
                report["cases"].push_back({{"expected_accepted",desired},{"decision",decision.json()},
                    {"committed_prefix_tokens",committed_prefix},{"preview_and_verify_read_only",true},
                    {"state_hash_equal",true},{"directory",dir.string()}});
            }
            report["revision"]=store.index.at("revision");report["scope"]="full-official-40-layer-state-validation";
            report["penalties"]="disabled for controlled acceptance oracle; production generation uses checkpoint defaults";
            report["stats"]=store.stats();report["memory"]=memory_snapshot();report["passed"]=true;
        }
        CLOVER_CUDA(cudaStreamDestroy(stream));std::ofstream out(root/"result.json");out<<report.dump(2)<<'\n';
        std::cout<<"PASS full-model all/partial/first-reject state transitions\n";return 0;
    } catch(const std::exception& e) {std::cerr<<"strata-clover-state-test: "<<e.what()<<'\n';return 1;}
}
