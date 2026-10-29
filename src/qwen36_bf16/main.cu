#include "model.hpp"
#include <set>

namespace {
using namespace strata::qwen36_bf16;
int argmax(const std::vector<float>& logits,int row,int vocab) {
    const auto first=logits.begin()+size_t(row)*vocab;
    return int(std::max_element(first,first+vocab)-first);
}
struct Penalties {
    float presence,frequency,repetition;
    std::vector<int> counts;std::set<int> stops;
    Penalties(const Json& gen,const Json& cfg,int vocab):counts(size_t(vocab),0) {
        presence=cfg.value("presence_penalty",0.0f);frequency=cfg.value("frequency_penalty",0.0f);repetition=cfg.value("repetition_penalty",1.0f);
        require(std::isfinite(presence)&&std::abs(presence)<=2&&std::isfinite(frequency)&&std::abs(frequency)<=2&&std::isfinite(repetition)&&repetition>0&&repetition<=2,"Invalid penalties");
        const auto& eos=gen.at("eos_token_id");
        if(eos.is_array()) for(const auto& t:eos) stops.insert(t.get<int>());else stops.insert(eos.get<int>());
        for(int t:stops) require(t>=0&&t<vocab,"Invalid EOS token");
    }
    void apply(std::vector<float>& logits) const {
        for(size_t i=0;i<logits.size();++i) {
            const int n=counts[i%counts.size()];float& v=logits[i];
            v-=float(n)*frequency;if(n) {v-=presence;v=v<0?v*repetition:v/repetition;}
        }
    }
};
void help() {
    std::cout<<"strata-q36-bf16 --config RUN.json\n"
        "Experimental native CUDA Qwen3.6 BF16 text CLI, bounded expert offload.\n"
        "Required: index (complete full-text-model), input_tokens (nonempty integer array).\n"
        "Options: mode=causal|forward (causal), max_new_tokens=16, context_tokens=256, prefill_chunk=32, repeats=1.\n"
        "Budgets: ram_cache_mib=4096, vram_cache_mib=6144, staging_mib=16.\n"
        "Optional: result_file, dump_final_state, trace_directory, presence_penalty=0, frequency_penalty=0, repetition_penalty=1.\n"
        "Sampling is explicitly greedy for comparison; checkpoint sampling defaults are not used.\n"
        "No vision, MTP, preview or coreset. All 40 text decoder layers and experts are retained.\n"
        "Phase/IO/memory logs use the same measurement contract as the Clover baseline. Physical SSD bytes unmeasured.\n";
}
}
int main(int argc,char** argv) {
    using namespace strata::qwen36_bf16;
    try {
        if(argc==2&&std::string(argv[1])=="--help") {help();return 0;}
        require(argc==3&&std::string(argv[1])=="--config","Usage: strata-q36-bf16 --config RUN.json");
        const Json cfg=read_json(argv[2]);
        const std::set<std::string> allowed={"index","input_tokens","mode","max_new_tokens","context_tokens","prefill_chunk","repeats","ram_cache_mib","vram_cache_mib","staging_mib","result_file","dump_final_state","trace_directory","presence_penalty","frequency_penalty","repetition_penalty"};
        for(const auto& p:cfg.items()) require(allowed.count(p.key()),"Unknown option: "+p.key());
        const auto prompt=cfg.at("input_tokens").get<std::vector<int>>();require(!prompt.empty(),"Empty prompt");
        const auto mode=cfg.value("mode",std::string("causal"));require(mode=="causal"||mode=="forward","Only causal or forward mode supported");
        const int context=cfg.value("context_tokens",256),limit=cfg.value("max_new_tokens",16),chunk=cfg.value("prefill_chunk",32),repeats=cfg.value("repeats",1);
        require(context>0&&context<=8192&&chunk>0&&chunk<=32&&repeats>0&&repeats<=100,"Invalid context/chunk/repeats");
        require(prompt.size()<=size_t(context)&&(mode=="forward"||(limit>0&&prompt.size()+size_t(limit)<=size_t(context))),"Context capacity exceeded");
        auto budget=[&](const char* key,uint64_t fallback) {uint64_t n=cfg.value(key,fallback);require(n<=1048576,"Unreasonable budget");return n<<20;};
        const auto startup=Clock::now();cudaStream_t stream;Q36_CUDA(cudaStreamCreate(&stream));Json summaries=Json::array();
        {
            ExpertStore store(cfg.at("index").get<std::string>(),budget("ram_cache_mib",4096),budget("vram_cache_mib",6144),budget("staging_mib",16),stream);
            const auto gen=read_json(std::filesystem::path(store.index.at("model_root").get<std::string>())/"generation_config.json");
            Model model(store,context,32,stream);
            std::cout<<"LOAD "<<Json{{"startup_ms",milliseconds(startup)},{"revision",store.index.at("revision")},{"allocation",model.allocation()},{"memory",memory_snapshot()}}.dump()<<std::endl;
            for(int repeat=0;repeat<repeats;++repeat) {
                model.reset();Penalties penalties(gen,cfg,model.vocab);std::vector<int> output,committed=prompt;Json rounds=Json::array();size_t phase_id=0;
                auto forward=[&](const std::vector<int>& input,const std::string& phase) {
                    if(cfg.contains("trace_directory")) model.trace_to(std::filesystem::path(cfg.at("trace_directory").get<std::string>())/("repeat"+std::to_string(repeat))/(std::to_string(phase_id++)+"-"+phase));
                    return model.forward(input,false,true,phase);
                };
                const auto before=store.stats();const auto start=Clock::now();ForwardResult last;
                for(size_t at=0;at<prompt.size();) {const size_t n=std::min<size_t>(size_t(chunk),prompt.size()-at);last=forward({prompt.begin()+std::ptrdiff_t(at),prompt.begin()+std::ptrdiff_t(at+n)},"prefill");at+=n;}
                double ttft=0;int pending=-1;
                if(mode!="forward") {
                    penalties.apply(last.logits);pending=argmax(last.logits,int(last.logits.size()/model.vocab)-1,model.vocab);output.push_back(pending);++penalties.counts[size_t(pending)];ttft=milliseconds(start);
                }
                std::vector<float>().swap(last.logits);
                while(mode=="causal"&&int(output.size())<limit&&!penalties.stops.count(output.back())) {
                    const auto round_start=Clock::now();auto result=forward({pending},"causal_decode");committed.push_back(pending);penalties.apply(result.logits);
                    pending=argmax(result.logits,0,model.vocab);output.push_back(pending);++penalties.counts[size_t(pending)];
                    Json round={{"round",rounds.size()},{"actual_output_tokens",1},{"committed_input_tokens",1},{"elapsed_ms",milliseconds(round_start)}};rounds.push_back(round);std::cout<<"ROUND "<<round.dump()<<std::endl;
                }
                const double elapsed=milliseconds(start),steady=elapsed-ttft;auto io=counter_delta(store.stats(),before);
                Json summary={{"repeat",repeat},{"run_kind",repeat?"warm-application-cache":"first-request-application-cache"},{"mode",mode},{"sampling","greedy; explicit comparison setting"},{"input_tokens",prompt},{"output_tokens",output},{"committed_token_ids",committed},{"committed_prefix_length",model.prefix},{"pending_uncommitted_token",pending>=0?Json(pending):Json(nullptr)},{"ttft_ms",ttft},{"generation_elapsed_ms",elapsed},{"steady_token_per_second",steady>0&&output.size()>1?double(output.size()-1)*1000/steady:0},{"rounds",rounds},{"runtime_cache_io",io},{"memory",memory_snapshot()},{"logical_expert_read_bytes_per_output_token",output.empty()?Json(nullptr):Json(double(io.at("logical_file_read_bytes").get<uint64_t>())/output.size())}};
                if(cfg.contains("dump_final_state")) {auto dir=std::filesystem::path(cfg.at("dump_final_state").get<std::string>());if(repeats>1) dir/=("repeat"+std::to_string(repeat));model.save_state(dir);summary["state_directory"]=dir.string();}
                require(committed.size()==size_t(model.prefix),"Committed token/state mismatch");summaries.push_back(summary);std::cout<<"RESULT "<<summary.dump()<<std::endl;
            }
            if(cfg.contains("result_file")) {std::ofstream file(cfg.at("result_file").get<std::string>());file<<Json{{"revision",store.index.at("revision")},{"runs",summaries}}.dump(2)<<'\n';require(bool(file),"Result write failed");}
        }
        Q36_CUDA(cudaStreamDestroy(stream));return 0;
    } catch(const std::exception& e) {std::cerr<<"strata-q36-bf16: "<<e.what()<<'\n';return 1;}
}
