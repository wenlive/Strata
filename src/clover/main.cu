#include "model.hpp"
#include "speculation.hpp"
#include <set>

namespace {
using namespace strata::clover;
int argmax(const std::vector<float>& logits,int row,int vocab) {return greedy_argmax(logits,row,vocab);}
struct Penalties {
    float presence=0,frequency=0,repetition=1,length=0,think=0;
    int length_start=8192,length_window=8192,think_start=4096,think_window=4096,think_token=248069;
    std::vector<int> counts;std::set<int> stops;int output_length=0;
    explicit Penalties(const Json& gen,const Json& cfg,int vocab) :counts(size_t(vocab),0) {
        presence=cfg.value("presence_penalty",gen.value("presence_penalty",0.0f));
        frequency=cfg.value("frequency_penalty",gen.value("frequency_penalty",0.0f));
        repetition=cfg.value("repetition_penalty",gen.value("repetition_penalty",1.0f));
        require(std::isfinite(presence)&&std::abs(presence)<=2&&std::isfinite(frequency)&&std::abs(frequency)<=2&&std::isfinite(repetition)&&repetition>0&&repetition<=2,"Invalid sampling penalties");
        length=gen.value("dllm_length_penalty",0.0f);think=gen.value("dllm_think_end_bias",0.0f);
        length_start=gen.value("dllm_length_penalty_start",8192);length_window=gen.value("dllm_length_penalty_window",8192);
        think_start=gen.value("dllm_think_end_start",4096);think_window=gen.value("dllm_think_end_window",4096);
        think_token=gen.value("dllm_think_end_token_id",248069);
        require(length_window>0&&think_window>0&&length_start>=0&&think_start>=0&&think_token>=0&&think_token<vocab,"Invalid length/thinking penalty configuration");
        const auto& eos=gen.at("eos_token_id");
        if(eos.is_array()) for(const auto& token:eos) stops.insert(token.get<int>());
        else stops.insert(eos.get<int>());
        for(int token:stops) require(token>=0&&token<vocab,"Invalid EOS id");
    }
    void commit(int token) {++counts[size_t(token)];++output_length;}
    void apply(std::vector<float>& logits,int rows,int vocab,bool preview,const std::vector<int>& visible,bool custom=true) const {
        for(int row=0;row<rows;++row) for(int token=0;token<vocab;++token) {
            float& value=logits[size_t(row)*vocab+token];int count=counts[size_t(token)];
            if(frequency) value-=float(count)*frequency;
            if(presence&&count) value-=presence;
            if(repetition!=1&&count) value=value<0?value*repetition:value/repetition;
        }
        if(!custom) return;
        const int first=output_length+(preview?0:1);
        if(length) for(int row=0;row<rows;++row) {
            float ramp=length*std::clamp(float(first+row-length_start)/float(length_window),0.0f,1.0f);
            for(int token:stops) logits[size_t(row)*vocab+token]+=ramp;
        }
        bool close_visible=std::find(visible.begin(),visible.end(),think_token)!=visible.end();
        if(think&&!counts[size_t(think_token)]&&(!preview||!close_visible)) {
            const int row=std::max(int(preview),think_start+1-first);
            if(row<rows) logits[size_t(row)*vocab+think_token]+=think*std::min(float(first+row-think_start)/float(think_window),1.0f);
        }
    }
};
void help() {
    std::cout<<"strata-clover --config RUN.json\n"
        "Native Windows/CUDA BF16 Clover, bounded SSD/RAM/VRAM experts. Experimental.\n"
        "Required JSON fields: index (complete official BF16 index), input_tokens (nonempty integer array).\n"
        "Options: mode=selfspec|causal|forward (selfspec); max_new_tokens=16; context_tokens=256; draft_width=8; accept_rule=exact.\n"
        "Expert budgets: ram_cache_mib=4096; vram_cache_mib=6144; staging_mib=16. All are independent.\n"
        "Other fields: repeats=1, audit_state=false, dump_final_state (new directory), result_file (JSON).\n"
        "Optional presence_penalty, frequency_penalty, repetition_penalty override checkpoint generation defaults.\n"
        "Greedy sampling only. Exact acceptance compares drafts with penalized causal verify argmax.\n"
        "Checkpoint block size remains 32; runtime draft width must be 2/4/8/16/32.\n"
        "forward mode causally commits input_tokens only and can dump state for validation.\n"
        "Outputs: LOAD, PHASE, ROUND, RESULT JSON lines. Physical SSD bytes are unmeasured.\n";
}
}
int main(int argc,char** argv) {
    using namespace strata::clover;
    try {
        if(argc==2&&std::string(argv[1])=="--help") {help();return 0;}
        require(argc==3&&std::string(argv[1])=="--config","Usage: strata-clover --config RUN.json (or --help)");
        const Json cfg=read_json(argv[2]);
        const std::set<std::string> keys={"index","input_tokens","mode","max_new_tokens","context_tokens","draft_width","accept_rule","ram_cache_mib","vram_cache_mib","staging_mib","repeats","audit_state","dump_final_state","result_file","presence_penalty","frequency_penalty","repetition_penalty"};
        for(const auto& item:cfg.items()) require(keys.count(item.key())!=0,"Unknown run option: "+item.key());
        const auto prompt=cfg.at("input_tokens").get<std::vector<int>>();require(!prompt.empty(),"Empty prompt");
        const std::string mode=cfg.value("mode",std::string("selfspec"));
        require(mode=="selfspec"||mode=="causal"||mode=="forward","Unknown mode");
        require(cfg.value("accept_rule",std::string("exact"))=="exact","Only exact acceptance is implemented");
        const int width=cfg.value("draft_width",8),context=cfg.value("context_tokens",256),limit=cfg.value("max_new_tokens",16),repeats=cfg.value("repeats",1);
        require(width>=2&&width<=32&&32%width==0,"Runtime draft width must divide unchanged checkpoint block size 32");
        require(repeats>=1&&repeats<=100,"Invalid repeat count");
        require(mode=="forward"||(limit>0&&prompt.size()+size_t(limit)+size_t(width)<=size_t(context)),"Prompt/output/draft reserve exceeds context capacity");
        auto budget=[&](const char* key,uint64_t fallback) {uint64_t mib=cfg.value(key,fallback);require(mib<=1048576,"Unreasonable cache/staging budget");return mib<<20;};
        const auto startup=Clock::now();cudaStream_t stream;CLOVER_CUDA(cudaStreamCreate(&stream));
        Json summaries=Json::array();
        {
            ExpertStore source(cfg.at("index").get<std::string>(),budget("ram_cache_mib",4096),budget("vram_cache_mib",6144),budget("staging_mib",16),stream);
            const auto gen=read_json(std::filesystem::path(source.index.at("model_root").get<std::string>())/"generation_config.json");
            require(!gen.value("do_sample",false),"This first native CLI supports greedy checkpoint sampling only");
            Model model(source,context,32,stream);
            std::cout<<"LOAD "<<Json{{"startup_ms",milliseconds(startup)},{"revision",source.index.at("revision")},{"allocation",model.allocation()},{"memory",memory_snapshot()}}.dump()<<std::endl;
            for(int repeat=0;repeat<repeats;++repeat) {
                model.reset();Penalties penalty(gen,cfg,model.vocab);
                const bool audit=cfg.value("audit_state",false);double audit_ms=0;
                std::vector<int> output,committed=prompt;Json rounds=Json::array();
                const auto before=source.stats();const auto generation=Clock::now();ForwardResult prefill;
                for(size_t at=0;at<prompt.size();) {
                    const size_t n=std::min<size_t>(32,prompt.size()-at);
                    prefill=model.forward(std::vector<int>(prompt.begin()+std::ptrdiff_t(at),prompt.begin()+std::ptrdiff_t(at+n)),false,true,"prefill");at+=n;
                }
                double ttft=0;int pending=-1;Clock::time_point first_token;
                if(mode!="forward") {
                    const int last=int(prefill.logits.size()/size_t(model.vocab))-1;
                    std::vector<float> initial(prefill.logits.begin()+size_t(last)*model.vocab,prefill.logits.begin()+size_t(last+1)*model.vocab);
                    penalty.apply(initial,1,model.vocab,false,{},false);pending=argmax(initial,0,model.vocab);
                    output.push_back(pending);penalty.commit(pending);first_token=Clock::now();ttft=milliseconds(generation);
                }
                while(mode!="forward" && int(output.size())<limit&&!penalty.stops.count(output.back())) {
                    const auto round_started=Clock::now();
                    if(mode=="causal") {
                        auto result=model.forward({pending},false,true,"causal_decode");committed.push_back(pending);
                        penalty.apply(result.logits,1,model.vocab,false,{});
                        pending=argmax(result.logits,0,model.vocab);output.push_back(pending);penalty.commit(pending);
                        Json round={{"round",rounds.size()},{"actual_output_tokens",1},{"committed_input_tokens",1},{"elapsed_ms",milliseconds(round_started)}};
                        rounds.push_back(round);std::cout<<"ROUND "<<round.dump()<<std::endl;continue;
                    }
                    std::vector<int> block(size_t(width),model.config.at("mask_token_id").get<int>());block[0]=pending;
                    uint64_t clean=0;
                    if(audit) {auto start=Clock::now();clean=model.state_hash();audit_ms+=milliseconds(start);}
                    auto preview=model.forward(block,true,false,"preview");
                    penalty.apply(preview.logits,width,model.vocab,true,block);
                    std::vector<int> candidates(size_t(width),0);candidates[0]=pending;
                    for(int row=1;row<width;++row) candidates[size_t(row)]=argmax(preview.logits,row,model.vocab);
                    auto verify=model.forward(candidates,false,false,"verify");
                    penalty.apply(verify.logits,width,model.vocab,false,candidates);
                    if(audit) {auto start=Clock::now();require(clean==model.state_hash(),"Full-model preview/verify polluted persistent state");audit_ms+=milliseconds(start);}
                    const auto decision=decide_exact(candidates,verify.logits,model.vocab,limit-int(output.size()),penalty.stops);
                    for(int token:decision.output) {output.push_back(token);penalty.commit(token);}
                    const int replay_count=int(decision.commit_inputs.size());
                    const auto& replay=decision.commit_inputs;
                    auto replay_result=model.forward(replay,false,true,"commit_replay");
                    committed.insert(committed.end(),replay.begin(),replay.end());pending=decision.pending;
                    Json round={{"round",rounds.size()},{"draft_width",width},{"matched_drafts",decision.matched_drafts},{"emitted_drafts",decision.emitted_drafts},
                                {"emitted_bonus",decision.emitted_bonus},{"bonus_token",decision.bonus},{"actual_output_tokens",decision.output.size()},{"committed_input_tokens",replay_count},
                                {"preview_ms",preview.measurement.at("elapsed_ms")},{"verify_ms",verify.measurement.at("elapsed_ms")},
                                {"commit_replay_ms",replay_result.measurement.at("elapsed_ms")},{"elapsed_ms",milliseconds(round_started)}};
                    rounds.push_back(round);std::cout<<"ROUND "<<round.dump()<<std::endl;
                }
                const double elapsed=milliseconds(generation);
                const double steady=mode=="forward"?0:milliseconds(first_token);
                Json summary={{"repeat",repeat},{"run_kind",repeat==0?"first_request_in_process":"warm_application_caches"},
                    {"mode",mode},{"accept_rule","exact"},{"checkpoint_block_size",model.block_size},{"runtime_draft_width",width},
                    {"input_tokens",prompt},{"output_tokens",output},{"committed_token_ids",committed},{"committed_prefix_length",model.prefix},
                    {"pending_uncommitted_token",pending>=0?Json(pending):Json(nullptr)},{"rounds",rounds},{"ttft_ms",ttft},
                    {"generation_elapsed_ms",elapsed},{"steady_token_per_second",steady>0?double(output.size()-1)*1000/steady:0},
                    {"state_audit_ms",audit_ms},{"runtime_cache_io",counter_delta(source.stats(),before)},{"memory",memory_snapshot()},
                    {"physical_io_note","Physical SSD bytes unmeasured; no cold-cache claim"}};
                const uint64_t logical=summary["runtime_cache_io"]["logical_file_read_bytes"];
                summary["logical_expert_read_bytes_per_output_token"]=output.empty()?Json(nullptr):Json(double(logical)/double(output.size()));
                if(cfg.contains("dump_final_state")) {auto path=std::filesystem::path(cfg.at("dump_final_state").get<std::string>());if(repeats>1) path/=("repeat"+std::to_string(repeat));model.save_state(path);summary["state_directory"]=path.string();}
                summaries.push_back(summary);std::cout<<"RESULT "<<summary.dump()<<std::endl;
            }
            if(cfg.contains("result_file")) {std::ofstream out(cfg.at("result_file").get<std::string>());out<<Json{{"revision",source.index.at("revision")},{"runs",summaries}}.dump(2)<<'\n';require(bool(out),"Cannot write result file");}
        }
        CLOVER_CUDA(cudaStreamDestroy(stream));return 0;
    } catch(const std::exception& e) {std::cerr<<"strata-clover: "<<e.what()<<'\n';return 1;}
}
