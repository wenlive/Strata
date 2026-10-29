#include "model.hpp"
#include "speculation.hpp"
#include "generation.hpp"
#include <set>

namespace {
using namespace strata::clover;
int argmax(const std::vector<float>& logits,int row,int vocab) {return greedy_argmax(logits,row,vocab);}
void help() {
    std::cout<<"strata-clover --config RUN.json\n"
        "Native Windows/CUDA BF16 Clover, bounded SSD/RAM/VRAM experts. Experimental.\n"
        "Required JSON fields: index (complete official BF16 index), input_tokens (nonempty integer array).\n"
        "Options: mode=selfspec|causal|denoise|forward (selfspec); max_new_tokens=16; context_tokens=256; prefill_chunk=32; draft_width=8; accept_rule=exact.\n"
        "denoise: HF LowConfidence whole-block decoding; denoise_block_size=32 (1/2/4/8/16/32), denoise_threshold=checkpoint default 0.95.\n"
        "denoise has no causal verify; commits resolved blocks causally only when generation continues. Final block stays uncommitted, like HF.\n"
        "Denoise TTFT is first resolved block; steady excludes that entire first block (null for a one-block request).\n"
        "Expert budgets: ram_cache_mib=4096; vram_cache_mib=6144; staging_mib=16. All are independent.\n"
        "Other fields: repeats=1, audit_state=false, dump_final_state (new directory), trace_directory (new directory), result_file (JSON).\n"
        "Optional presence_penalty, frequency_penalty, repetition_penalty override checkpoint generation defaults.\n"
        "Greedy sampling only. Selfspec exact acceptance compares drafts with penalized causal verify argmax.\n"
        "Checkpoint block size remains 32; runtime draft width must be 2/4/8/16/32.\n"
        "forward mode causally commits input_tokens only and can dump state for validation.\n"
        "Outputs: LOAD, PHASE, DENOISE_STEP, ROUND, RESULT JSON lines. Physical SSD bytes are unmeasured.\n";
}
}
int main(int argc,char** argv) {
    using namespace strata::clover;
    try {
        if(argc==2&&std::string(argv[1])=="--help") {help();return 0;}
        require(argc==3&&std::string(argv[1])=="--config","Usage: strata-clover --config RUN.json (or --help)");
        const Json cfg=read_json(argv[2]);
        const std::set<std::string> keys={"index","input_tokens","mode","max_new_tokens","context_tokens","prefill_chunk","draft_width","accept_rule","denoise_block_size","denoise_threshold","ram_cache_mib","vram_cache_mib","staging_mib","repeats","audit_state","dump_final_state","trace_directory","result_file","presence_penalty","frequency_penalty","repetition_penalty"};
        for(const auto& item:cfg.items()) require(keys.count(item.key())!=0,"Unknown run option: "+item.key());
        const auto prompt=cfg.at("input_tokens").get<std::vector<int>>();require(!prompt.empty(),"Empty prompt");
        const std::string mode=cfg.value("mode",std::string("selfspec"));
        require(mode=="selfspec"||mode=="causal"||mode=="denoise"||mode=="forward","Unknown mode");
        require(mode=="denoise"||(!cfg.contains("denoise_block_size")&&!cfg.contains("denoise_threshold")),"Denoising options require mode=denoise");
        require(cfg.value("accept_rule",std::string("exact"))=="exact","Only exact acceptance is implemented");
        const int width=cfg.value("draft_width",8),context=cfg.value("context_tokens",256),limit=cfg.value("max_new_tokens",16),repeats=cfg.value("repeats",1);
        const int prefill_chunk=cfg.value("prefill_chunk",32);
        require(context>0&&context<=8192&&prefill_chunk>=1&&prefill_chunk<=32,"Context must be 1..8192 and prefill_chunk 1..32");
        require(width>=2&&width<=32&&32%width==0,"Runtime draft width must divide unchanged checkpoint block size 32");
        require(repeats>=1&&repeats<=100,"Invalid repeat count");
        require(mode=="forward"||(limit>0&&limit<=context&&(mode=="denoise"||prompt.size()+size_t(limit)+size_t(width)<=size_t(context))),"Prompt/output/draft reserve exceeds context capacity");
        auto budget=[&](const char* key,uint64_t fallback) {uint64_t mib=cfg.value(key,fallback);require(mib<=1048576,"Unreasonable cache/staging budget");return mib<<20;};
        const auto startup=Clock::now();cudaStream_t stream;CLOVER_CUDA(cudaStreamCreate(&stream));
        Json summaries=Json::array();
        {
            ExpertStore source(cfg.at("index").get<std::string>(),budget("ram_cache_mib",4096),budget("vram_cache_mib",6144),budget("staging_mib",16),stream);
            const auto gen=read_json(std::filesystem::path(source.index.at("model_root").get<std::string>())/"generation_config.json");
            require(!gen.value("do_sample",false),"This first native CLI supports greedy checkpoint sampling only");
            Model model(source,context,32,stream);
            const auto denoise_config=model.config.at("denoising_config");
            const int denoise_width=cfg.value("denoise_block_size",denoise_config.value("block_size",model.block_size));
            const float denoise_threshold=cfg.value("denoise_threshold",denoise_config.at("threshold").get<float>());
            require(denoise_config.at("algorithm")=="low_confidence","Only checkpoint LowConfidence denoising is implemented");
            require(denoise_width>0&&denoise_width<=model.block_size&&model.block_size%denoise_width==0,"Denoising width must divide unchanged checkpoint block size");
            require(std::isfinite(denoise_threshold)&&denoise_threshold>=0&&denoise_threshold<=1,"Invalid denoising threshold");
            if(mode=="denoise") require(prompt.size()+uint64_t((limit+denoise_width-1)/denoise_width)*denoise_width<=uint64_t(context),"Prompt plus full denoising blocks exceeds context capacity");
            std::cout<<"LOAD "<<Json{{"startup_ms",milliseconds(startup)},{"revision",source.index.at("revision")},{"allocation",model.allocation()},{"memory",memory_snapshot()}}.dump()<<std::endl;
            // STRATA_CLOVER_GPU_LOGITS=1: penalties, argmax and confidence run on the
            // GPU; only per-row (token, confidence) pairs return to the host.
            const char* gpu_env=std::getenv("STRATA_CLOVER_GPU_LOGITS");
            const bool gpu_logits=gpu_env&&std::atoi(gpu_env)!=0;model.host_logits=!gpu_logits;
            for(int repeat=0;repeat<repeats;++repeat) {
                model.reset();Penalties penalty(gen,cfg,model.vocab);
                auto picks=[&](const std::vector<LogitEdit>& edits,int first,int rows) {
                    return model.pick(edits,first,rows,penalty.presence,penalty.frequency,penalty.repetition);
                };
                size_t phase_id=0;
                auto forward=[&](const std::vector<int>& input,bool preview,bool commit,const std::string& phase) {
                    if(cfg.contains("trace_directory")) model.trace_to(std::filesystem::path(cfg.at("trace_directory").get<std::string>())/("repeat"+std::to_string(repeat))/(std::to_string(phase_id++)+"-"+phase));
                    return model.forward(input,preview,commit,phase);
                };
                const bool audit=cfg.value("audit_state",false);double audit_ms=0;
                std::vector<int> output,committed=prompt;Json rounds=Json::array();
                const auto before=source.stats();const auto generation=Clock::now();ForwardResult prefill;
                int last_chunk=0;
                for(size_t at=0;at<prompt.size();) {
                    const size_t n=std::min<size_t>(size_t(prefill_chunk),prompt.size()-at);
                    prefill=forward(std::vector<int>(prompt.begin()+std::ptrdiff_t(at),prompt.begin()+std::ptrdiff_t(at+n)),false,true,"prefill");at+=n;last_chunk=int(n);
                }
                const double prefill_wall_ms=milliseconds(generation);int first_chunk_tokens=0,output_chunks=0;
                double ttft=0;int pending=-1;Clock::time_point first_token;
                if(mode=="causal"||mode=="selfspec") {
                    if(gpu_logits) pending=picks(penalty.edits(1,false,{},false),last_chunk-1,1)[0].token;
                    else {
                    const int last=int(prefill.logits.size()/size_t(model.vocab))-1;
                    std::vector<float> initial(prefill.logits.begin()+size_t(last)*model.vocab,prefill.logits.begin()+size_t(last+1)*model.vocab);
                    penalty.apply(initial,1,model.vocab,false,{},false);pending=argmax(initial,0,model.vocab);
                    }
                    output.push_back(pending);penalty.commit(pending);first_token=Clock::now();ttft=milliseconds(generation);
                    first_chunk_tokens=1;++output_chunks;
                }
                std::vector<float>().swap(prefill.logits);
                while(mode!="forward" && int(output.size())<limit&&(output.empty()||!penalty.stops.count(output.back()))) {
                    const auto round_started=Clock::now();
                    if(mode=="denoise") {
                        const int mask=model.config.at("mask_token_id");
                        std::vector<int> block(size_t(denoise_width),mask);std::vector<bool> unresolved(size_t(denoise_width),true);
                        Json steps=Json::array();double previews_ms=0,selection_ms=0;
                        uint64_t clean=0;
                        if(audit) {const auto start=Clock::now();clean=model.state_hash();audit_ms+=milliseconds(start);}
                        for(int iteration=0;iteration<denoise_width;++iteration) {
                            auto result=forward(block,true,false,"denoise_preview");previews_ms+=result.measurement.at("elapsed_ms").get<double>();
                            if(audit) {const auto start=Clock::now();require(clean==model.state_hash(),"Denoising preview polluted persistent state");audit_ms+=milliseconds(start);}
                            const auto selecting=Clock::now();const auto before_block=block;
                            DenoiseDecision decision;
                            if(gpu_logits) {
                                const auto rows=picks(penalty.edits(denoise_width,true,{},true,true,block,mask),0,denoise_width);
                                int fallback=-1;
                                for(int row=0;row<denoise_width;++row) {
                                    decision.prediction.push_back(rows[size_t(row)].token);decision.confidence.push_back(rows[size_t(row)].confidence);
                                    if(!unresolved[size_t(row)]) continue;
                                    if(fallback<0||rows[size_t(row)].confidence>rows[size_t(fallback)].confidence) fallback=row;
                                    if(rows[size_t(row)].confidence>denoise_threshold) decision.transfer.push_back(row);
                                }
                                require(fallback>=0,"Denoising step has no pending positions");
                                if(decision.transfer.empty()) decision.transfer.push_back(fallback);
                            } else {
                            penalty.apply_denoise(result.logits,model.vocab,block,mask);
                            decision=decide_denoise(result.logits,unresolved,model.vocab,denoise_threshold);
                            }
                            for(int position:decision.transfer) {require(unresolved[size_t(position)],"Resolved denoising position overwritten");block[size_t(position)]=decision.prediction[size_t(position)];unresolved[size_t(position)]=false;}
                            const double selecting_ms=milliseconds(selecting);selection_ms+=selecting_ms;
                            Json step={{"block",rounds.size()},{"iteration",iteration},{"input_block",before_block},{"predictions",decision.prediction},{"confidence",decision.confidence},
                                {"transferred_positions",decision.transfer},{"pending_after",std::count(unresolved.begin(),unresolved.end(),true)},
                                {"preview_ms",result.measurement.at("elapsed_ms")},{"selection_ms",selecting_ms}};
                            steps.push_back(step);std::cout<<"DENOISE_STEP "<<step.dump()<<std::endl;
                            if(std::none_of(unresolved.begin(),unresolved.end(),[](bool x){return x;})) break;
                        }
                        require(std::none_of(unresolved.begin(),unresolved.end(),[](bool x){return x;}),"Denoising step limit reached with unresolved masks");
                        const auto emitted=denoise_output(block,limit-int(output.size()),penalty.stops);
                        if(output.empty()) {first_token=Clock::now();ttft=milliseconds(generation);first_chunk_tokens=int(emitted.size());}
                        ++output_chunks;output.insert(output.end(),emitted.begin(),emitted.end());for(int token:emitted) penalty.commit(token);
                        double replay_ms=0;int replay_count=0;
                        if(int(output.size())<limit&&!penalty.stops.count(output.back())) {
                            auto replay=forward(emitted,false,true,"denoise_commit");replay_ms=replay.measurement.at("elapsed_ms");
                            replay_count=int(emitted.size());committed.insert(committed.end(),emitted.begin(),emitted.end());
                        }
                        Json round={{"round",rounds.size()},{"denoise_block_size",denoise_width},{"denoise_steps",steps.size()},{"steps",steps},
                            {"resolved_block",block},{"actual_output_tokens",emitted.size()},{"committed_input_tokens",replay_count},
                            {"preview_ms",previews_ms},{"selection_ms",selection_ms},{"commit_replay_ms",replay_ms},{"elapsed_ms",milliseconds(round_started)}};
                        rounds.push_back(round);std::cout<<"ROUND "<<round.dump()<<std::endl;continue;
                    }
                    if(mode=="causal") {
                        auto result=forward({pending},false,true,"causal_decode");committed.push_back(pending);
                        if(gpu_logits) pending=picks(penalty.edits(1,false,{},true),0,1)[0].token;
                        else {penalty.apply(result.logits,1,model.vocab,false,{});pending=argmax(result.logits,0,model.vocab);}
                        output.push_back(pending);penalty.commit(pending);
                        ++output_chunks;
                        Json round={{"round",rounds.size()},{"actual_output_tokens",1},{"committed_input_tokens",1},{"elapsed_ms",milliseconds(round_started)}};
                        rounds.push_back(round);std::cout<<"ROUND "<<round.dump()<<std::endl;continue;
                    }
                    std::vector<int> block(size_t(width),model.config.at("mask_token_id").get<int>());block[0]=pending;
                    uint64_t clean=0;
                    if(audit) {auto start=Clock::now();clean=model.state_hash();audit_ms+=milliseconds(start);}
                    auto preview=forward(block,true,false,"preview");
                    std::vector<int> candidates(size_t(width),0);candidates[0]=pending;
                    if(gpu_logits) {
                        const auto rows=picks(penalty.edits(width,true,block,true),0,width);
                        for(int row=1;row<width;++row) candidates[size_t(row)]=rows[size_t(row)].token;
                    } else {
                    penalty.apply(preview.logits,width,model.vocab,true,block);
                    for(int row=1;row<width;++row) candidates[size_t(row)]=argmax(preview.logits,row,model.vocab);
                    }
                    std::vector<float>().swap(preview.logits);
                    auto verify=forward(candidates,false,false,"verify");
                    std::vector<int> verify_ids;
                    if(gpu_logits) {for(const auto& r:picks(penalty.edits(width,false,candidates,true),0,width)) verify_ids.push_back(r.token);}
                    else {penalty.apply(verify.logits,width,model.vocab,false,candidates);
                          for(int row=0;row<width;++row) verify_ids.push_back(argmax(verify.logits,row,model.vocab));}
                    if(audit) {auto start=Clock::now();require(clean==model.state_hash(),"Full-model preview/verify polluted persistent state");audit_ms+=milliseconds(start);}
                    const auto decision=decide_exact_ids(candidates,verify_ids,limit-int(output.size()),penalty.stops);
                    std::vector<float>().swap(verify.logits);
                    for(int token:decision.output) {output.push_back(token);penalty.commit(token);}
                    ++output_chunks;
                    const int replay_count=int(decision.commit_inputs.size());
                    const auto& replay=decision.commit_inputs;
                    auto replay_result=forward(replay,false,true,"commit_replay");
                    committed.insert(committed.end(),replay.begin(),replay.end());pending=decision.pending;
                    Json round={{"round",rounds.size()},{"draft_width",width},{"matched_drafts",decision.matched_drafts},{"emitted_drafts",decision.emitted_drafts},
                                {"emitted_bonus",decision.emitted_bonus},{"bonus_token",decision.bonus},{"actual_output_tokens",decision.output.size()},{"committed_input_tokens",replay_count},
                                {"preview_ms",preview.measurement.at("elapsed_ms")},{"verify_ms",verify.measurement.at("elapsed_ms")},
                                {"commit_replay_ms",replay_result.measurement.at("elapsed_ms")},{"elapsed_ms",milliseconds(round_started)}};
                    rounds.push_back(round);std::cout<<"ROUND "<<round.dump()<<std::endl;
                }
                const double elapsed=milliseconds(generation);
                const double steady=mode=="forward"?0:milliseconds(first_token);
                const size_t decode_outputs=output.size()-((mode=="causal"||mode=="selfspec")&&!output.empty()?1:0);
                Json summary={{"repeat",repeat},{"run_kind",repeat==0?"first_request_in_process":"warm_application_caches"},
                    {"mode",mode},{"accept_rule",mode=="denoise"?Json(nullptr):Json("exact")},{"checkpoint_block_size",model.block_size},{"runtime_draft_width",mode=="denoise"?Json(nullptr):Json(width)},
                    {"input_tokens",prompt},{"output_tokens",output},{"committed_token_ids",committed},{"committed_prefix_length",model.prefix},
                    {"pending_uncommitted_token",pending>=0?Json(pending):Json(nullptr)},{"rounds",rounds},{"ttft_ms",ttft},
                    {"generation_elapsed_ms",elapsed},{"ttft_definition",mode=="denoise"?"first complete denoising block available, excludes model load":"first output token available, excludes model load"},
                    {"first_chunk_tokens",first_chunk_tokens},{"output_chunks",output_chunks},{"prefill_wall_ms",prefill_wall_ms},{"decode_wall_ms",elapsed-prefill_wall_ms},
                    {"end_to_end_token_per_second",output.empty()?Json(nullptr):Json(double(output.size())*1000/elapsed)},
                    {"decode_output_tokens",decode_outputs},{"decode_only_token_per_second",decode_outputs?Json(double(decode_outputs)*1000/(elapsed-prefill_wall_ms)):Json(nullptr)},
                    {"steady_token_per_second",mode=="denoise"?(output.size()>size_t(first_chunk_tokens)&&steady>0?Json(double(output.size()-size_t(first_chunk_tokens))*1000/steady):Json(nullptr)):(steady>0?Json(double(output.size()-1)*1000/steady):Json(0))},
                    {"state_audit_ms",audit_ms},{"runtime_cache_io",counter_delta(source.stats(),before)},{"memory",memory_snapshot()},
                    {"physical_io_note","Physical SSD bytes unmeasured; no cold-cache claim"}};
                if(mode=="denoise") {
                    summary["denoise_block_size"]=denoise_width;summary["denoise_threshold"]=denoise_threshold;
                    summary["uncommitted_output_suffix"]=std::vector<int>(output.begin()+std::ptrdiff_t(committed.size()-prompt.size()),output.end());
                    summary["steady_rate_definition"]="Tokens after first resolved block / wall time after that block; null for a one-block request";
                }
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
