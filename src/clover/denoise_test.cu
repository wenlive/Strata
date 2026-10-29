#include "generation.hpp"

// Test the exact controller helpers used by production on saved FP32 logits.
// No model or GPU forward is substituted by this executable.
int main(int argc,char** argv) {
    using namespace strata::clover;
    try {
        require(argc==3,"Usage: strata-clover-denoise-test JOB.json RESULT.json");
        const auto job=read_json(argv[1]);Json results=Json::array();
        for(const auto& test:job.at("cases")) {
            const int vocab=test.at("vocab"),mask=test.at("mask");
            const auto block=test.at("block").get<std::vector<int>>();
            require(vocab>0&&mask>=0&&mask<vocab,"Invalid denoising test geometry");
            std::vector<float> logits(block.size()*size_t(vocab));
            std::ifstream file(test.at("logits_file").get<std::string>(),std::ios::binary);
            file.read(reinterpret_cast<char*>(logits.data()),std::streamsize(logits.size()*4));require(file.gcount()==std::streamsize(logits.size()*4),"Truncated logits file");
            Penalties penalties(test.at("generation_config"),test.value("config",Json::object()),vocab);
            for(int token:test.at("history")) penalties.commit(token);
            penalties.apply_denoise(logits,vocab,block,mask);
            std::vector<bool> pending;for(int token:block) pending.push_back(token==mask);
            const auto decision=decide_denoise(logits,pending,vocab,test.at("threshold"));
            auto updated=block;for(int position:decision.transfer) updated[size_t(position)]=decision.prediction[size_t(position)];
            std::ofstream out(test.at("penalized_file").get<std::string>(),std::ios::binary);
            out.write(reinterpret_cast<const char*>(logits.data()),std::streamsize(logits.size()*4));require(bool(out),"Cannot write penalized logits");
            results.push_back(Json{{"prediction",decision.prediction},{"confidence",decision.confidence},{"transfer",decision.transfer},{"updated_block",updated},
                {"output_prefix",denoise_output(test.at("resolved_block").get<std::vector<int>>(),test.at("remaining"),penalties.stops)}});
        }
        std::ofstream out(argv[2]);out<<results.dump(2)<<'\n';require(bool(out),"Cannot write denoising test result");return 0;
    } catch(const std::exception& e) {std::cerr<<"strata-clover-denoise-test: "<<e.what()<<'\n';return 1;}
}
