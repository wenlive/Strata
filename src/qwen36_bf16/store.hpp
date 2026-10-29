#pragma once
#include "common.hpp"
#include "strata/platform/direct_file.hpp"
#include <map>

namespace strata::qwen36_bf16 {
// All expert computation and copies share one stream. Each reusable GPU slot
// has a completion event. Host scratch is reused only after its H2D completes.
// No background prefetch or mmap. RAM and VRAM form exclusive bounded LRUs:
// GPU eviction demotes to RAM; promotion removes the host cache entry.
class ExpertStore {
public:
    ExpertStore(const std::filesystem::path& index, uint64_t ram_budget,
                uint64_t vram_budget,uint64_t staging_budget,cudaStream_t stream,
                bool expert_test=false);
    ~ExpertStore();
    ExpertStore(const ExpertStore&)=delete;
    ExpertStore& operator=(const ExpertStore&)=delete;
    const BF* acquire(int layer,int expert);
    void used(int layer,int expert);
    DeviceBuffer load_tensor(const std::string& name);
    void read_bytes(const Json& slice,void* output,uint64_t bytes);
    Json stats() const;
    Json index;
    int hidden=0,ff=0,layers=0,experts=0;
    uint64_t blob_bytes=0,gu_bytes=0;
private:
    struct Slot { int key=-1;uint64_t age=0;cudaEvent_t done=nullptr; };
    std::filesystem::path root_;
    cudaStream_t stream_;
    std::map<std::string,std::unique_ptr<platform::DirectFile>> files_;
    std::unique_ptr<uint8_t[]> ram_;
    std::vector<Slot> ram_slots_,gpu_slots_;
    DeviceBuffer gpu_,gpu_stage_;
    void* scratch_=nullptr;
    void* eviction_=nullptr;
    uint64_t scratch_bytes_=0,eviction_bytes_=0;
    cudaEvent_t scratch_done_=nullptr,stage_done_=nullptr;
    int active_key_=-1,active_slot_=-1;
    uint64_t age_=0,ram_budget_=0,vram_budget_=0,staging_budget_=0;
    uint64_t gpu_hit_=0,gpu_miss_=0,ram_hit_=0,ram_miss_=0;
    uint64_t logical_bytes_=0,aligned_requested_=0,aligned_completed_=0,read_calls_=0,h2d_bytes_=0,d2h_bytes_=0;
    double read_ms_=0,h2d_wait_ms_=0;
    uint64_t host_copy_bytes_=0;
    double host_copy_ms_=0,demotion_wait_ms_=0,ram_staging_wait_ms_=0;
    platform::DirectFile& file(const std::string& name);
    const uint8_t* direct_slice(const Json& slice,uint64_t at,uint64_t n);
    void copy_host(const uint8_t* host,BF* dst,uint64_t n);
    static int find(const std::vector<Slot>& slots,int key);
    static int victim(const std::vector<Slot>& slots);
};
} // namespace strata::qwen36_bf16
