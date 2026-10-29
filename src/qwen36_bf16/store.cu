#include "store.hpp"
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#endif
#include <cstring>
#include <limits>
#include <unordered_set>

namespace strata::qwen36_bf16 {
Json memory_snapshot() {
    Json j;
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS_EX p{};p.cb=sizeof(p);
    require(GetProcessMemoryInfo(GetCurrentProcess(),reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&p),sizeof(p)),"GetProcessMemoryInfo failed");
    j["working_set_bytes"]=p.WorkingSetSize;j["peak_working_set_bytes"]=p.PeakWorkingSetSize;
    j["private_commit_bytes"]=p.PrivateUsage;j["peak_pagefile_bytes"]=p.PeakPagefileUsage;
    MEMORYSTATUSEX m{};m.dwLength=sizeof(m);GlobalMemoryStatusEx(&m);
    j["available_ram_bytes"]=m.ullAvailPhys;j["available_commit_bytes"]=m.ullAvailPageFile;
    IO_COUNTERS io{};
    if(GetProcessIoCounters(GetCurrentProcess(),&io)) {
        j["process_read_transfer_bytes"]=io.ReadTransferCount;
        j["process_read_transfer_note"]="Windows process I/O counter; not a device physical SSD byte counter";
    }
#endif
    size_t free=0,total=0;Q36_CUDA(cudaMemGetInfo(&free,&total));
    j["cuda_free_bytes"]=free;j["cuda_total_bytes"]=total;
    return j;
}
int ExpertStore::find(const std::vector<Slot>& slots,int key) {
    for(size_t i=0;i<slots.size();++i) if(slots[i].key==key) return int(i);
    return -1;
}
int ExpertStore::victim(const std::vector<Slot>& slots) {
    require(!slots.empty(),"Empty cache victim");
    for(size_t i=0;i<slots.size();++i) if(slots[i].key<0) return int(i);
    return int(std::min_element(slots.begin(),slots.end(),[](const Slot& a,const Slot& b){return a.age<b.age;})-slots.begin());
}
ExpertStore::ExpertStore(const std::filesystem::path& path,uint64_t ram_budget,
                        uint64_t vram_budget,uint64_t staging_budget,cudaStream_t stream,bool expert_test)
    :stream_(stream),ram_budget_(ram_budget),vram_budget_(vram_budget),staging_budget_(staging_budget) {
    index=read_json(path);
    require(index.at("format")=="strata-qwen36-bf16-index" && index.at("version")==1,"Unsupported Qwen3.6 BF16 index format");
    require(index.at("complete").get<bool>(),"Incomplete/metadata-only index cannot run");
    const auto scope=index.at("scope").get<std::string>();
    require(scope=="full-text-model" || (expert_test && (scope=="synthetic-expert-test"||scope=="selected-official-expert-test"||scope=="selected-official-layer-test")),"Index scope cannot execute this mode");
    root_=index.at("model_root").get<std::string>();
    const auto& c=index.at("config");
    hidden=c.at("hidden_size");ff=c.at("moe_intermediate_size");
    layers=c.at("num_hidden_layers");experts=c.at("num_experts");
    require(hidden>0&&hidden<=16384&&ff>0&&ff<=16384&&layers>0&&layers<=512&&experts>0&&experts<=4096,"Invalid model geometry");
    gu_bytes=uint64_t(2)*ff*hidden*2;blob_bytes=uint64_t(3)*ff*hidden*2;
    for(const auto& pair:index.at("tensors").items()) {
        const auto& t=pair.value();
        require(t.at("dtype")=="BF16" && t.at("layout")=="row-major","Invalid tensor dtype/layout: "+pair.key());
        uint64_t elements=1;
        for(const auto& dim:t.at("shape")) {
            const uint64_t d=dim.get<uint64_t>();
            require(d>0&&elements<=UINT64_MAX/d,"Invalid/overflowing tensor shape: "+pair.key());elements*=d;
        }
        require(elements<=UINT64_MAX/2 && elements*2==t.at("bytes").get<uint64_t>(),"Tensor shape/byte mismatch: "+pair.key());
        const auto filename=t.at("file").get<std::string>();
        const uint64_t size=index.at("files").at(filename).at("file_size");
        const uint64_t off=t.at("offset"),n=t.at("bytes");
        require(off<=size&&n<=size-off,"Tensor outside file: "+pair.key());
    }
    require(index.at("experts").size()==size_t(layers)*experts,"Missing expert slices");
    for(int l=0;l<layers;++l) for(int e=0;e<experts;++e) {
        auto& entry=index.at("experts").at(size_t(l)*experts+e);
        require(entry.at("layer")==l && entry.at("expert")==e,"Expert order/identity mismatch");
        require(entry.at("gate_up").at("bytes")==gu_bytes && entry.at("down").at("bytes")==blob_bytes-gu_bytes,"Invalid expert byte length");
        const auto& gu=index.at("tensors").at("model.layers."+std::to_string(l)+".mlp.experts.gate_up_proj");
        const auto& down=index.at("tensors").at("model.layers."+std::to_string(l)+".mlp.experts.down_proj");
        require(gu.at("shape")==Json::array({experts,2*ff,hidden}) && down.at("shape")==Json::array({experts,hidden,ff}),"Wrong routed expert shape");
        require(entry.at("gate_up").at("file")==gu.at("file") && entry.at("gate_up").at("offset").get<uint64_t>()==gu.at("offset").get<uint64_t>()+uint64_t(e)*gu_bytes,"Wrong gate/up expert slice");
        require(entry.at("down").at("file")==down.at("file") && entry.at("down").at("offset").get<uint64_t>()==down.at("offset").get<uint64_t>()+uint64_t(e)*(blob_bytes-gu_bytes),"Wrong down expert slice");
    }
    // Re-check every file size before allocating any expert memory. Payload hashes
    // were checked by index.py; files are opened read-only, never mapped.
    for(const auto& pair:index.at("files").items()) {
        const std::filesystem::path name=pair.key();
        require(name.filename()==name && name.extension()==".safetensors","Unsafe source filename");
        require(std::filesystem::file_size(root_/name)==pair.value().at("file_size").get<uint64_t>(),"Missing/truncated source: "+name.string());
    }
    // 4 MiB + one page of prefix/suffix alignment. Small synthetic tensors use
    // proportionally smaller buffers. One expert GPU staging slot when cache=0.
    scratch_bytes_=((std::min<uint64_t>(gu_bytes,4ull<<20)+8191)/4096)*4096;
    const uint64_t gpu_slots=std::min<uint64_t>(vram_budget/blob_bytes,uint64_t(layers)*experts);
    const uint64_t ram_slots=std::min<uint64_t>(ram_budget/blob_bytes,uint64_t(layers)*experts);
    const uint64_t gpu_stage=gpu_slots?0:blob_bytes;
    // A separate bounded pinned slot preserves a GPU victim while a RAM hit
    // is promoted, including when RAM contains only one expert (a swap).
    eviction_bytes_=gpu_slots&&ram_slots?blob_bytes:0;
    const uint64_t staging=scratch_bytes_+eviction_bytes_+gpu_stage;
    require(staging<=staging_budget,"Staging budget too small: need "+std::to_string(staging)+" bytes (host+GPU)");
#if defined(_WIN32)
    MEMORYSTATUSEX mem{};mem.dwLength=sizeof(mem);require(GlobalMemoryStatusEx(&mem),"Cannot query host capacity");
    const uint64_t host=ram_slots*blob_bytes+scratch_bytes_+eviction_bytes_;
    require(host+(4ull<<30)<std::min<uint64_t>(mem.ullAvailPhys,mem.ullAvailPageFile),"RAM/commit budget would leave under 4 GiB headroom");
#endif
    size_t gpu_free=0,gpu_total=0;Q36_CUDA(cudaMemGetInfo(&gpu_free,&gpu_total));
    require(gpu_slots*blob_bytes+gpu_stage+(512ull<<20)<gpu_free,"VRAM expert allocation would leave under 512 MiB headroom");
    ram_slots_.resize(size_t(ram_slots));gpu_slots_.resize(size_t(gpu_slots));
    if(ram_slots) ram_.reset(new uint8_t[size_t(ram_slots*blob_bytes)]);
    gpu_.allocate(size_t(gpu_slots*blob_bytes));gpu_stage_.allocate(size_t(gpu_stage));
    scratch_=platform::DirectFile::alloc_aligned(size_t(scratch_bytes_));
    require(scratch_!=nullptr,"Cannot allocate aligned staging");
    Q36_CUDA(cudaHostRegister(scratch_,size_t(scratch_bytes_),cudaHostRegisterDefault));
    if(eviction_bytes_) Q36_CUDA(cudaMallocHost(&eviction_,size_t(eviction_bytes_)));
    Q36_CUDA(cudaEventCreateWithFlags(&scratch_done_,cudaEventDisableTiming));
    Q36_CUDA(cudaEventRecord(scratch_done_,stream_));
    Q36_CUDA(cudaEventCreateWithFlags(&stage_done_,cudaEventDisableTiming));
    Q36_CUDA(cudaEventRecord(stage_done_,stream_));
    resident_=gpu_slots==uint64_t(layers)*experts && !expert_test;
    if(resident_) {preload();return;}
    for(auto& s:gpu_slots_) {
        Q36_CUDA(cudaEventCreateWithFlags(&s.done,cudaEventDisableTiming));
        Q36_CUDA(cudaEventRecord(s.done,stream_));
    }
}
// Slot k holds expert key k for the process lifetime: nothing is evicted, so
// no per-slot events, LRU ages or RAM tier are needed after this one pass.
void ExpertStore::preload() {
    const auto start=Clock::now();
    for(int key=0;key<layers*experts;++key) {
        const auto& ex=index.at("experts").at(size_t(key));
        auto* target=gpu_.as<uint8_t>()+uint64_t(key)*blob_bytes;uint64_t base=0;
        for(const char* part:{"gate_up","down"}) {
            const auto& sl=ex.at(part);const uint64_t bytes=sl.at("bytes");
            for(uint64_t at=0;at<bytes;) {
                const uint64_t n=std::min<uint64_t>(bytes-at,scratch_bytes_-4096);
                const auto* source=direct_slice(sl,at,n);
                Q36_CUDA(cudaMemcpyAsync(target+base+at,source,size_t(n),cudaMemcpyHostToDevice,stream_));
                Q36_CUDA(cudaEventRecord(scratch_done_,stream_));h2d_bytes_+=n;at+=n;
            }
            base+=bytes;
        }
        gpu_slots_[size_t(key)].key=key;
    }
    Q36_CUDA(cudaStreamSynchronize(stream_));
    preload_ms_=milliseconds(start);
}
ExpertStore::~ExpertStore() {
    cudaStreamSynchronize(stream_);
    for(auto& s:gpu_slots_) if(s.done) cudaEventDestroy(s.done);
    if(stage_done_) cudaEventDestroy(stage_done_);
    if(scratch_done_) cudaEventDestroy(scratch_done_);
    if(scratch_) {cudaHostUnregister(scratch_);platform::DirectFile::free_aligned(scratch_);}
    if(eviction_) cudaFreeHost(eviction_);
}
platform::DirectFile& ExpertStore::file(const std::string& name) {
    require(index.at("files").contains(name),"Unknown slice source: "+name);
    auto& ptr=files_[name];
    if(!ptr) {
        ptr=std::make_unique<platform::DirectFile>();std::string error;
        require(ptr->open((root_/name).string(),error),error);
    }
    return *ptr;
}
const uint8_t* ExpertStore::direct_slice(const Json& slice,uint64_t at,uint64_t n) {
    require(at<=slice.at("bytes").get<uint64_t>() && n<=slice.at("bytes").get<uint64_t>()-at,"Slice overrun");
    const uint64_t offset=slice.at("offset").get<uint64_t>()+at;
    require(offset>=at,"Offset overflow");
    const uint64_t begin=offset/4096*4096,prefix=offset-begin;
    const uint64_t length=(prefix+n+4095)/4096*4096;
    require(length<=scratch_bytes_ && length<=UINT32_MAX,"Direct read exceeds staging budget");
    auto t=Clock::now();Q36_CUDA(cudaEventSynchronize(scratch_done_));h2d_wait_ms_+=milliseconds(t);
    auto& f=file(slice.at("file").get<std::string>());
    require(offset<=f.size() && n<=f.size()-offset,"Truncated slice");
    std::string error;t=Clock::now();
    require(f.submit(begin,scratch_,uint32_t(length),1,error),error);
    platform::Completion completion;
    do { if(!f.wait(&completion,1,1000)) continue;
         if(completion.tag!=platform::DirectFile::WAKE_TAG) break;
    } while(true);
    read_ms_+=milliseconds(t);
    require(completion.ok && completion.bytes>=prefix+n,"DirectFile read failed/short at offset "+std::to_string(offset));
    logical_bytes_+=n;aligned_requested_+=length;aligned_completed_+=completion.bytes;++read_calls_;
    return static_cast<const uint8_t*>(scratch_)+prefix;
}
void ExpertStore::copy_host(const uint8_t* host,BF* target,uint64_t bytes) {
    for(uint64_t at=0;at<bytes;) {
        const uint64_t n=std::min<uint64_t>(bytes-at,scratch_bytes_-4096);
        auto wait=Clock::now();Q36_CUDA(cudaEventSynchronize(scratch_done_));ram_staging_wait_ms_+=milliseconds(wait);
        auto copy=Clock::now();std::memcpy(scratch_,host+at,size_t(n));host_copy_ms_+=milliseconds(copy);host_copy_bytes_+=n;
        Q36_CUDA(cudaMemcpyAsync(reinterpret_cast<uint8_t*>(target)+at,scratch_,size_t(n),cudaMemcpyHostToDevice,stream_));
        Q36_CUDA(cudaEventRecord(scratch_done_,stream_));h2d_bytes_+=n;at+=n;
    }
}
const BF* ExpertStore::acquire(int layer,int expert) {
    require(active_key_<0,"Previous expert must be marked used before acquiring another");
    require(layer>=0&&layer<layers&&expert>=0&&expert<experts,"Expert id out of range");
    const int key=layer*experts+expert;
    if(resident_) {++gpu_hit_;active_key_=key;active_slot_=key;return gpu_.as<BF>()+uint64_t(key)*blob_bytes/2;}
    int gs=find(gpu_slots_,key);
    if(gs>=0) {
        ++gpu_hit_;gpu_slots_[size_t(gs)].age=++age_;
        active_slot_=gs;active_key_=key;
        return gpu_.as<BF>()+uint64_t(gs)*blob_bytes/2;
    }
    ++gpu_miss_;
    BF* target=nullptr;
    int demoted_key=-1;
    if(!gpu_slots_.empty()) {
        gs=victim(gpu_slots_);Q36_CUDA(cudaEventSynchronize(gpu_slots_[size_t(gs)].done));
        target=gpu_.as<BF>()+uint64_t(gs)*blob_bytes/2;
        if(eviction_ && gpu_slots_[size_t(gs)].key>=0) {
            demoted_key=gpu_slots_[size_t(gs)].key;
            // Finish D2H before overwriting the GPU slot. One stream also
            // orders this copy after the prior expert's last computation.
            auto demote=Clock::now();
            Q36_CUDA(cudaMemcpyAsync(eviction_,target,size_t(blob_bytes),cudaMemcpyDeviceToHost,stream_));
            Q36_CUDA(cudaStreamSynchronize(stream_));demotion_wait_ms_+=milliseconds(demote);d2h_bytes_+=blob_bytes;
        }
        gpu_slots_[size_t(gs)].key=-1;
    } else {
        Q36_CUDA(cudaEventSynchronize(stage_done_));target=gpu_stage_.as<BF>();
    }
    int rs=find(ram_slots_,key);
    if(rs>=0) {
        ++ram_hit_;ram_slots_[size_t(rs)].age=++age_;
        copy_host(ram_.get()+uint64_t(rs)*blob_bytes,target,blob_bytes);
        // copy_host has consumed all RAM bytes into pinned scratch on return.
        // Its final asynchronous transfer reads scratch, not this RAM slot.
        if(gs>=0) ram_slots_[size_t(rs)].key=-1;
    } else {
        ++ram_miss_;
        uint8_t* host=nullptr;
        if(gs<0 && !ram_slots_.empty()) {rs=victim(ram_slots_);host=ram_.get()+uint64_t(rs)*blob_bytes;ram_slots_[size_t(rs)].key=-1;}
        const auto& ex=index.at("experts").at(size_t(key));
        uint64_t base=0;
        for(const char* part:{"gate_up","down"}) {
            const auto& sl=ex.at(part);const uint64_t bytes=sl.at("bytes");
            for(uint64_t at=0;at<bytes;) {
                const uint64_t n=std::min<uint64_t>(bytes-at,scratch_bytes_-4096);
                const auto* source=direct_slice(sl,at,n);
                if(host) {auto copy=Clock::now();std::memcpy(host+base+at,source,size_t(n));host_copy_ms_+=milliseconds(copy);host_copy_bytes_+=n;}
                Q36_CUDA(cudaMemcpyAsync(reinterpret_cast<uint8_t*>(target)+base+at,source,size_t(n),cudaMemcpyHostToDevice,stream_));
                Q36_CUDA(cudaEventRecord(scratch_done_,stream_));h2d_bytes_+=n;at+=n;
            }
            base+=bytes;
        }
        if(host) {ram_slots_[size_t(rs)].key=key;ram_slots_[size_t(rs)].age=++age_;}
    }
    if(demoted_key>=0) {
        const int dest=victim(ram_slots_);
        auto copy=Clock::now();std::memcpy(ram_.get()+uint64_t(dest)*blob_bytes,eviction_,size_t(blob_bytes));host_copy_ms_+=milliseconds(copy);host_copy_bytes_+=blob_bytes;
        ram_slots_[size_t(dest)].key=demoted_key;ram_slots_[size_t(dest)].age=++age_;
    }
    if(gs>=0) {gpu_slots_[size_t(gs)].key=key;gpu_slots_[size_t(gs)].age=++age_;}
    active_key_=key;active_slot_=gs;
    return target;
}
void ExpertStore::used(int layer,int expert) {
    require(active_key_==layer*experts+expert,"Expert lifetime mismatch: used after another acquire");
    if(resident_) {active_key_=-1;return;}
    Q36_CUDA(cudaEventRecord(active_slot_>=0?gpu_slots_[size_t(active_slot_)].done:stage_done_,stream_));
    active_key_=-1;
}
DeviceBuffer ExpertStore::load_tensor(const std::string& name) {
    const auto& t=index.at("tensors").at(name);
    require(t.at("dtype")=="BF16" && name.find(".experts.")==std::string::npos,"Dense loader refuses non-BF16 or routed tensor");
    const uint64_t bytes=t.at("bytes");DeviceBuffer result{size_t(bytes)};
    for(uint64_t at=0;at<bytes;) {
        const uint64_t n=std::min<uint64_t>(bytes-at,scratch_bytes_-4096);
        auto* host=direct_slice(t,at,n);
        Q36_CUDA(cudaMemcpyAsync(result.as<uint8_t>()+at,host,size_t(n),cudaMemcpyHostToDevice,stream_));
        Q36_CUDA(cudaEventRecord(scratch_done_,stream_));h2d_bytes_+=n;at+=n;
    }
    return result;
}
void ExpertStore::read_bytes(const Json& slice,void* output,uint64_t bytes) {
    require(bytes==slice.at("bytes").get<uint64_t>(),"read_bytes length mismatch");
    for(uint64_t at=0;at<bytes;) {
        const uint64_t n=std::min<uint64_t>(bytes-at,scratch_bytes_-4096);
        auto* source=direct_slice(slice,at,n);
        std::memcpy(static_cast<uint8_t*>(output)+at,source,size_t(n));at+=n;
    }
}
Json ExpertStore::stats() const {
    std::unordered_set<int> gpu_keys;
    for(const auto& s:gpu_slots_) if(s.key>=0) gpu_keys.insert(s.key);
    uint64_t duplicates=0;
    for(const auto& s:ram_slots_) if(s.key>=0 && gpu_keys.count(s.key)) ++duplicates;
    return {{"gpu_hits",gpu_hit_},{"gpu_misses",gpu_miss_},{"ram_hits",ram_hit_},{"ram_misses",ram_miss_},
            {"logical_file_read_bytes",logical_bytes_},{"aligned_file_requested_bytes",aligned_requested_},
            {"aligned_file_completed_bytes",aligned_completed_},{"read_calls",read_calls_},{"h2d_bytes",h2d_bytes_},{"d2h_bytes",d2h_bytes_},
            {"read_ms",read_ms_},{"host_staging_wait_ms",h2d_wait_ms_},
            {"host_copy_bytes",host_copy_bytes_},{"host_copy_ms",host_copy_ms_},{"demotion_wait_ms",demotion_wait_ms_},{"ram_staging_wait_ms",ram_staging_wait_ms_},
            {"ram_cache_budget_bytes",ram_budget_},{"ram_cache_allocated_bytes",ram_slots_.size()*blob_bytes},
            {"vram_cache_budget_bytes",vram_budget_},{"vram_cache_allocated_bytes",gpu_.bytes},
            {"staging_budget_bytes",staging_budget_},{"host_staging_bytes",scratch_bytes_+eviction_bytes_},
            {"host_read_staging_bytes",scratch_bytes_},{"host_eviction_staging_bytes",eviction_bytes_},{"gpu_staging_bytes",gpu_stage_.bytes},
            {"cache_policy",resident_?"resident: every routed expert preloaded in VRAM":"exclusive GPU/RAM LRU; GPU eviction demotes to RAM"},{"preload_ms",preload_ms_},{"ram_gpu_duplicate_experts",duplicates},
            {"io_mode","Strata DirectFile / unbuffered aligned reads; no mmap"},
            {"physical_ssd_bytes",nullptr},{"physical_ssd_note","Not measured at device level; file bytes are logical unbuffered completions"}};
}
} // namespace strata::qwen36_bf16
