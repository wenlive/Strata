#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::qwen36_bf16 {
using Json = nlohmann::json;
using BF = __nv_bfloat16;
using Clock = std::chrono::steady_clock;
inline double milliseconds(Clock::time_point t) {
    return std::chrono::duration<double,std::milli>(Clock::now()-t).count();
}
inline void check(cudaError_t e, const char* expr) {
    if(e!=cudaSuccess) throw std::runtime_error(std::string(expr)+": "+cudaGetErrorString(e));
}
#define Q36_CUDA(expr) ::strata::qwen36_bf16::check((expr),#expr)
inline void require(bool b,const std::string& message) {
    if(!b) throw std::runtime_error(message);
}
inline Json read_json(const std::filesystem::path& path) {
    std::ifstream f(path);
    require(bool(f),"Cannot open JSON: "+path.string());
    return Json::parse(f);
}
struct DeviceBuffer {
    void* p=nullptr;
    size_t bytes=0;
    DeviceBuffer()=default;
    explicit DeviceBuffer(size_t n) { allocate(n); }
    DeviceBuffer(const DeviceBuffer&)=delete;
    DeviceBuffer& operator=(const DeviceBuffer&)=delete;
    DeviceBuffer(DeviceBuffer&& o) noexcept : p(o.p),bytes(o.bytes) { o.p=nullptr;o.bytes=0; }
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        if(this!=&o) { if(p) cudaFree(p);p=o.p;bytes=o.bytes;o.p=nullptr;o.bytes=0; } return *this;
    }
    ~DeviceBuffer() { if(p) cudaFree(p); }
    void allocate(size_t n) {
        require(p==nullptr,"DeviceBuffer double allocation");
        if(n) Q36_CUDA(cudaMalloc(&p,n));
        bytes=n;
    }
    template<class T> T* as() { return static_cast<T*>(p); }
    template<class T> const T* as() const { return static_cast<const T*>(p); }
};
Json memory_snapshot();
} // namespace strata::qwen36_bf16
