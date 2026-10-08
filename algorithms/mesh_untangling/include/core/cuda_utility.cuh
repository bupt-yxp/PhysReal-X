#pragma once

#include <cuda_runtime.h>
#include <cusparse.h>
#include <spdlog/spdlog.h>

#if defined(MU_ENABLE_CUDSS)
#include <cudss.h>
#endif
#include <cstdio>

#define MU_CUDA_CHECK(ans)                                       \
    {                                                            \
        ::mu::cuda::CudaAssert((ans), __FILE__, __LINE__, true); \
    }

#define MU_CHECK_CUSPARSE(func)                                                      \
    {                                                                                \
        cusparseStatus_t status = (func);                                            \
        if (status != CUSPARSE_STATUS_SUCCESS)                                       \
        {                                                                            \
            printf("CUSPARSE API failed at line %d with error: %s (%d)\n", __LINE__, \
                   cusparseGetErrorString(status), status);                          \
            exit(-1);                                                                \
        }                                                                            \
    }

namespace mu::cuda
{

constexpr int kWarpSize = 32;

template <typename Scalar>
__forceinline__ __device__ auto WarpReduceSum(Scalar val) -> Scalar
{
    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2)
    {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <typename Scalar>
__forceinline__ __device__ auto WarpBroadcast(Scalar val, int srcLane) -> Scalar
{
    return __shfl_sync(0xffffffff, val, srcLane);
}


inline void CudaAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        std::fprintf(stderr, "CUDA Error: %s (%s:%d)\n", cudaGetErrorString(code), file, line);
        if (abort)
        {
            std::exit(code);
        }
    }
}

#if defined(MU_ENABLE_CUDSS)
inline const char *CudssStatusToString(cudssStatus_t status)
{
    switch (status)
    {
        case CUDSS_STATUS_SUCCESS:
            return "success";
        case CUDSS_STATUS_NOT_INITIALIZED:
            return "not initialized";
        case CUDSS_STATUS_ALLOC_FAILED:
            return "alloc failed";
        case CUDSS_STATUS_INVALID_VALUE:
            return "invalid value";
        case CUDSS_STATUS_NOT_SUPPORTED:
            return "not supported";
        case CUDSS_STATUS_EXECUTION_FAILED:
            return "execution failed";
        case CUDSS_STATUS_INTERNAL_ERROR:
            return "internal error";
        default:
            return "unknown";
    }
}

class CuDssHandleManager
{
public:
    static auto Instance() -> CuDssHandleManager &
    {
        static CuDssHandleManager instance{};
        return instance;
    }

    CuDssHandleManager(const CuDssHandleManager &) = delete;
    CuDssHandleManager &operator=(const CuDssHandleManager &) = delete;

    auto GetHandle() -> cudssHandle_t
    {
        if (handle_ == nullptr)
        {
            const auto status = cudssCreate(&handle_);
            if (status != CUDSS_STATUS_SUCCESS)
            {
                spdlog::warn("cuDSS handle creation failed: {}", CudssStatusToString(status));
                handle_ = nullptr;
            }
        }
        return handle_;
    }

    auto DestroyHandle() -> void
    {
        if (handle_ != nullptr)
        {
            const auto status = cudssDestroy(handle_);
            if (status != CUDSS_STATUS_SUCCESS)
            {
                spdlog::warn("cuDSS handle destruction failed: {}", CudssStatusToString(status));
            }
            handle_ = nullptr;
        }
    }

private:
    CuDssHandleManager() = default;
    ~CuDssHandleManager()
    {
        if (handle_ != nullptr)
        {
            const auto status = cudssDestroy(handle_);
            if (status != CUDSS_STATUS_SUCCESS)
            {
                spdlog::warn("cuDSS handle destruction failed: {}", CudssStatusToString(status));
            }
            handle_ = nullptr;
        }
    }

    cudssHandle_t handle_ = nullptr;
};
#else
using cudssHandle_t = void *;
using cudssStatus_t = int;

inline const char *CudssStatusToString(cudssStatus_t)
{
    return "cuDSS disabled";
}

class CuDssHandleManager
{
public:
    static auto Instance() -> CuDssHandleManager &
    {
        static CuDssHandleManager instance{};
        return instance;
    }

    CuDssHandleManager(const CuDssHandleManager &) = delete;
    CuDssHandleManager &operator=(const CuDssHandleManager &) = delete;

    auto GetHandle() -> cudssHandle_t
    {
        return nullptr;
    }

    auto DestroyHandle() -> void {}

private:
    CuDssHandleManager() = default;
    ~CuDssHandleManager() = default;
};
#endif

class CusparseHandleManager
{
public:
    static auto Instance() -> CusparseHandleManager &
    {
        static CusparseHandleManager instance{};
        return instance;
    }

    CusparseHandleManager(const CusparseHandleManager &) = delete;
    CusparseHandleManager &operator=(const CusparseHandleManager &) = delete;

    auto GetHandle() -> cusparseHandle_t
    {
        if (handle_ == nullptr)
        {
            const auto status = cusparseCreate(&handle_);
            if (status != CUSPARSE_STATUS_SUCCESS)
            {
                spdlog::warn("cuSPARSE handle creation failed: {}", cusparseGetErrorString(status));
                handle_ = nullptr;
            }
        }
        return handle_;
    }

private:
    CusparseHandleManager() = default;
    ~CusparseHandleManager()
    {
        if (handle_ != nullptr)
        {
            const auto status = cusparseDestroy(handle_);
            if (status != CUSPARSE_STATUS_SUCCESS)
            {
                spdlog::warn("cuSPARSE handle destruction failed: {}",
                             cusparseGetErrorString(status));
            }
            handle_ = nullptr;
        }
    }

    cusparseHandle_t handle_ = nullptr;
};
}
