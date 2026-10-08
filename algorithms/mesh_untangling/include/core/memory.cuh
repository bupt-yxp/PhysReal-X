#pragma once
#include <cstdlib>
#include <utility>

#include "core/base.cuh"

namespace mu::memory
{
enum class MUMemoryType
{
    kHostMemory = 0,
    kDeviceMemory = 1
};

template <typename DerivedAllocator>
class MUAllocatorBase
{
public:
    template <typename T, typename... Argument>
    auto Allocate(T *&ptr, Argument &&...argument) -> void
    {
        if constexpr (sizeof...(argument))
        {
            static_cast<DerivedAllocator *>(this)->AllocateImpl(
                ptr, std::forward<Argument &&>(argument)...);
        }
        else
        {
            static_cast<DerivedAllocator *>(this)->AllocateImpl(ptr);
        }
    }

    template <typename T, typename... Argument>
    auto Reallocate(T *&ptr, Argument &&...argument) -> void
    {
        if constexpr (sizeof...(argument))
        {
            static_cast<DerivedAllocator *>(this)->ReallocateImpl(
                ptr, std::forward<Argument &&>(argument)...);
        }
        else
        {
            static_cast<DerivedAllocator *>(this)->ReallocateImpl(ptr);
        }
    }

    template <typename T, typename... Argument>
    auto Deallocate(T *&ptr, Argument &&...argument) -> void
    {
        if constexpr (sizeof...(argument))
        {
            static_cast<DerivedAllocator *>(this)->DeallocateImpl(
                ptr, std::forward<Argument &&>(argument)...);
        }
        else
        {
            static_cast<DerivedAllocator *>(this)->DeallocateImpl(ptr);
        }
    }
};

template <typename T>
concept MUAllocatorType = std::is_base_of_v<MUAllocatorBase<T>, T>;

class MUDefaultHostAllocator : public MUAllocatorBase<MUDefaultHostAllocator>
{
public:
    friend class MUAllocatorBase<MUDefaultHostAllocator>;
    constexpr static MUMemoryType kMemoryType_ = MUMemoryType::kHostMemory;

protected:
    template <typename T, typename... Arg>
    auto AllocateImpl(T *&ptr, size_t size, Arg &&...) -> void
    {
        ptr = reinterpret_cast<T *>(std::malloc(size));
    }

    template <typename T, typename... Arg>
    auto DeallocateImpl(T *&ptr, Arg &&...) -> void
    {
        std::free(ptr);
    }

    template <typename T, typename... Arg>
    auto ReallocateImpl(T *&ptr, size_t size, Arg &&...) -> void
    {
        ptr = std::realloc(ptr, size);
    }
};

class MUDefaultCudaAllocator : public MUAllocatorBase<MUDefaultCudaAllocator>
{
public:
    friend class MUAllocatorBase<MUDefaultCudaAllocator>;
    constexpr static MUMemoryType kMemoryType_ = MUMemoryType::kDeviceMemory;

protected:
    template <typename T, typename... Arg>
    auto AllocateImpl(T *&ptr, size_t size, Arg &&...) -> void
    {
        cudaMalloc(reinterpret_cast<void **>(&ptr), size);
    }

    template <typename T, typename... Arg>
    auto DeallocateImpl(T *&ptr, Arg &&...) -> void
    {
        cudaFree(reinterpret_cast<void *>(ptr));
    }

    template <typename T, typename... Arg>
    auto ReallocateImpl(T *&ptr, size_t size, Arg &&...) -> void
    {
        cudaFree(reinterpret_cast<void *>(ptr));
        cudaMalloc(reinterpret_cast<void **>(&ptr), size);
    }
};

}
