#pragma once

#include <cuda.h>

#include <exception>

#include "core/cuda_utility.cuh"

#ifndef MU_HOST_DEV_FUNC
#define MU_HOST_DEV_FUNC __device__ __host__
#endif

#ifndef MU_DEV_FUNC
#define MU_DEV_FUNC __device__
#endif

#ifndef MU_HOST_FUNC
#define MU_HOST_FUNC __device__
#endif

#ifndef MU_KERNEL_FUNC
#define MU_KERNEL_FUNC __global__
#endif

#ifndef MU_FORCE_INLINE
#define MU_FORCE_INLINE __forceinline__
#endif

#ifndef MU_CONSTEXPR
#define MU_CONSTEXPR constexpr
#endif

#if __APPLE__
#define VK_USE_PLATFORM_MACOS_MVK
#define GLFW_EXPOSE_NATIVE_COCOA
#elif _WIN32
#define VK_USE_PLATFORM_WIN32_KHR
#define GLFW_EXPOSE_NATIVE_WIN32
#define NOMINMAX
#elif linux
#define VK_USE_PLATFORM_XCB_KHR
#define GLFW_EXPOSE_NATIVE_X11
#endif

MU_DEV_FUNC MU_FORCE_INLINE float atomicMin(float *addr, float val)
{
    if (__builtin_isnan(val))
    {
        return *addr;
    }
    int *ai = reinterpret_cast<int *>(addr);
    int old = *ai;
    int assumed;

    for (;;)
    {
        assumed = old;
        float oldf = __int_as_float(assumed);
        float newf = fminf(oldf, val);
        if (newf == oldf)
        {
            break;
        }

        int newi = __float_as_int(newf);
        old = atomicCAS(ai, assumed, newi);
        if (old == assumed)
        {
            break;
        }
    }
    return __int_as_float(old);
}

MU_DEV_FUNC MU_FORCE_INLINE float atomicMax(float *addr, float val)
{
    if (__builtin_isnan(val))
    {
        return *addr;
    }
    int *ai = reinterpret_cast<int *>(addr);
    int old = *ai;
    int assumed;

    for (;;)
    {
        assumed = old;
        float oldf = __int_as_float(assumed);
        float newf = fmaxf(oldf, val);
        if (newf == oldf)
        {
            break;
        }

        int newi = __float_as_int(newf);
        old = atomicCAS(ai, assumed, newi);
        if (old == assumed)
        {
            break;
        }
    }
    return __int_as_float(old);
}

MU_DEV_FUNC MU_FORCE_INLINE double atomicMin(double *addr, double val)
{
    if (__builtin_isnan(val))
    {
        return *addr;
    }
    unsigned long long *ai = reinterpret_cast<unsigned long long *>(addr);
    unsigned long long old = *ai;
    unsigned long long assumed;

    for (;;)
    {
        assumed = old;
        double oldd = __longlong_as_double(assumed);
        double newd = fmin(oldd, val);
        if (newd == oldd)
        {
            break;
        }

        unsigned long long newi = __double_as_longlong(newd);
        old = atomicCAS(ai, assumed, newi);
        if (old == assumed)
        {
            break;
        }
    }
    return __longlong_as_double(old);
}

MU_DEV_FUNC MU_FORCE_INLINE double atomicMax(double *addr, double val)
{
    if (__builtin_isnan(val))
    {
        return *addr;
    }
    unsigned long long *ai = reinterpret_cast<unsigned long long *>(addr);
    unsigned long long old = *ai;
    unsigned long long assumed;

    for (;;)
    {
        assumed = old;
        double oldd = __longlong_as_double(assumed);
        double newd = fmax(oldd, val);
        if (newd == oldd)
        {
            break;
        }

        unsigned long long newi = __double_as_longlong(newd);
        old = atomicCAS(ai, assumed, newi);
        if (old == assumed)
        {
            break;
        }
    }
    return __longlong_as_double(old);
}
