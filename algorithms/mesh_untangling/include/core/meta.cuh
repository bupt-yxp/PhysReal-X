#pragma once
#include <type_traits>

#include "core/base.cuh"

namespace mu::meta
{

namespace internal
{
template <typename T, T Index>
struct ConstexprIndex
{
    static constexpr T kIndex_ = Index;
};

template <int kBegin, int kStep, int... Indices>
MU_HOST_DEV_FUNC MU_FORCE_INLINE auto ConstexprLoopImpl(
    auto func, std::integer_sequence<int, Indices...>) -> void
{
    (func(internal::ConstexprIndex<int, Indices * kStep + kBegin>{}), ...);
}
}

template <typename T>
constexpr MU_FORCE_INLINE MU_HOST_DEV_FUNC auto ConstexprLoopIndex(T) -> int
{
    return T::kIndex_;
}

template <int kBegin, int kEnd, int kStep = 1>
MU_HOST_DEV_FUNC MU_FORCE_INLINE auto ConstexprLoop(auto func) -> void
{
    internal::ConstexprLoopImpl<kBegin, kStep>(
        func, std::make_integer_sequence<int, int(double(kEnd - kBegin) / kStep + 0.5)>{});
}

template <typename... Attribute>
struct AttributeWrapper
{
private:
    template <size_t I, size_t N, typename... Ts>
        requires(I <= N)
    struct AttributeOffsetWrapper
    {
    private:
        consteval inline static auto ComputeOffset() -> size_t
        {
            if constexpr (I == 0)
            {
                return 0;
            }
            else
            {
                return AttributeOffsetWrapper<I - 1, N, Ts...>::kOffset_ +
                       sizeof(std::tuple_element_t<I - 1, std::tuple<Ts...>>);
            }
        }

    public:
        constexpr static size_t kOffset_ = ComputeOffset();
    };

public:
    template <size_t AttributeIndex>
    struct AttributeWrapperAccessor
    {
        typedef std::tuple_element_t<AttributeIndex, std::tuple<Attribute...>> Type;
        constexpr static size_t kOffset_ =
            AttributeOffsetWrapper<AttributeIndex, AttributeIndex, Attribute...>::kOffset_;
        constexpr static size_t kSize_ = sizeof(Type);
    };

    constexpr static size_t kSize_ = (sizeof(Attribute) + ...);
};

template <typename T>
    requires(std::is_integral_v<T>)
MU_HOST_DEV_FUNC constexpr auto GetNextPowerOf2(T x) -> T
{
    using U = std::make_unsigned_t<T>;

    if (x <= T{1})
    {
        return T{1};
    }

    U v = static_cast<U>(x);

    constexpr unsigned kDigits = std::numeric_limits<U>::digits;
    if (v > (U{1} << (kDigits - 1)))
    {
        return T{0};
    }

    --v;

#if defined(__CUDA_ARCH__)
#pragma unroll
#endif
    for (U i = 1; i < std::numeric_limits<U>::digits; i <<= 1)
    {
        v |= v >> i;
    }

    ++v;

    return static_cast<T>(v);
}

template <typename T, typename U, std::size_t kUnfoldSize, std::size_t... kIndex>
MU_CONSTEXPR MU_HOST_DEV_FUNC MU_FORCE_INLINE auto InitializeTypeStructureOfArrayHelper(
    const U (&soa)[kUnfoldSize], std::size_t index,
    std::integer_sequence<std::size_t, kIndex...>) -> T
{
    return T{soa[kIndex][index]...};
}

template <typename T, typename U, std::size_t kUnfoldSize>
MU_CONSTEXPR MU_HOST_DEV_FUNC MU_FORCE_INLINE auto InitializeTypeStructureOfArray(
    const U (&soa)[kUnfoldSize], std::size_t index) -> T
{
    return InitializeTypeStructureOfArrayHelper<T>(soa, index,
                                                   std::make_index_sequence<kUnfoldSize>());
}

template <typename T, typename U, typename W, std::size_t... kIndex>
MU_CONSTEXPR MU_HOST_DEV_FUNC MU_FORCE_INLINE auto InitializeTypeStructureOfArrayHelper(
    const U &soa, const W &index, std::integer_sequence<std::size_t, kIndex...>) -> T
{
    return T{soa[index[kIndex]]...};
}

template <typename T, std::size_t kUnfoldSize, typename U, typename W>
MU_CONSTEXPR MU_HOST_DEV_FUNC MU_FORCE_INLINE auto InitializeTypeStructureOfArray(
    const U &soa, const W &index) -> T
{
    return InitializeTypeStructureOfArrayHelper<T>(soa, index,
                                                   std::make_index_sequence<kUnfoldSize>());
}

}
