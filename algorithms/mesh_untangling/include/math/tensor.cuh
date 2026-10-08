#pragma once
#include <array>
#include <iostream>
#include <limits>
#include <span>
#include <string_view>
#include <tuple>
#include <type_traits>

#include "core/base.cuh"
#include "math/dense_matrix.cuh"
#include "core/meta.cuh"

namespace mu::math
{
namespace detail
{

enum class EinsteinSumIndexRole
{
    kAny = 0,
    kContracted = 1,
    kFree = 2
};

MU_FORCE_INLINE MU_HOST_DEV_FUNC constexpr auto EncodeSymbol(char c) -> std::size_t
{
    if ('a' <= c && c <= 'z')
    {
        return c - 'a';
    }

    if ('A' <= c && c <= 'Z')
    {
        return c - 'A' + 26;
    }

    return std::numeric_limits<std::size_t>::max();
}

MU_FORCE_INLINE MU_HOST_DEV_FUNC constexpr auto DecodeSymbol(std::size_t index) -> char
{
    if (index < 26)
    {
        return static_cast<char>('a' + index);
    }

    if (index < 52)
    {
        return static_cast<char>('A' + index - 26);
    }

    return 0;
}

template <typename Symbol = char>
struct EinsteinSumIndexDescriptor
{
    typedef Symbol Symbol_;
    Symbol symbol_ = 0;
    EinsteinSumIndexRole role_ = EinsteinSumIndexRole::kAny;
    std::size_t extent_ = 0;

    MU_HOST_DEV_FUNC constexpr EinsteinSumIndexDescriptor(
        Symbol symbol = 0, EinsteinSumIndexRole role = EinsteinSumIndexRole::kAny,
        std::size_t extent = 0)
        : symbol_(symbol), role_(role), extent_(extent)
    {
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE auto GetSymbol() -> char &
    {
        return symbol_;
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE constexpr auto GetSymbol() const -> char
    {
        return symbol_;
    };

    MU_HOST_DEV_FUNC MU_FORCE_INLINE auto GetRole() -> EinsteinSumIndexRole &
    {
        return role_;
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE constexpr auto GetRole() const -> EinsteinSumIndexRole
    {
        return role_;
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE auto GetExtent() -> std::size_t &
    {
        return extent_;
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE constexpr auto GetExtent() const -> std::size_t
    {
        return extent_;
    }
};

template <std::size_t kMaxUniqueSymbolCount, typename Symbol = char>
struct EinsteinSumIndexDescriptorPack
{
    EinsteinSumIndexDescriptor<Symbol> indexDescriptor_[kMaxUniqueSymbolCount] = {};
    int symbolToIndexHashMap_[kMaxUniqueSymbolCount] = {};
    std::size_t rank_ = 0;
    std::size_t contractedRank_ = 0;

    constexpr EinsteinSumIndexDescriptorPack() = default;

    MU_HOST_DEV_FUNC MU_FORCE_INLINE auto GetExtent() const -> std::size_t
    {
        std::size_t flattenedSize = 1;
        for (std::size_t i = 0; i < rank_; ++i)
        {
            flattenedSize *= indexDescriptor_[i].GetSize();
        }

        return flattenedSize;
    }

    template <EinsteinSumIndexRole IndexRole>
    MU_HOST_DEV_FUNC constexpr auto GetRank() const -> std::size_t
    {
        if constexpr (IndexRole == EinsteinSumIndexRole::kAny)
        {
            return rank_;
        }

        if constexpr (IndexRole == EinsteinSumIndexRole::kContracted)
        {
            return contractedRank_;
        }

        if constexpr (IndexRole == EinsteinSumIndexRole::kFree)
        {
            return rank_ - contractedRank_;
        }
    }

    MU_FORCE_INLINE MU_HOST_DEV_FUNC constexpr auto AddIndexDescriptor(
        const EinsteinSumIndexDescriptor<Symbol> &parameter) -> void
    {
        indexDescriptor_[rank_] = parameter;

        symbolToIndexHashMap_[EncodeSymbol(parameter.symbol_)] = rank_ + 1;

        ++rank_;
        if (parameter.role_ == EinsteinSumIndexRole::kContracted)
        {
            contractedRank_++;
        }
    }

    MU_FORCE_INLINE MU_HOST_DEV_FUNC constexpr auto GetSymbolRankIndex(char name) const
        -> std::size_t
    {
        return symbolToIndexHashMap_[EncodeSymbol(name)];
    }

    MU_FORCE_INLINE MU_HOST_DEV_FUNC constexpr auto operator[](std::size_t i)
        -> EinsteinSumIndexDescriptor<Symbol> &
    {
        return indexDescriptor_[i];
    }

    MU_FORCE_INLINE MU_HOST_DEV_FUNC constexpr auto operator[](std::size_t i) const
        -> const EinsteinSumIndexDescriptor<Symbol> &
    {
        return indexDescriptor_[i];
    }
};

template <auto kIndexDescriptorPack>
MU_HOST_DEV_FUNC MU_FORCE_INLINE constexpr auto GetSuffixStrideExclusive()
    -> std::array<std::size_t, kIndexDescriptorPack.template GetRank<EinsteinSumIndexRole::kAny>()>
{
    std::size_t suffixExtent = 1;
    std::array<std::size_t, kIndexDescriptorPack.template GetRank<EinsteinSumIndexRole::kAny>()>
        suffixStride = {};
    for (std::size_t i = kIndexDescriptorPack.template GetRank<EinsteinSumIndexRole::kAny>();
         i-- > 0;)
    {
        suffixStride[i] = suffixExtent;
        suffixExtent *= kIndexDescriptorPack.indexDescriptor_[i].GetExtent();
    }

    return suffixStride;
}

template <std::size_t kSize, typename Character = char>
struct EinsteinSumSpecString
{
    typedef Character Character_;
    constexpr static std::size_t kSize_ = kSize;
    Character specString_[kSize] = {};
    MU_HOST_DEV_FUNC constexpr EinsteinSumSpecString(const Character (&specString)[kSize])
    {
        for (std::size_t i = 0; i < kSize_; ++i)
        {
            specString_[i] = specString[i];
        }
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE auto operator[](std::size_t index) -> Character &
    {
        return specString_[index];
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE constexpr auto operator[](std::size_t index) const -> Character
    {
        return specString_[index];
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE constexpr auto GetSize() const -> std::size_t
    {
        return kSize_ - 1;
    }
};

template <typename Character, Character... C>
consteval auto operator""_einsteinsum()
{
    return EinsteinSumSpecString<sizeof...(C) + 1, Character>({C..., Character{}});
};

template <EinsteinSumIndexDescriptor... kUniqueSymbol>
struct EinsteinSumLoop
{
    constexpr static auto Parameter_ = std::tuple{kUniqueSymbol...};
    constexpr static std::size_t kLoopSize_ = (std::size_t{1} * ... * kUniqueSymbol.GetExtent());

    MU_FORCE_INLINE MU_HOST_DEV_FUNC constexpr auto ExpandIndex(std::size_t i) const
        -> std::array<std::size_t, sizeof...(kUniqueSymbol)>
    {
        std::array<std::size_t, sizeof...(kUniqueSymbol)> expandedIndices = {};

        meta::ConstexprLoop<0, sizeof...(kUniqueSymbol)>(
            [&](auto x) -> void
            {
                constexpr std::size_t kIndex =
                    sizeof...(kUniqueSymbol) - meta::ConstexprLoopIndex(x) - 1;
                constexpr std::size_t kExtent = std::get<kIndex>(Parameter_).GetExtent();

                expandedIndices[kIndex] = i % kExtent;
                i /= kExtent;
            });
        return expandedIndices;
    }

    MU_FORCE_INLINE MU_HOST_DEV_FUNC constexpr auto operator()(auto &&fn) const
    {
        for (std::size_t index = 0; index < kLoopSize_; ++index)
        {
            fn.template operator()<kUniqueSymbol...>(ExpandIndex(index));
        }
    }
};

template <auto kIndexDescriptorPack, auto... kUniqueIndexDescriptor>
struct EinsteinSumIndexer
{
    inline constexpr static std::size_t kRank_ =
        kIndexDescriptorPack.template GetRank<EinsteinSumIndexRole::kAny>();
    inline constexpr static auto kUniqueIndexDescriptor_ = std::tuple{kUniqueIndexDescriptor...};

    template <std::size_t kN>
        requires(sizeof...(kUniqueIndexDescriptor) == kN)
    MU_FORCE_INLINE MU_HOST_DEV_FUNC constexpr auto GetLinearIndex(
        const std::array<std::size_t, kN> &indices) const -> std::size_t
    {
        constexpr auto kSuffixStride = detail::GetSuffixStrideExclusive<kIndexDescriptorPack>();

        std::size_t linearIndex = 0;

        meta::ConstexprLoop<0, sizeof...(kUniqueIndexDescriptor)>(
            [&](auto x)
            {
                constexpr std::size_t kIndex = meta::ConstexprLoopIndex(x);

                constexpr char kUniqueIndexDescriptorSymbol =
                    std::get<kIndex>(kUniqueIndexDescriptor_).GetSymbol();

                constexpr std::size_t kUniqueIndexDescriptorSymbolIndex =
                    kIndexDescriptorPack.GetSymbolRankIndex(kUniqueIndexDescriptorSymbol);

                static_assert((kUniqueIndexDescriptorSymbolIndex + 1) !=
                              std::numeric_limits<std::size_t>::max());
                static_assert(kUniqueIndexDescriptorSymbolIndex < (kSuffixStride.size() + 1));

                if constexpr (kUniqueIndexDescriptorSymbolIndex > 0)
                {
                    linearIndex +=
                        indices[kIndex] * kSuffixStride[kUniqueIndexDescriptorSymbolIndex - 1];
                }
            });

        return linearIndex;
    }
};

template <auto kSpecString, typename... Tensor>
struct EinsteinSumSpecStringParser
{
    constexpr static std::size_t kN_ = kSpecString.GetSize();
    constexpr static std::size_t kMaxUniqueNameCount_ = 52;

    constexpr static std::array<std::span<const std::size_t>, sizeof...(Tensor)> kTensorDimension_ =
        {std::span{Tensor::kDimension_}...};

    EinsteinSumIndexDescriptorPack<kMaxUniqueNameCount_,
                                   typename std::remove_cvref_t<decltype(kSpecString)>::Character_>
        indexDescriptorPack_[kN_] = {};
    EinsteinSumIndexDescriptorPack<kMaxUniqueNameCount_,
                                   typename std::remove_cvref_t<decltype(kSpecString)>::Character_>
        uniqueIndexDescriptorPack_;
    std::size_t parsedOperandCount_ = 0;
    bool valid_ = true;

    MU_HOST_DEV_FUNC constexpr EinsteinSumSpecStringParser()
    {
        int symbolCount[kMaxUniqueNameCount_] = {};
        std::size_t uniqueNameDimensionSize[kMaxUniqueNameCount_] = {};

        for (std::size_t i = 0; i < kN_; ++i)
        {
            std::size_t symbolCode = EncodeSymbol(kSpecString[i]);
            if (symbolCode != std::numeric_limits<std::size_t>::max())
            {
                symbolCount[symbolCode]++;
                valid_ &= symbolCount[symbolCode] <= 2;
            }
        }

        if (!valid_)
        {
            return;
        }

        bool foundArrow = false;
        for (std::size_t i = 0; i < kN_; ++i)
        {
            char symbol = kSpecString[i];
            std::size_t symbolCode = EncodeSymbol(symbol);

            if (symbolCode != std::numeric_limits<std::size_t>::max())
            {
                std::size_t dimensionSize =
                    kTensorDimension_[parsedOperandCount_]
                        .data()[indexDescriptorPack_[parsedOperandCount_]
                                    .template GetRank<EinsteinSumIndexRole::kAny>()];
                indexDescriptorPack_[parsedOperandCount_].AddIndexDescriptor(
                    {symbol,
                     symbolCount[symbolCode] == 2 ? EinsteinSumIndexRole::kContracted
                                                  : EinsteinSumIndexRole::kFree,
                     dimensionSize});
                if (!uniqueNameDimensionSize[EncodeSymbol(symbol)])
                {
                    uniqueNameDimensionSize[EncodeSymbol(symbol)] = dimensionSize;
                }
                else
                {
                    valid_ &= (uniqueNameDimensionSize[EncodeSymbol(symbol)] == dimensionSize);
                }
            }
            else
            {
                if (symbol == ',')
                {
                    valid_ &= indexDescriptorPack_[parsedOperandCount_]
                                  .template GetRank<EinsteinSumIndexRole::kAny>() > 0;
                    valid_ &= kTensorDimension_[parsedOperandCount_].size() ==
                              indexDescriptorPack_[parsedOperandCount_]
                                  .template GetRank<EinsteinSumIndexRole::kAny>();

                    ++parsedOperandCount_;
                }
                else if (symbol == '-')
                {
                    valid_ &= ((i + 1) < kN_) && (!foundArrow);
                    if (!valid_)
                    {
                        break;
                    }

                    valid_ &= kSpecString[i + 1] == '>';
                    if (!valid_)
                    {
                        break;
                    }

                    foundArrow = true;
                    ++parsedOperandCount_;
                    i++;
                }
                else if (symbol != ' ')
                {
                    valid_ = false;
                }
            }

            if (!valid_)
            {
                break;
            }
        }

        valid_ &= foundArrow;
        if (!valid_)
        {
            return;
        }

        for (std::size_t i = 0; i < kMaxUniqueNameCount_; ++i)
        {
            if (symbolCount[i] > 0)
            {
                uniqueIndexDescriptorPack_.AddIndexDescriptor(
                    {DecodeSymbol(i),
                     symbolCount[i] == 2 ? EinsteinSumIndexRole::kContracted
                                         : EinsteinSumIndexRole::kFree,
                     uniqueNameDimensionSize[i]});
            }
        }
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE constexpr auto GetUniqueIndexDescriptorPack() const
    {
        return uniqueIndexDescriptorPack_;
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE constexpr auto GetIndexDescriptor(std::size_t index) const
        -> const EinsteinSumIndexDescriptorPack<kMaxUniqueNameCount_> &
    {
        return indexDescriptorPack_[index];
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE constexpr auto GetOperandCount() const -> std::size_t
    {
        return parsedOperandCount_;
    }

    MU_HOST_DEV_FUNC MU_FORCE_INLINE constexpr auto GetValidity() const -> bool
    {
        return valid_;
    }
};

template <auto kSpecStringParser>
struct EinsteinSumHelper
{
    constexpr static auto kUniqueIndexDescriptorPack_ =
        kSpecStringParser.GetUniqueIndexDescriptorPack();

    template <auto kUniqueNameParameterPack, std::size_t... kIndex>
    static MU_HOST_DEV_FUNC MU_FORCE_INLINE constexpr auto BuildEinsteinSumLoopHelper(
        std::index_sequence<kIndex...>)
    {
        return EinsteinSumLoop<kUniqueNameParameterPack[kIndex]...>{};
    }

    static MU_HOST_DEV_FUNC MU_FORCE_INLINE constexpr auto BuildEinsteinSumLoop() -> auto
    {
        return BuildEinsteinSumLoopHelper<kUniqueIndexDescriptorPack_>(
            std::make_index_sequence<kUniqueIndexDescriptorPack_.template GetRank<
                detail::EinsteinSumIndexRole::kAny>()>{});
    }
};

template <typename T, typename... U>
struct HomogenousTypeHelper : std::conjunction<std::is_same<T, U>...>
{
};

template <typename... T>
concept HomogenousType = HomogenousTypeHelper<T...>::value;
}

using detail::operator""_einsteinsum;

template <auto kSpecString, typename... Tensor>
    requires(detail::HomogenousType<typename std::remove_cvref_t<Tensor>::Scalar_...> &&
             sizeof...(Tensor) > 1)
MU_HOST_DEV_FUNC auto EinsteinSum(Tensor &&...tensor) -> void
{
    typedef std::common_type_t<typename std::remove_cvref_t<Tensor>::Scalar_...> Scalar;
    constexpr static auto kSpecStringParser =
        detail::EinsteinSumSpecStringParser<kSpecString, std::remove_cvref_t<Tensor>...>();
#ifndef MU_DISABLE_EINSTEIN_SUM_VALIDATION
    static_assert(kSpecStringParser.GetValidity(),
                  "EinsteinSum specification does not match operand ranks.");
#endif

    constexpr static auto kEinsteinSumLoop =
        detail::EinsteinSumHelper<kSpecStringParser>::BuildEinsteinSumLoop();

    constexpr static std::size_t kUniqueNameCount =
        kSpecStringParser.GetUniqueIndexDescriptorPack()
            .template GetRank<detail::EinsteinSumIndexRole::kAny>();

    kEinsteinSumLoop(
        [&]<auto... kIndexDescriptor>(auto &&indices) -> void
        {
            Scalar innerProduct = Scalar{1};
            meta::ConstexprLoop<0, sizeof...(Tensor) - 1>(
                [&](auto x) -> void
                {
                    constexpr int kOperandIndex = meta::ConstexprLoopIndex(x);
                    constexpr auto kTensorIndexer = detail::EinsteinSumIndexer<
                        kSpecStringParser.GetIndexDescriptor(kOperandIndex), kIndexDescriptor...>{};
                    innerProduct *= std::get<kOperandIndex>(std::forward_as_tuple(
                        std::forward<Tensor>(tensor)...))[kTensorIndexer.GetLinearIndex(indices)];
                });

            constexpr std::size_t kOutputOperandIndex = sizeof...(Tensor) - 1;
            constexpr auto kOutputTensorIndexer = detail::EinsteinSumIndexer<
                kSpecStringParser.GetIndexDescriptor(kOutputOperandIndex), kIndexDescriptor...>{};
            std::get<kOutputOperandIndex>(std::forward_as_tuple(std::forward<Tensor>(
                tensor)...))[kOutputTensorIndexer.GetLinearIndex(indices)] += innerProduct;
        });
}

namespace detail
{
template <size_t kUnrollSize, size_t kTotalSize, typename T, typename Func>
MU_FORCE_INLINE MU_HOST_DEV_FUNC auto TensorElementwiseUnaryOperationUnroller(T &lhs,
                                                                              Func &&op) -> void
{
    if constexpr (kTotalSize <= kUnrollSize)
    {
        meta::ConstexprLoop<0, kTotalSize>(
            [&](auto x) -> void
            {
                constexpr static int index = meta::ConstexprLoopIndex(x);
                op(lhs[index]);
            });
    }
    else
    {
        size_t i = 0;
        do
        {
            meta::ConstexprLoop<0, kUnrollSize>(
                [&](auto x) -> void
                {
                    constexpr static int index = meta::ConstexprLoopIndex(x);
                    op(lhs[i + index]);
                });
            i += kUnrollSize;
        } while (i < kTotalSize - kUnrollSize);

        for (; i < kTotalSize; ++i)
        {
            op(lhs[i]);
        }
    }
}

template <size_t kUnrollSize, size_t kTotalSize, typename T, typename U, typename Func>
    requires(std::is_scalar_v<U>)
MU_FORCE_INLINE MU_HOST_DEV_FUNC auto TensorElementwiseBinaryOperationUnroller(T &lhs, U rhs,
                                                                               Func &&op) -> void
{
    if constexpr (kTotalSize <= kUnrollSize)
    {
        meta::ConstexprLoop<0, kTotalSize>(
            [&](auto x) -> void
            {
                constexpr static int index = meta::ConstexprLoopIndex(x);
                op(lhs[index], rhs);
            });
    }
    else
    {
        size_t i = 0;
        do
        {
            meta::ConstexprLoop<0, kUnrollSize>(
                [&](auto x) -> void
                {
                    constexpr static int index = meta::ConstexprLoopIndex(x);
                    op(lhs[i + index], rhs);
                });
            i += kUnrollSize;
        } while (i < kTotalSize - kUnrollSize);

        for (; i < kTotalSize; ++i)
        {
            op(lhs[i], rhs);
        }
    }
}

template <size_t kUnrollSize, size_t kTotalSize, typename T, typename U, typename Func>
    requires(!std::is_scalar_v<U>)
MU_FORCE_INLINE MU_HOST_DEV_FUNC auto TensorElementwiseBinaryOperationUnroller(T &lhs, const U &rhs,
                                                                               Func &&op) -> void
{
    if constexpr (kTotalSize <= kUnrollSize)
    {
        meta::ConstexprLoop<0, kTotalSize>(
            [&](auto x) -> void
            {
                constexpr static int index = meta::ConstexprLoopIndex(x);
                op(lhs[index], rhs[index]);
            });
    }
    else
    {
        size_t i = 0;
        do
        {
            meta::ConstexprLoop<0, kUnrollSize>(
                [&](auto x) -> void
                {
                    constexpr static int index = meta::ConstexprLoopIndex(x);
                    op(lhs[i + index], rhs[i + index]);
                });
            i += kUnrollSize;
        } while (i < kTotalSize - kUnrollSize);

        for (; i < kTotalSize; ++i)
        {
            op(lhs[i], rhs[i]);
        }
    }
}
}

template <typename Scalar, size_t... kDimension>
class Tensor
{
public:
    typedef Scalar Scalar_;
    constexpr static int kUnrollSize_ = 16;
    constexpr static auto kDimension_ =
        std::array<std::size_t, sizeof...(kDimension)>{kDimension...};
    constexpr static auto kExtent_ = (kDimension * ... * std::size_t{1});

    constexpr Tensor() = default;

    template <typename VectorScalar, std::size_t kN>
        requires(sizeof...(kDimension) == 1 && kDimension_[0] == kN)
    constexpr explicit MU_HOST_DEV_FUNC Tensor(const DenseVector<VectorScalar, kN> &x)
    {
        for (std::size_t i = 0; i < kExtent_; ++i)
        {
            data_[i] = x[i];
        }
    }

    template <typename VectorScalar, std::size_t kN, std::size_t kM>
        requires(kExtent_ == kN * kM && sizeof...(kDimension) == 2 && kDimension_[0] == kN &&
                 kDimension_[1] == kM)
    constexpr explicit MU_HOST_DEV_FUNC Tensor(const DenseMatrix<VectorScalar, kN, kM> &x)
    {
        for (std::size_t i = 0; i < kExtent_; ++i)
        {
            data_[i] = x[i];
        }
    }

    template <typename T>
        requires std::is_convertible_v<T, Scalar_> && std::is_scalar_v<T>
    constexpr MU_HOST_DEV_FUNC Tensor(T x)
    {
        for (std::size_t i = 0; i < kExtent_; ++i)
        {
            data_[i] = x;
        }
    }

    template <typename... T>
        requires(sizeof...(T) == kExtent_) && (std::is_scalar_v<T> && ...) &&
                (std::is_convertible_v<T, Scalar> && ...)
    constexpr MU_HOST_DEV_FUNC Tensor(T... data) : data_{static_cast<Scalar>(data)...}
    {
    }

    constexpr MU_HOST_DEV_FUNC Tensor(const Tensor &rhs)
    {
        *this = rhs;
    }

    template <typename T>
        requires std::is_convertible_v<T, Scalar_> && (!std::is_same_v<T, Scalar_>)
    constexpr MU_HOST_DEV_FUNC Tensor(const Tensor<T, kDimension...> &rhs)
    {
        detail::TensorElementwiseBinaryOperationUnroller<kUnrollSize_, kExtent_>(
            *this, rhs,
            [&](Scalar_ &x, Scalar_ y) -> void
            {
                x = static_cast<Scalar_>(y);
            });
    }

    MU_FORCE_INLINE MU_HOST_DEV_FUNC auto operator[](size_t index) -> Scalar_ &
    {
        return data_[index];
    }

    MU_FORCE_INLINE MU_HOST_DEV_FUNC auto operator[](size_t index) const -> const Scalar_
    {
        return data_[index];
    }

    template <typename T>
        requires std::is_scalar_v<T> && std::is_convertible_v<T, Scalar_>
    constexpr MU_HOST_DEV_FUNC auto operator=(T rhs) -> Tensor &
    {
        detail::TensorElementwiseUnaryOperationUnroller<kUnrollSize_, kExtent_>(
            *this,
            [&](Scalar_ &x) -> void
            {
                x = rhs;
            });
        return *this;
    }

    template <typename T>
        requires std::is_convertible_v<T, Scalar_>
    MU_HOST_DEV_FUNC auto operator=(const Tensor<T, kExtent_> &rhs) -> Tensor &
    {
        detail::TensorElementwiseBinaryOperationUnroller<kUnrollSize_, kExtent_>(
            *this, rhs,
            [&](Scalar_ &x, Scalar_ y) -> void
            {
                x = static_cast<Scalar_>(y);
            });
        return *this;
    }

    MU_HOST_DEV_FUNC auto operator+=(const Tensor &rhs) -> void
    {
        detail::TensorElementwiseBinaryOperationUnroller<kUnrollSize_, kExtent_>(
            *this, rhs,
            [&](Scalar_ &x, Scalar_ y) -> void
            {
                x += y;
            });
    }

    MU_HOST_DEV_FUNC auto operator-=(const Tensor &rhs) -> void
    {
        detail::TensorElementwiseBinaryOperationUnroller<kUnrollSize_, kExtent_>(
            *this, rhs,
            [&](Scalar_ &x, Scalar_ y) -> void
            {
                x -= y;
            });
    }

    MU_HOST_DEV_FUNC auto operator*=(const Tensor &rhs) -> void
    {
        detail::TensorElementwiseBinaryOperationUnroller<kUnrollSize_, kExtent_>(
            *this, rhs,
            [&](Scalar_ &x, Scalar_ y) -> void
            {
                x *= y;
            });
    }

    MU_HOST_DEV_FUNC auto operator/=(const Tensor &rhs) -> void
    {
        detail::TensorElementwiseBinaryOperationUnroller<kUnrollSize_, kExtent_>(
            *this, rhs,
            [&](Scalar_ &x, Scalar_ y) -> void
            {
                x /= y;
            });
    }

    MU_HOST_DEV_FUNC auto Dot(const Tensor &rhs) const -> Scalar_
    {
        Scalar_ dot = 0;
        detail::TensorElementwiseBinaryOperationUnroller<kUnrollSize_, kExtent_>(
            *this, rhs,
            [&](Scalar_ x, Scalar_ y) -> void
            {
                dot += x * y;
            });
        return dot;
    }

    template <typename T>
        requires std::is_scalar_v<T>
    MU_HOST_DEV_FUNC auto operator+=(T rhs) -> void
    {
        detail::TensorElementwiseUnaryOperationUnroller<kUnrollSize_, kExtent_>(
            *this,
            [&](Scalar_ &x) -> void
            {
                x += rhs;
            });
    }

    template <typename T>
        requires std::is_scalar_v<T>
    MU_HOST_DEV_FUNC auto operator-=(T rhs) -> void
    {
        detail::TensorElementwiseUnaryOperationUnroller<kUnrollSize_, kExtent_>(
            *this,
            [&](Scalar_ &x) -> void
            {
                x -= rhs;
            });
    }

    template <typename T>
        requires std::is_scalar_v<T>
    MU_HOST_DEV_FUNC auto operator*=(T rhs) -> void
    {
        detail::TensorElementwiseUnaryOperationUnroller<kUnrollSize_, kExtent_>(
            *this,
            [&](Scalar_ &x) -> void
            {
                x *= rhs;
            });
    }

    template <typename T>
        requires std::is_scalar_v<T>
    MU_HOST_DEV_FUNC auto operator/=(T rhs) -> void
    {
        detail::TensorElementwiseUnaryOperationUnroller<kUnrollSize_, kExtent_>(
            *this,
            [&](Scalar_ &x) -> void
            {
                x /= rhs;
            });
    }

    MU_HOST_DEV_FUNC auto operator+(const Tensor &rhs) const -> Tensor
    {
        Tensor res(*this);
        res += rhs;
        return res;
    }

    MU_HOST_DEV_FUNC auto operator-(const Tensor &rhs) const -> Tensor
    {
        Tensor res(*this);
        res -= rhs;
        return res;
    }

    MU_HOST_DEV_FUNC auto operator-() const -> Tensor
    {
        Tensor res;
        detail::TensorElementwiseBinaryOperationUnroller<kUnrollSize_, kExtent_>(
            res, *this,
            [&](Scalar_ &x, Scalar_ y) -> void
            {
                x = -y;
            });
        return res;
    }

    MU_HOST_DEV_FUNC auto operator*(const Tensor &rhs) const -> Tensor
    {
        Tensor res(*this);
        res *= rhs;
        return res;
    }

    MU_HOST_DEV_FUNC auto operator/(const Tensor &rhs) const -> Tensor
    {
        Tensor res(*this);
        res /= rhs;
        return res;
    }


    MU_FORCE_INLINE MU_HOST_DEV_FUNC auto Norm() const -> Scalar_
    {
        Scalar_ res = 0;
        detail::TensorElementwiseUnaryOperationUnroller<kUnrollSize_, kExtent_>(
            *this,
            [&](Scalar_ x) -> void
            {
                res += x * x;
            });
        return sqrt(res);
    }

    MU_FORCE_INLINE MU_HOST_DEV_FUNC auto Normalize() const -> Tensor<Scalar_, kDimension...>
    {
        auto res = *this;
        res /= res.Norm();
        return res;
    }

private:
    Scalar_ data_[kExtent_] = {};
};


template <typename VectorScalar, std::size_t kN>
explicit Tensor(const DenseVector<VectorScalar, kN> &x) -> Tensor<VectorScalar, kN>;

template <typename VectorScalar, std::size_t kN, std::size_t kM>
explicit Tensor(const DenseMatrix<VectorScalar, kN, kM> &x) -> Tensor<VectorScalar, kN, kM>;

template <typename Scalar>
MU_FORCE_INLINE MU_HOST_DEV_FUNC auto GetLeviCivitaTensor() -> Tensor<Scalar, 3, 3, 3>
{
    constexpr Tensor<Scalar, 3, 3, 3> leviCivitaTensor{0, 0,
                                                       0, 0,
                                                       0, static_cast<Scalar>(1),
                                                       0, static_cast<Scalar>(-1),
                                                       0, 0,
                                                       0, -static_cast<Scalar>(1),
                                                       0, 0,
                                                       0, static_cast<Scalar>(1),
                                                       0, 0,
                                                       0, static_cast<Scalar>(1),
                                                       0, -static_cast<Scalar>(1),
                                                       0, 0,
                                                       0, 0,
                                                       0};

    return leviCivitaTensor;
}


template <typename T, size_t... Dimension, typename U>
    requires std::is_scalar_v<U> && std::is_convertible_v<U, T>
MU_HOST_DEV_FUNC auto operator+(const Tensor<T, Dimension...> &lhs,
                                const U rhs) -> Tensor<T, Dimension...>
{
    Tensor<T, Dimension...> res(lhs);
    res += static_cast<T>(rhs);
    return res;
}

template <typename T, size_t... Dimension, typename U>
    requires std::is_scalar_v<U> && std::is_convertible_v<U, T>
MU_HOST_DEV_FUNC auto operator+(const U lhs,
                                const Tensor<T, Dimension...> &rhs) -> Tensor<T, Dimension...>
{
    Tensor<T, Dimension...> res(rhs);
    res += static_cast<T>(lhs);
    return res;
}

template <typename T, size_t... Dimension, typename U>
    requires std::is_scalar_v<U> && std::is_convertible_v<U, T>
MU_HOST_DEV_FUNC auto operator-(const Tensor<T, Dimension...> &lhs,
                                const U rhs) -> Tensor<T, Dimension...>
{
    Tensor<T, Dimension...> res(lhs);
    res -= rhs;
    return res;
}

template <typename T, size_t... Dimension, typename U>
    requires std::is_scalar_v<U> && std::is_convertible_v<U, T>
MU_HOST_DEV_FUNC auto operator-(const U lhs,
                                const Tensor<T, Dimension...> &rhs) -> Tensor<T, Dimension...>
{
    Tensor<T, Dimension...> res{};
    detail::TensorElementwiseBinaryOperationUnroller<Tensor<T, Dimension...>::kUnrollSize_,
                                                     Tensor<T, Dimension...>::kExtent_>(
        res, rhs,
        [&](T &x, U y) -> void
        {
            x = lhs - y;
        });

    return res;
}

template <typename T, size_t... Dimension, typename U>
    requires std::is_scalar_v<U> && std::is_convertible_v<U, T>
MU_HOST_DEV_FUNC auto operator*(const Tensor<T, Dimension...> &lhs,
                                const U rhs) -> Tensor<T, Dimension...>
{
    Tensor<T, Dimension...> res(lhs);
    res *= rhs;
    return res;
}

template <typename T, size_t... Dimension, typename U>
    requires std::is_scalar_v<U> && std::is_convertible_v<U, T>
MU_HOST_DEV_FUNC auto operator*(const U lhs,
                                const Tensor<T, Dimension...> &rhs) -> Tensor<T, Dimension...>
{
    Tensor<T, Dimension...> res(rhs);
    res *= lhs;
    return res;
}

template <typename T, size_t... Dimension, typename U>
    requires std::is_scalar_v<U> && std::is_convertible_v<U, T>
MU_HOST_DEV_FUNC auto operator/(const Tensor<T, Dimension...> &lhs,
                                const U rhs) -> Tensor<T, Dimension...>
{
    Tensor<T, Dimension...> res(lhs);
    res /= rhs;
    return res;
}

template <typename T, size_t... Dimension, typename U>
    requires std::is_scalar_v<U> && std::is_convertible_v<U, T>
MU_HOST_DEV_FUNC auto operator/(const U lhs,
                                const Tensor<T, Dimension...> &rhs) -> Tensor<T, Dimension...>
{
    Tensor<T, Dimension...> res{};
    detail::TensorElementwiseBinaryOperationUnroller<Tensor<T, Dimension...>::kUnrollSize_,
                                                     Tensor<T, Dimension...>::kExtent_>(
        res, rhs,
        [&](T &x, U y) -> void
        {
            x = lhs / y;
        });

    return res;
}

}
