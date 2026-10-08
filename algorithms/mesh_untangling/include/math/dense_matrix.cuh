#pragma once
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <type_traits>

#include "core/base.cuh"
#include "core/meta.cuh"

namespace mu::math
{
template <typename ScalarType, size_t kN, size_t kM, bool kIsDense>
class Matrix;

template <typename ScalarType, size_t kN, bool kIsDense>
using Vector = Matrix<ScalarType, kN, 1, kIsDense>;

template <typename ScalarType, size_t kN, size_t kM, bool kIsDense>
class Matrix;

template <typename ScalarType, size_t kN, size_t kM>
using DenseMatrix = Matrix<ScalarType, kN, kM, true>;

template <typename ScalarType, size_t kN>
using DenseVector = Matrix<ScalarType, kN, 1, true>;

namespace detail
{
template <size_t kUnrollSize, size_t kTotalSize, typename T, typename Func>
MU_FORCE_INLINE MU_HOST_DEV_FUNC auto MatrixElementwiseUnaryOperationUnroller(T &lhs,
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
MU_FORCE_INLINE MU_HOST_DEV_FUNC auto MatrixElementwiseBinaryOperationUnroller(T &lhs, U rhs,
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
MU_FORCE_INLINE MU_HOST_DEV_FUNC auto MatrixElementwiseBinaryOperationUnroller(T &lhs, const U &rhs,
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

enum class MatrixMemoryAlignment
{
    kRowMajor = 0,
    kColumnMajor = 1,
    kMaxEnum = 2
};

template <typename ScalarType, size_t kN, size_t kM>
class Matrix<ScalarType, kN, kM, true>
{
public:
    typedef ScalarType ScalarType_;
    constexpr static int kUnrollSize_ = 16;
    constexpr static auto kN_ = kN;
    constexpr static auto kM_ = kM;
    constexpr static auto kSize_ = kN * kM;

    Matrix() = default;

    template <typename... ScalarTypes>
        requires(sizeof...(ScalarTypes) == kN * kM) && (std::is_scalar_v<ScalarTypes> && ...) &&
                (std::is_convertible_v<ScalarTypes, ScalarType> && ...)
    constexpr MU_HOST_DEV_FUNC Matrix(ScalarTypes... data) : data_{static_cast<ScalarType>(data)...}
    {
    }

    constexpr MU_HOST_DEV_FUNC Matrix(const Matrix &rhs) = default;

    template <typename T>
        requires std::is_convertible_v<T, ScalarType_> && (!std::is_same_v<T, ScalarType_>)
    constexpr MU_HOST_DEV_FUNC Matrix(const Matrix<T, kN_, kM_, true> &rhs)
    {
        detail::MatrixElementwiseBinaryOperationUnroller<kUnrollSize_, kN_ * kM_>(
            *this, rhs,
            [&](ScalarType_ &x, ScalarType_ y) -> void
            {
                x = static_cast<ScalarType_>(y);
            });
    }

    constexpr MU_HOST_DEV_FUNC auto operator=(const Matrix &rhs) -> Matrix & = default;

    MU_FORCE_INLINE MU_HOST_DEV_FUNC auto operator[](size_t index) -> ScalarType_ &
    {
        return data_[index];
    }

    MU_FORCE_INLINE MU_HOST_DEV_FUNC auto operator[](size_t index) const -> const ScalarType_
    {
        return data_[index];
    }

    MU_FORCE_INLINE MU_HOST_DEV_FUNC auto GetData() -> ScalarType_ *
    {
        return data_;
    }

    MU_FORCE_INLINE MU_HOST_DEV_FUNC auto GetData() const -> const ScalarType_ *
    {
        return data_;
    }

    template <typename T>
        requires std::is_scalar_v<T> && std::is_convertible_v<T, ScalarType_>
    constexpr MU_HOST_DEV_FUNC auto operator=(T rhs) -> Matrix &
    {
        detail::MatrixElementwiseUnaryOperationUnroller<kUnrollSize_, kN_ * kM_>(
            *this,
            [&](ScalarType_ &x) -> void
            {
                x = rhs;
            });
        return *this;
    }

    template <typename T>
        requires std::is_convertible_v<T, ScalarType_>
    MU_HOST_DEV_FUNC auto operator=(const Matrix<T, kN_, kM_, true> &rhs) -> Matrix &
    {
        detail::MatrixElementwiseBinaryOperationUnroller<kUnrollSize_, kN_ * kM_>(
            *this, rhs,
            [&](ScalarType_ &x, ScalarType_ y) -> void
            {
                x = static_cast<ScalarType_>(y);
            });
        return *this;
    }

    MU_HOST_DEV_FUNC auto operator+=(const Matrix &rhs) -> void
    {
        detail::MatrixElementwiseBinaryOperationUnroller<kUnrollSize_, kN_ * kM_>(
            *this, rhs,
            [&](ScalarType_ &x, ScalarType_ y) -> void
            {
                x += y;
            });
    }

    MU_HOST_DEV_FUNC auto operator-=(const Matrix &rhs) -> void
    {
        detail::MatrixElementwiseBinaryOperationUnroller<kUnrollSize_, kN_ * kM_>(
            *this, rhs,
            [&](ScalarType_ &x, ScalarType_ y) -> void
            {
                x -= y;
            });
    }

    MU_HOST_DEV_FUNC auto operator*=(const Matrix &rhs) -> void
    {
        detail::MatrixElementwiseBinaryOperationUnroller<kUnrollSize_, kN_ * kM_>(
            *this, rhs,
            [&](ScalarType_ &x, ScalarType_ y) -> void
            {
                x *= y;
            });
    }

    MU_HOST_DEV_FUNC auto operator/=(const Matrix &rhs) -> void
    {
        detail::MatrixElementwiseBinaryOperationUnroller<kUnrollSize_, kN_ * kM_>(
            *this, rhs,
            [&](ScalarType_ &x, ScalarType_ y) -> void
            {
                x /= y;
            });
    }

    template <typename T>
        requires std::is_scalar_v<T>
    MU_HOST_DEV_FUNC auto operator+=(T rhs) -> void
    {
        detail::MatrixElementwiseUnaryOperationUnroller<kUnrollSize_, kN_ * kM_>(
            *this,
            [&](ScalarType_ &x) -> void
            {
                x += rhs;
            });
    }

    template <typename T>
        requires std::is_scalar_v<T>
    MU_HOST_DEV_FUNC auto operator-=(T rhs) -> void
    {
        detail::MatrixElementwiseUnaryOperationUnroller<kUnrollSize_, kN_ * kM_>(
            *this,
            [&](ScalarType_ &x) -> void
            {
                x -= rhs;
            });
    }

    template <typename T>
        requires std::is_scalar_v<T>
    MU_HOST_DEV_FUNC auto operator*=(T rhs) -> void
    {
        detail::MatrixElementwiseUnaryOperationUnroller<kUnrollSize_, kN_ * kM_>(
            *this,
            [&](ScalarType_ &x) -> void
            {
                x *= rhs;
            });
    }

    template <typename T>
        requires std::is_scalar_v<T>
    MU_HOST_DEV_FUNC auto operator/=(T rhs) -> void
    {
        detail::MatrixElementwiseUnaryOperationUnroller<kUnrollSize_, kN_ * kM_>(
            *this,
            [&](ScalarType_ &x) -> void
            {
                x /= rhs;
            });
    }

    MU_HOST_DEV_FUNC auto operator+(const Matrix &rhs) const -> Matrix
    {
        Matrix res(*this);
        res += rhs;
        return res;
    }

    MU_HOST_DEV_FUNC auto operator-(const Matrix &rhs) const -> Matrix
    {
        Matrix res(*this);
        res -= rhs;
        return res;
    }

    MU_HOST_DEV_FUNC auto operator-() const -> Matrix
    {
        Matrix res;
        detail::MatrixElementwiseBinaryOperationUnroller<kUnrollSize_, kN_ * kM_>(
            res, *this,
            [&](ScalarType_ &x, ScalarType_ y) -> void
            {
                x = -y;
            });
        return res;
    }

    MU_HOST_DEV_FUNC auto operator*(const Matrix &rhs) const -> Matrix
    {
        Matrix res(*this);
        res *= rhs;
        return res;
    }

    MU_HOST_DEV_FUNC auto operator/(const Matrix &rhs) const -> Matrix
    {
        Matrix res(*this);
        res /= rhs;
        return res;
    }


    template <size_t kP>
        requires(kN_ != 1 || kP != 1)
    MU_HOST_DEV_FUNC auto MatrixMultiplication(const Matrix<ScalarType_, kM_, kP, true> &rhs) const
        -> Matrix<ScalarType_, kN_, kP, true>
    {
        Matrix<ScalarType_, kN_, kP, true> res;

        for (size_t i = 0; i < kN_; ++i)
        {
            for (size_t k = 0; k < kP; ++k)
            {
                for (size_t j = 0; j < kM_; ++j)
                {
                    res[i * kP + k] += data_[i * kM_ + j] * rhs[j * kP + k];
                }
            }
        }
        return res;
    }

    MU_HOST_DEV_FUNC static auto Identity() -> Matrix
        requires(kN_ == kM_)
    {
        Matrix<ScalarType_, kN_, kN_, true> I;
        for (size_t i = 0; i < kN_; ++i)
        {
            I[i * (1 + kN_)] = 1;
        }
        return I;
    }

    template <typename T, size_t kP, size_t kQ>
        requires(kN_ == 1 || kM_ == 1) && (kP == 1 || kQ == 1) && (kN_ * kM_ == kP * kQ)
    MU_HOST_DEV_FUNC auto Dot(const Matrix<T, kP, kQ, true> &rhs) const -> ScalarType_
    {
        ScalarType_ res = 0;
        detail::MatrixElementwiseBinaryOperationUnroller<kUnrollSize_, kN_ * kM_>(
            *this, rhs,
            [&](ScalarType_ x, ScalarType_ y) -> void
            {
                res += x * y;
            });
        return res;
    }

    template <typename T, size_t kP, size_t kQ>
        requires(kN_ == 1 || kM_ == 1) && (kP == 1 || kQ == 1) && (kN_ * kM_ == kP * kQ)
    MU_HOST_DEV_FUNC auto OuterProduct(const Matrix<T, kP, kQ, true> &rhs) const
        -> Matrix<ScalarType_, kN_ * kM_, kP * kQ, true>
    {
        constexpr size_t kTotalSize = kN_ * kM_ * kP * kQ;
        Matrix<ScalarType_, kN_ * kM_, kP * kQ, true> res;

        if constexpr (kTotalSize <= kUnrollSize_)
        {
            meta::ConstexprLoop<0, kTotalSize>(
                [&](auto x) -> void
                {
                    constexpr static int index = meta::ConstexprLoopIndex(x);
                    constexpr static int lhsIndex = index / (kP * kQ);
                    constexpr static int rhsIndex = index % (kP * kQ);
                    res[index] = data_[lhsIndex] * rhs[rhsIndex];
                });
        }
        else
        {
            size_t i = 0;
            do
            {
                meta::ConstexprLoop<0, kUnrollSize_>(
                    [&](auto x) -> void
                    {
                        constexpr static int index = meta::ConstexprLoopIndex(x);
                        int index_ = index + i;
                        res[index_] = data_[index_ / (kP * kQ)] * rhs[index_ % (kP * kQ)];
                    });
                i += kUnrollSize_;
            } while (i < kTotalSize - kUnrollSize_);
            for (; i < kTotalSize; ++i)
            {
                res[i] = data_[i / (kP * kQ)] * rhs[i % (kP * kQ)];
            }
        }
        return res;
    }

    MU_FORCE_INLINE MU_HOST_DEV_FUNC auto Norm() const -> ScalarType_
    {
        ScalarType_ res = 0;
        detail::MatrixElementwiseUnaryOperationUnroller<kUnrollSize_, kN_ * kM_>(
            *this,
            [&](ScalarType_ x) -> void
            {
                res += x * x;
            });
        return sqrt(res);
    }

    MU_FORCE_INLINE MU_HOST_DEV_FUNC auto Trace() const -> ScalarType_
        requires(kN_ == kM_)
    {
        ScalarType_ res = 0;
        detail::MatrixElementwiseUnaryOperationUnroller<kUnrollSize_, kN_ * kM_>(
            *this,
            [&](ScalarType_ x) -> void
            {
                res += x;
            });
        return res;
    }

    MU_FORCE_INLINE MU_HOST_DEV_FUNC auto Inverse() const -> Matrix<ScalarType_, kM_, kN_, true>
        requires(kM_ == kN_)
    {
        auto res = Matrix<ScalarType_, kM_, kN_, true>{};

        if constexpr (kM_ == 2)
        {
            return (1.0 / Determinant()) * math::DenseMatrix<ScalarType_, kM_, kN_>{
                                               data_[3], -data_[1], -data_[2], data_[0]};
        }
        else if constexpr (kM_ == 3)
        {
            const auto inverseDeterminant = static_cast<ScalarType_>(1) / Determinant();

            res[0] = inverseDeterminant * (data_[4] * data_[8] - data_[5] * data_[7]);
            res[1] = inverseDeterminant * (data_[2] * data_[7] - data_[1] * data_[8]);
            res[2] = inverseDeterminant * (data_[1] * data_[5] - data_[2] * data_[4]);
            res[3] = inverseDeterminant * (data_[5] * data_[6] - data_[3] * data_[8]);
            res[4] = inverseDeterminant * (data_[0] * data_[8] - data_[2] * data_[6]);
            res[5] = inverseDeterminant * (data_[2] * data_[3] - data_[0] * data_[5]);
            res[6] = inverseDeterminant * (data_[3] * data_[7] - data_[4] * data_[6]);
            res[7] = inverseDeterminant * (data_[1] * data_[6] - data_[0] * data_[7]);
            res[8] = inverseDeterminant * (data_[0] * data_[4] - data_[1] * data_[3]);
        }
        else
        {
            assert(false);
        }

        return res;
    }

    MU_FORCE_INLINE MU_HOST_DEV_FUNC auto Transpose() const -> Matrix<ScalarType_, kM_, kN_, true>
    {
        auto res = Matrix<ScalarType_, kM_, kN_, true>{};
        if constexpr (kN_ * kM_ <= kUnrollSize_)
        {
            meta::ConstexprLoop<0, kN_ * kM_>(
                [&](auto x) -> void
                {
                    constexpr static int index = meta::ConstexprLoopIndex(x);
                    constexpr static int transposeIndex = (index % kM_) * kN_ + (index / kM_);
                    res[transposeIndex] = data_[index];
                });
        }
        else
        {
            size_t i = 0;
            do
            {
                meta::ConstexprLoop<0, kUnrollSize_>(
                    [&](auto x) -> void
                    {
                        constexpr static int index = meta::ConstexprLoopIndex(x);
                        int index_ = index + i;
                        res[(index_ % kM_) * kN_ + (index_ / kM_)] = data_[index_];
                    });
                i += kUnrollSize_;
            } while (i < kN_ * kM_ - kUnrollSize_);
            for (; i < kN_ * kM_; ++i)
            {
                res[(i % kM_) * kN_ + (i / kM_)] = data_[i];
            }
        }
        return res;
    }

    template <size_t kP, size_t kQ>
        requires(kN_ == 3 && kM_ == 1 && kP == 3 && kQ == 1)
    MU_FORCE_INLINE MU_HOST_DEV_FUNC auto CrossProduct(
        const DenseMatrix<ScalarType_, kP, kQ> &rhs) const -> DenseMatrix<ScalarType_, 3, 1>
    {
        return DenseMatrix<ScalarType_, 3, 1>{data_[1] * rhs[2] - data_[2] * rhs[1],
                                              data_[2] * rhs[0] - data_[0] * rhs[2],
                                              data_[0] * rhs[1] - data_[1] * rhs[0]};
    }

    MU_FORCE_INLINE MU_HOST_DEV_FUNC auto Normalize() const -> DenseMatrix<ScalarType_, kN_, kM_>
    {
        auto res = *this;
        res /= res.Norm();
        return res;
    }

    MU_FORCE_INLINE MU_HOST_DEV_FUNC auto Determinant() const -> ScalarType_
        requires((kN_ == kM_) && (kN_ == 3))
    {
        auto res = data_[0] * data_[4] * data_[8] + data_[1] * data_[5] * data_[6] +
                   data_[2] * data_[3] * data_[7];
        res -= data_[2] * data_[4] * data_[6] + data_[1] * data_[3] * data_[8] +
               data_[0] * data_[5] * data_[7];
        return res;
    }

    MU_FORCE_INLINE MU_HOST_DEV_FUNC auto Determinant() const -> ScalarType_
        requires((kN_ == kM_) && (kN_ == 2))
    {
        return data_[0] * data_[3] - data_[1] * data_[2];
    }

    template <MatrixMemoryAlignment MemoryAlignment>
    MU_FORCE_INLINE MU_HOST_DEV_FUNC auto Flatten() const -> math::DenseVector<ScalarType_, kN * kM>
    {

        math::DenseVector<ScalarType_, kN * kM> res = {};
        if constexpr (MemoryAlignment == MatrixMemoryAlignment::kRowMajor)
        {
            detail::MatrixElementwiseBinaryOperationUnroller<kUnrollSize_, kN_ * kM_>(
                res, *this,
                [&](ScalarType_ &x, ScalarType_ y) -> void
                {
                    x = y;
                });
        }
        else
        {
            detail::MatrixElementwiseBinaryOperationUnroller<kUnrollSize_, kN_ * kM_>(
                res, Transpose(),
                [&](ScalarType_ &x, ScalarType_ y) -> void
                {
                    x = y;
                });
        }
        return res;
    }

    MU_HOST_DEV_FUNC auto Svd(DenseMatrix<ScalarType_, 3, 3> &U,
                              DenseMatrix<ScalarType_, 3, 3> &Sigma,
                              DenseMatrix<ScalarType_, 3, 3> &V) const -> void
    {
        DenseMatrix B = *this;

        Sigma = 0;
        U = Identity();
        V = Identity();

        SvdUpperBidiagonalize(U, B, V);
        ScalarType_ alpha0 = B[0], alpha1 = B[4], alpha2 = B[8];
        ScalarType_ beta0 = B[1], beta1 = B[5];
        ScalarType_ gamma0 = alpha0 * beta0, gamma1 = alpha1 * beta1;

        constexpr ScalarType_ eta = 128 * std::numeric_limits<ScalarType_>::epsilon();
        ScalarType_ tau =
            eta * max(static_cast<ScalarType_>(0.5) * B.Norm(), static_cast<ScalarType_>(1));

        DenseMatrix<ScalarType_, 3, 3> G;
        while (std::abs(beta0) > tau && std::abs(beta1) > tau && std::abs(alpha0) > tau &&
               std::abs(alpha1) > tau && std::abs(alpha2) > tau)
        {
            auto a0 = alpha1 * alpha1 + beta0 * beta0;
            auto a1 = alpha2 * alpha2 + beta1 * beta1;
            auto b0 = gamma1;
            auto d = (a0 - a1) / 2;

            ScalarType_ mu = std::copysign(b0 * b0 / (std::abs(d) + std::sqrt(d * d + b0 * b0)), d);
            G = SvdGivensRotation<3, 0, 1, true>(alpha0 * alpha0 - mu, gamma0);
            auto C = U.MatrixMultiplication(B).MatrixMultiplication(V.Transpose());

            B = B.MatrixMultiplication(G);
            V = V.MatrixMultiplication(G);

            C = U.MatrixMultiplication(B).MatrixMultiplication(V.Transpose());

            SvdZerochasing(U, B, V);

            alpha0 = B[0];
            alpha1 = B[4];
            alpha2 = B[8];
            beta0 = B[1];
            beta1 = B[5];
            gamma0 = alpha0 * beta0;
            gamma1 = alpha1 * beta1;
        }

        DenseVector<ScalarType_, 3> alpha = {alpha0, alpha1, alpha2};
        DenseVector<ScalarType_, 2> beta = {beta0, beta1};
        SvdPostProcess(tau, B, U, Sigma, V, alpha, beta);
    }

private:
    template <bool kIsGivensConventional, typename Scalar>
    MU_FORCE_INLINE MU_HOST_DEV_FUNC auto SvdGivensRotationScalars(Scalar x, Scalar y) const
        -> DenseVector<Scalar, 2>
    {
        auto d = x * x + y * y;
        Scalar c = kIsGivensConventional ? 1 : 0;
        Scalar s = kIsGivensConventional ? 0 : 1;
        if (d != 0)
        {
            auto t = rsqrt(d);
            c = kIsGivensConventional ? x * t : y * t;
            s = kIsGivensConventional ? -y * t : x * t;
        }

        return {c, s};
    }

    template <size_t kNDim, size_t i, size_t j, bool kIsGivensConventional, typename Scalar>
    MU_FORCE_INLINE MU_HOST_DEV_FUNC auto SvdGivensRotation(Scalar x, Scalar y) const
    {
        if constexpr (kNDim == 2)
        {
            auto cs = SvdGivensRotationScalars<kIsGivensConventional>(x, y);
            return DenseMatrix<Scalar, 2, 2>{cs[0], cs[1], -cs[1], cs[0]};
        }
        else if (kNDim == 3)
        {
            auto cs = SvdGivensRotationScalars<kIsGivensConventional>(x, y);
            if constexpr (i == 0 && j == 1)
            {
                return DenseMatrix<Scalar, 3, 3>{cs[0], cs[1], 0, -cs[1], cs[0], 0, 0, 0, 1};
            }
            else if constexpr (i == 1 && j == 2)
            {
                return DenseMatrix<Scalar, 3, 3>{1, 0, 0, 0, cs[0], cs[1], 0, -cs[1], cs[0]};
            }
            else if constexpr (i == 0 && j == 2)
            {
                return DenseMatrix<Scalar, 3, 3>{cs[0], 0, cs[1], 0, 1, 0, -cs[1], 0, cs[0]};
            }

            assert(false);
            return DenseMatrix<Scalar, 3, 3>{};
        }
        assert(false);
        return DenseMatrix<Scalar, kNDim, kNDim>{};
    }

    template <typename Scalar>
    MU_HOST_DEV_FUNC auto SvdZerochasing(DenseMatrix<Scalar, 3, 3> &U, DenseMatrix<Scalar, 3, 3> &A,
                                         DenseMatrix<Scalar, 3, 3> &V) const -> void
    {
        auto G = SvdGivensRotation<3, 0, 1, true>(A[0], A[3]);
        A = G.Transpose().MatrixMultiplication(A);
        U = U.MatrixMultiplication(G);

        G = SvdGivensRotation<3, 1, 2, true>(A[1], A[2]);
        A = A.MatrixMultiplication(G);
        V = V.MatrixMultiplication(G);

        G = SvdGivensRotation<3, 1, 2, true>(A[4], A[7]);
        A = G.Transpose().MatrixMultiplication(A);
        U = U.MatrixMultiplication(G);
    }

    template <typename Scalar>
    MU_HOST_DEV_FUNC auto SvdUpperBidiagonalize(DenseMatrix<Scalar, 3, 3> &U,
                                                DenseMatrix<Scalar, 3, 3> &A,
                                                DenseMatrix<Scalar, 3, 3> &V) const -> void
    {
        auto G = SvdGivensRotation<3, 1, 2, true>(A[3], A[6]);
        A = G.Transpose().MatrixMultiplication(A);
        U = U.MatrixMultiplication(G);
        SvdZerochasing(U, A, V);
    }

    template <typename Scalar>
    MU_FORCE_INLINE MU_HOST_DEV_FUNC auto SvdPolarDecomposition2D(
        const DenseMatrix<Scalar, 2, 2> &A, DenseMatrix<Scalar, 2, 2> &R,
        DenseMatrix<Scalar, 2, 2> &S) const
    {
        auto x = A[0] + A[3];
        auto y = A[2] - A[1];

        auto d = std::sqrt(x * x + y * y);

        R = d != 0 ? DenseMatrix<Scalar, 2, 2>{x / d, -y / d, y / d, x / d}
                   : DenseMatrix<Scalar, 2, 2>{1, 0, 0, 1};
        S = R.Transpose().MatrixMultiplication(A);
    }

    template <typename Scalar>
    MU_HOST_DEV_FUNC auto Svd2D(const DenseMatrix<Scalar, 2, 2> &A, DenseMatrix<Scalar, 2, 2> &U,
                                DenseMatrix<Scalar, 2, 2> &V, Scalar &sigma1, Scalar &sigma2) const
    {
        DenseMatrix<Scalar, 2, 2> S;
        SvdPolarDecomposition2D(A, U, S);

        Scalar cs[2] = {};
        Scalar sigma[2] = {};
        if (S[1] == 0)
        {
            cs[0] = 1;
            cs[1] = 0;
            sigma[0] = S[0];
            sigma[1] = S[3];
        }
        else
        {
            auto tau = static_cast<Scalar>(0.5) * (S[0] - S[3]);
            auto w = std::sqrt(tau * tau + S[1] * S[1]);

            auto t = (tau > 0) ? S[1] / (tau + w) : S[1] / (tau - w);

            cs[0] = static_cast<Scalar>(1) / std::sqrt(t * t + static_cast<Scalar>(1));
            cs[1] = -t * cs[0];

            sigma[0] = cs[0] * cs[0] * S[0] - 2 * cs[0] * cs[1] * S[1] + cs[1] * cs[1] * S[3];
            sigma[1] = cs[1] * cs[1] * S[0] + 2 * cs[0] * cs[1] * S[1] + cs[0] * cs[0] * S[3];
        }

        if (sigma[0] < sigma[1])
        {
            std::swap(sigma[0], sigma[1]);
            V = DenseMatrix<Scalar, 2, 2>{-cs[1], cs[0], -cs[0], -cs[1]};
        }
        else
        {
            V = DenseMatrix<Scalar, 2, 2>{cs[0], cs[1], -cs[1], cs[0]};
        }
        U = U.MatrixMultiplication(V);
        sigma1 = sigma[0];
        sigma2 = sigma[1];
    }

    template <bool kIsTopLeft, typename Scalar>
    MU_HOST_DEV_FUNC auto SvdSolveReducedBlock(DenseMatrix<Scalar, 3, 3> &B,
                                               DenseMatrix<Scalar, 3, 3> &U,
                                               DenseMatrix<Scalar, 3, 3> &Sigma,
                                               DenseMatrix<Scalar, 3, 3> &V) const -> void
    {
        if constexpr (kIsTopLeft)
        {
            Sigma[8] = B[8];
            auto uTemp =
                SvdGivensRotation<2, 0, 1, true>(static_cast<Scalar>(0), static_cast<Scalar>(0));
            auto vTemp =
                SvdGivensRotation<2, 0, 1, true>(static_cast<Scalar>(0), static_cast<Scalar>(0));
            DenseMatrix<Scalar, 2, 2> subB = {B[0], B[1], B[3], B[4]};
            Svd2D(subB, uTemp, vTemp, Sigma[0], Sigma[4]);
            auto u = DenseMatrix<Scalar, 3, 3>{};
            auto v = DenseMatrix<Scalar, 3, 3>{};
            u[0] = uTemp[0];
            u[1] = uTemp[1];
            u[3] = uTemp[2];
            u[4] = uTemp[3];
            u[8] = 1;

            v[0] = vTemp[0];
            v[1] = vTemp[1];
            v[3] = vTemp[2];
            v[4] = vTemp[3];
            v[8] = 1;

            U = U.MatrixMultiplication(u);
            V = V.MatrixMultiplication(v);
        }
        else
        {
            Sigma[0] = B[0];
            auto uTemp =
                SvdGivensRotation<2, 0, 1, true>(static_cast<Scalar>(0), static_cast<Scalar>(0));
            auto vTemp =
                SvdGivensRotation<2, 0, 1, true>(static_cast<Scalar>(0), static_cast<Scalar>(0));

            DenseMatrix<Scalar, 2, 2> subB = {B[4], B[5], B[7], B[8]};
            Svd2D(subB, uTemp, vTemp, Sigma[4], Sigma[8]);

            auto u = DenseMatrix<Scalar, 3, 3>{};
            auto v = DenseMatrix<Scalar, 3, 3>{};
            u[0] = 1;
            u[4] = uTemp[0];
            u[5] = uTemp[1];
            u[7] = uTemp[2];
            u[8] = uTemp[3];

            v[0] = 1;
            v[4] = vTemp[0];
            v[5] = vTemp[1];
            v[7] = vTemp[2];
            v[8] = vTemp[3];
            U = U.MatrixMultiplication(u);
            V = V.MatrixMultiplication(v);
        }
    }

    template <int kColumnIndex, bool kIsFlipSigmaSign = true, typename Scalar>
    MU_FORCE_INLINE MU_HOST_DEV_FUNC auto SvdFlipSign(DenseMatrix<Scalar, 3, 3> &U,
                                                      DenseMatrix<Scalar, 3, 3> &Sigma) const
    {
        if constexpr (kIsFlipSigmaSign)
        {
            Sigma[kColumnIndex * 3 + kColumnIndex] = -Sigma[kColumnIndex * 3 + kColumnIndex];
        }

        U[kColumnIndex] = -U[kColumnIndex];
        U[3 + kColumnIndex] = -U[3 + kColumnIndex];
        U[6 + kColumnIndex] = -U[6 + kColumnIndex];
    }

    template <int kColumnIndex0, int kColumnIndex1, typename Scalar>
        requires(kColumnIndex0 != kColumnIndex1) && (0 <= kColumnIndex0 && kColumnIndex0 < 3) &&
                (0 <= kColumnIndex1 && kColumnIndex1 < 3)
    MU_FORCE_INLINE MU_HOST_DEV_FUNC auto SvdSwapColumn(DenseMatrix<Scalar, 3, 3> &U) const
    {
        std::swap(U[kColumnIndex0], U[kColumnIndex1]);
        std::swap(U[3 + kColumnIndex0], U[3 + kColumnIndex1]);
        std::swap(U[6 + kColumnIndex0], U[6 + kColumnIndex1]);
    }

    template <bool kIsTopLeft, typename Scalar>
    MU_FORCE_INLINE MU_HOST_DEV_FUNC auto SvdSort(DenseMatrix<Scalar, 3, 3> &U,
                                                  DenseMatrix<Scalar, 3, 3> &Sigma,
                                                  DenseMatrix<Scalar, 3, 3> &V) const
    {
        if constexpr (kIsTopLeft)
        {
            if (std::abs(Sigma[4]) >= std::abs(Sigma[8]))
            {
                if (Sigma[4] < 0)
                {
                    SvdFlipSign<1>(U, Sigma);
                    SvdFlipSign<2>(U, Sigma);
                }
                return;
            }

            if (Sigma[8] < 0)
            {
                SvdFlipSign<1>(U, Sigma);
                SvdFlipSign<2>(U, Sigma);
            }

            std::swap(Sigma[4], Sigma[8]);
            SvdSwapColumn<1, 2>(U);
            SvdSwapColumn<1, 2>(V);

            if (Sigma[4] > Sigma[0])
            {
                std::swap(Sigma[0], Sigma[4]);
                SvdSwapColumn<0, 1>(U);
                SvdSwapColumn<0, 1>(V);
            }
            else
            {
                SvdFlipSign<2, false>(U, Sigma);
                SvdFlipSign<2, false>(V, Sigma);
            }
        }
        else
        {
            if (std::abs(Sigma[0]) >= std::abs(Sigma[4]))
            {
                if (Sigma[0] < 0)
                {
                    SvdFlipSign<0>(U, Sigma);
                    SvdFlipSign<2>(U, Sigma);
                }
                return;
            }
            std::swap(Sigma[0], Sigma[4]);
            SvdSwapColumn<0, 1>(U);
            SvdSwapColumn<0, 1>(V);

            if (std::abs(Sigma[4]) < std::abs(Sigma[8]))
            {
                std::swap(Sigma[4], Sigma[8]);
                SvdSwapColumn<1, 2>(U);
                SvdSwapColumn<1, 2>(V);
            }
            else
            {
                SvdFlipSign<1, false>(U, Sigma);
                SvdFlipSign<1, false>(V, Sigma);
            }

            if (Sigma[4] < 0)
            {
                SvdFlipSign<1>(U, Sigma);
                SvdFlipSign<2>(U, Sigma);
            }
        }
    }

    template <typename Scalar>
    MU_HOST_DEV_FUNC auto SvdPostProcess(Scalar tau, DenseMatrix<Scalar, 3, 3> &B,
                                         DenseMatrix<Scalar, 3, 3> &U,
                                         DenseMatrix<Scalar, 3, 3> &Sigma,
                                         DenseMatrix<Scalar, 3, 3> &V,
                                         DenseVector<Scalar, 3> &alpha,
                                         DenseVector<Scalar, 2> &beta) const -> void
    {
        if (std::abs(beta[1]) <= tau)
        {
            SvdSolveReducedBlock<true>(B, U, Sigma, V);
            SvdSort<true>(U, Sigma, V);
        }
        else if (std::abs(beta[0]) <= tau)
        {
            SvdSolveReducedBlock<false>(B, U, Sigma, V);
            SvdSort<false>(U, Sigma, V);
        }
        else if (std::abs(alpha[1]) <= tau)
        {
            auto G = SvdGivensRotation<3, 1, 2, false>(B[5], B[8]);
            B = G.Transpose().MatrixMultiplication(B);
            U = U.MatrixMultiplication(G);
            SvdSolveReducedBlock<true>(B, U, Sigma, V);
            SvdSort<true>(U, Sigma, V);
        }
        else if (std::abs(alpha[2]) <= tau)
        {
            auto G = SvdGivensRotation<3, 1, 2, true>(B[4], B[5]);
            B = B.MatrixMultiplication(G);
            V = V.MatrixMultiplication(G);
            G = SvdGivensRotation<3, 0, 2, true>(B[0], B[2]);
            B = B.MatrixMultiplication(G);
            V = V.MatrixMultiplication(G);
            SvdSolveReducedBlock<true>(B, U, Sigma, V);
            SvdSort<true>(U, Sigma, V);
        }
        else if (std::abs(alpha[0]) <= tau)
        {
            auto G = SvdGivensRotation<3, 0, 1, false>(B[1], B[4]);
            B = G.Transpose().MatrixMultiplication(B);
            U = U.MatrixMultiplication(G);
            G = SvdGivensRotation<3, 0, 2, false>(B[2], B[8]);
            B = G.Transpose().MatrixMultiplication(B);
            U = U.MatrixMultiplication(G);

            SvdSolveReducedBlock<false>(B, U, Sigma, V);
            SvdSort<false>(U, Sigma, V);
        }
    }

private:
    ScalarType_ data_[kN * kM] = {};
};


template <typename T, size_t kN, size_t kM, typename U>
    requires std::is_scalar_v<U> && std::is_convertible_v<U, T>
MU_HOST_DEV_FUNC auto operator+(const Matrix<T, kN, kM, true> &lhs,
                                const U rhs) -> Matrix<T, kN, kM, true>
{
    Matrix<T, kN, kM, true> res(lhs);
    res += static_cast<T>(rhs);
    return res;
}

template <typename T, size_t kN, size_t kM, typename U>
    requires std::is_scalar_v<U> && std::is_convertible_v<U, T>
MU_HOST_DEV_FUNC auto operator+(const U lhs,
                                const Matrix<T, kN, kM, true> &rhs) -> Matrix<T, kN, kM, true>
{
    Matrix<T, kN, kM, true> res(rhs);
    res += static_cast<T>(lhs);
    return res;
}

template <typename T, size_t kN, size_t kM, typename U>
    requires std::is_scalar_v<U> && std::is_convertible_v<U, T>
MU_HOST_DEV_FUNC auto operator-(const Matrix<T, kN, kM, true> &lhs,
                                const U rhs) -> Matrix<T, kN, kM, true>
{
    Matrix<T, kN, kM, true> res(lhs);
    res -= rhs;
    return res;
}

template <typename T, size_t kN, size_t kM, typename U>
    requires std::is_scalar_v<U> && std::is_convertible_v<U, T>
MU_HOST_DEV_FUNC auto operator-(const U lhs,
                                const Matrix<T, kN, kM, true> &rhs) -> Matrix<T, kN, kM, true>
{
    Matrix<T, kN, kM, true> res{};
    detail::MatrixElementwiseBinaryOperationUnroller<DenseMatrix<T, kN, kM>::kUnrollSize_,
                                                     DenseMatrix<T, kN, kM>::kExtent_>(
        res, rhs,
        [&](T &x, U y) -> void
        {
            x = lhs - y;
        });
    return res;
}

template <typename T, size_t kN, size_t kM, typename U>
    requires std::is_scalar_v<U> && std::is_convertible_v<U, T>
MU_HOST_DEV_FUNC auto operator*(const Matrix<T, kN, kM, true> &lhs,
                                const U rhs) -> Matrix<T, kN, kM, true>
{
    Matrix<T, kN, kM, true> res(lhs);
    res *= rhs;
    return res;
}

template <typename T, size_t kN, size_t kM, typename U>
    requires std::is_scalar_v<U> && std::is_convertible_v<U, T>
MU_HOST_DEV_FUNC auto operator*(const U lhs,
                                const Matrix<T, kN, kM, true> &rhs) -> Matrix<T, kN, kM, true>
{
    Matrix<T, kN, kM, true> res(rhs);
    res *= lhs;
    return res;
}

template <typename T, size_t kN, size_t kM, typename U>
    requires std::is_scalar_v<U> && std::is_convertible_v<U, T>
MU_HOST_DEV_FUNC auto operator/(const Matrix<T, kN, kM, true> &lhs,
                                const U rhs) -> Matrix<T, kN, kM, true>
{
    Matrix<T, kN, kM, true> res(lhs);
    res /= rhs;
    return res;
}

template <typename T, size_t kN, size_t kM, typename U>
    requires std::is_scalar_v<U> && std::is_convertible_v<U, T>
MU_HOST_DEV_FUNC auto operator/(const U lhs,
                                const Matrix<T, kN, kM, true> &rhs) -> Matrix<T, kN, kM, true>
{
    Matrix<T, kN, kM, true> res{};

    detail::MatrixElementwiseBinaryOperationUnroller<DenseMatrix<T, kN, kM>::kUnrollSize_,
                                                     DenseMatrix<T, kN, kM>::kExtent_>(
        res, rhs,
        [&](T &x, U y) -> void
        {
            x = lhs / y;
        });

    return res;
}

};
