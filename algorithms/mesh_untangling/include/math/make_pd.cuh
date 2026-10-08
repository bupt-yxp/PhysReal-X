#pragma once


#include <cmath>

#include "core/base.cuh"

namespace mu::math
{

namespace detail
{

template <typename Scalar, int N>
MU_DEV_FUNC MU_FORCE_INLINE auto MaxOffDiagonalIndex(int k, const Scalar* S) -> int
{
    int m = k + 1;
    for (int i = k + 2; i < N; ++i)
    {
        if (::fabs(S[k * N + i]) > ::fabs(S[k * N + m]))
        {
            m = i;
        }
    }
    return m;
}

template <typename Scalar>
MU_DEV_FUNC MU_FORCE_INLINE auto UpdateEigenvalue(int k, Scalar t, Scalar& y, int& state, Scalar* e,
                                                  int* changed) -> void
{
    y = e[k];
    e[k] = y + t;
    if (changed[k] && y == e[k])
    {
        changed[k] = 0;
        --state;
    }
    else if (!changed[k] && y != e[k])
    {
        changed[k] = 1;
        ++state;
    }
}

template <typename Scalar, int N>
MU_DEV_FUNC MU_FORCE_INLINE auto ApplyGivensRotation(int k, int l, int i, int j, Scalar s, Scalar c,
                                                     Scalar* S) -> void
{
    const Scalar Skl = S[k * N + l];
    const Scalar Sij = S[i * N + j];
    S[k * N + l] = c * Skl - s * Sij;
    S[i * N + j] = s * Skl + c * Sij;
}

}

template <typename Scalar, int N>
MU_DEV_FUNC auto MakePSDJacobi(Scalar* S, Scalar* E, Scalar tol = Scalar(1e-9)) -> Scalar
{
    Scalar maxS = Scalar(0);
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            const Scalar absVal = ::fabs(S[i * N + j]);
            if (absVal > maxS)
            {
                maxS = absVal;
            }
        }
    }

    if (maxS < Scalar(1e-12))
    {
        return Scalar(0);
    }

    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            E[i * N + j] = (i == j) ? Scalar(1) : Scalar(0);
        }
    }

    Scalar e[N];
    int ind[N];
    int changed[N];

    int state = N;
    for (int k = 0; k < N; ++k)
    {
        ind[k] = detail::MaxOffDiagonalIndex<Scalar, N>(k, S);
        e[k] = S[k * N + k];
        changed[k] = 1;
    }

    constexpr int kMaxIterations = 100;
    for (int iter = 0; iter < kMaxIterations && state != 0; ++iter)
    {
        int m = 0;
        for (int k = 1; k < N - 1; ++k)
        {
            if (::fabs(S[k * N + ind[k]]) > ::fabs(S[m * N + ind[m]]))
            {
                m = k;
            }
        }

        const int k = m;
        const int l = ind[m];
        const Scalar p = S[k * N + l];

        if (::fabs(p) < maxS * tol)
        {
            break;
        }

        Scalar y = (e[l] - e[k]) / Scalar(2);
        const Scalar d = ::fabs(y) + ::sqrt(p * p + y * y);
        const Scalar r = ::sqrt(p * p + d * d);
        const Scalar c = d / r;
        Scalar s = p / r;
        Scalar t = p * p / d;

        if (y < Scalar(0))
        {
            s = -s;
            t = -t;
        }

        S[k * N + l] = Scalar(0);

        detail::UpdateEigenvalue(k, -t, y, state, e, changed);
        detail::UpdateEigenvalue(l, t, y, state, e, changed);

        for (int i = 0; i <= k - 1; ++i)
        {
            detail::ApplyGivensRotation<Scalar, N>(i, k, i, l, s, c, S);
        }
        for (int i = k + 1; i <= l - 1; ++i)
        {
            detail::ApplyGivensRotation<Scalar, N>(k, i, i, l, s, c, S);
        }
        for (int i = l + 1; i < N; ++i)
        {
            detail::ApplyGivensRotation<Scalar, N>(k, i, l, i, s, c, S);
        }

        for (int i = 0; i < N; ++i)
        {
            const Scalar Eik = E[i * N + k];
            const Scalar Eil = E[i * N + l];
            E[i * N + k] = c * Eik - s * Eil;
            E[i * N + l] = s * Eik + c * Eil;
        }

        for (int i = 0; i < N; ++i)
        {
            ind[i] = detail::MaxOffDiagonalIndex<Scalar, N>(i, S);
        }
    }

    for (int i = 0; i < N; ++i)
    {
        e[i] = (e[i] > Scalar(0)) ? e[i] : Scalar(0);
    }

    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            S[i * N + j] = Scalar(0);
        }
    }

    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            const Scalar Eij_ej = E[i * N + j] * e[j];
            for (int k = 0; k < N; ++k)
            {
                S[i * N + k] += Eij_ej * E[k * N + j];
            }
        }
    }

    Scalar maxE = e[0];
    for (int i = 1; i < N; ++i)
    {
        if (e[i] > maxE)
        {
            maxE = e[i];
        }
    }
    return maxE;
}

template <typename Scalar>
MU_DEV_FUNC MU_FORCE_INLINE auto MakePSD6x6(Scalar* matrix) -> Scalar
{
    Scalar E[36];
    return MakePSDJacobi<Scalar, 6>(matrix, E);
}

template <typename Scalar>
MU_DEV_FUNC MU_FORCE_INLINE auto MakePSD9x9(Scalar* matrix) -> Scalar
{
    Scalar E[81];
    return MakePSDJacobi<Scalar, 9>(matrix, E);
}

template <typename Scalar>
MU_DEV_FUNC MU_FORCE_INLINE auto MakePSD12x12(Scalar* matrix) -> Scalar
{
    Scalar E[144];
    return MakePSDJacobi<Scalar, 12>(matrix, E);
}

}
