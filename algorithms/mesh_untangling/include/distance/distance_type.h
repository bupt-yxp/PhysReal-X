#pragma once

#include "distance_common.cuh"

namespace mu::distance
{
template <class T>
MU_FORCE_INLINE MU_HOST_DEV_FUNC void solveLdlt2D(
    const T A00, const T A01, const T A11, const T b0, const T b1, T& x0, T& x1)
{
    T D00, D11, L01;
    D00 = A00;
    L01 = A01 / D00;
    D11 = A11 - L01 * L01 * D00;

    T y0, y1;
    y0 = b0;
    y1 = b1 - L01 * y0;

    x1 = y1 / D11;
    x0 = (y0 - D00 * L01 * x1) / D00;
}

template <class T>
MU_FORCE_INLINE MU_HOST_DEV_FUNC int point_edge_distance_type(const T* p, const T* e0, const T* e1, const int dim)
{
    T e0e1[3], e0p[3], ratio;
    for (int i = 0; i < dim; i++) {
        e0e1[i] = e1[i] - e0[i];
        e0p[i] = p[i] - e0[i];
    }

    if (dim == 2) {
        ratio = math::dot<2>(e0e1, e0p) / math::norm_sqr<2>(e0e1);
    } else {
        ratio = math::dot<3>(e0e1, e0p) / math::norm_sqr<3>(e0e1);
    }
    if (ratio < 0) {
        return 0;
    } else if (ratio > 1) {
        return 1;
    } else {
        return 2;
    }
}

template <class T>
MU_FORCE_INLINE MU_HOST_DEV_FUNC int point_triangle_distance_type(
    const T* p, const T* t0, const T* t1, const T* t2)
{
    T basis_row0[3], basis_row1[3], nVec[3];
    basis_row0[0] = t1[0] - t0[0];
    basis_row0[1] = t1[1] - t0[1];
    basis_row0[2] = t1[2] - t0[2];
    basis_row1[0] = t2[0] - t0[0];
    basis_row1[1] = t2[1] - t0[1];
    basis_row1[2] = t2[2] - t0[2];
    math::cross_product3(basis_row0, basis_row1, nVec);
    math::cross_product3(basis_row0, nVec, basis_row1);

    T param_col0[2], param_col1[2], param_col2[2];
    T sys[4], rhs[2], b[3];
    sys[0] = basis_row0[0] * basis_row0[0] + basis_row0[1] * basis_row0[1]
        + basis_row0[2] * basis_row0[2];
    sys[1] = sys[2] = basis_row0[0] * basis_row1[0] + basis_row0[1] * basis_row1[1]
        + basis_row0[2] * basis_row1[2];
    sys[3] = basis_row1[0] * basis_row1[0] + basis_row1[1] * basis_row1[1]
        + basis_row1[2] * basis_row1[2];

    b[0] = p[0] - t0[0];
    b[1] = p[1] - t0[1];
    b[2] = p[2] - t0[2];

    rhs[0] = math::dot<3>(basis_row0, b);
    rhs[1] = math::dot<3>(basis_row1, b);
    solveLdlt2D(sys[0], sys[1], sys[3], rhs[0], rhs[1], param_col0[0], param_col0[1]);
    if (param_col0[0] > 0.0 && param_col0[0] < 1.0 && param_col0[1] >= 0.0) {
        return 3;
    } else {
        basis_row0[0] = t2[0] - t1[0];
        basis_row0[1] = t2[1] - t1[1];
        basis_row0[2] = t2[2] - t1[2];
        math::cross_product3(basis_row0, nVec, basis_row1);

        sys[0] = basis_row0[0] * basis_row0[0] + basis_row0[1] * basis_row0[1]
            + basis_row0[2] * basis_row0[2];
        sys[1] = sys[2] = basis_row0[0] * basis_row1[0] + basis_row0[1] * basis_row1[1]
            + basis_row0[2] * basis_row1[2];
        sys[3] = basis_row1[0] * basis_row1[0] + basis_row1[1] * basis_row1[1]
            + basis_row1[2] * basis_row1[2];
        b[0] = p[0] - t1[0];
        b[1] = p[1] - t1[1];
        b[2] = p[2] - t1[2];
        rhs[0] = math::dot<3>(basis_row0, b);
        rhs[1] = math::dot<3>(basis_row1, b);
        solveLdlt2D(sys[0], sys[1], sys[3], rhs[0], rhs[1], param_col1[0], param_col1[1]);
        if (param_col1[0] > 0.0 && param_col1[0] < 1.0 && param_col1[1] >= 0.0) {
            return 4;
        } else {
            basis_row0[0] = t0[0] - t2[0];
            basis_row0[1] = t0[1] - t2[1];
            basis_row0[2] = t0[2] - t2[2];
            math::cross_product3(basis_row0, nVec, basis_row1);
            sys[0] = basis_row0[0] * basis_row0[0] + basis_row0[1] * basis_row0[1]
                + basis_row0[2] * basis_row0[2];
            sys[1] = sys[2] = basis_row0[0] * basis_row1[0] + basis_row0[1] * basis_row1[1]
                + basis_row0[2] * basis_row1[2];
            sys[3] = basis_row1[0] * basis_row1[0] + basis_row1[1] * basis_row1[1]
                + basis_row1[2] * basis_row1[2];
            b[0] = p[0] - t2[0];
            b[1] = p[1] - t2[1];
            b[2] = p[2] - t2[2];
            rhs[0] = math::dot<3>(basis_row0, b);
            rhs[1] = math::dot<3>(basis_row1, b);
            solveLdlt2D(sys[0], sys[1], sys[3], rhs[0], rhs[1], param_col2[0], param_col2[1]);

            if (param_col2[0] > 0.0 && param_col2[0] < 1.0 && param_col2[1] >= 0.0) {
                return 5;
            } else {
                if (param_col0[0] <= 0.0 && param_col2[0] >= 1.0) {
                    return 0;
                } else if (param_col1[0] <= 0.0 && param_col0[0] >= 1.0) {
                    return 1;
                } else if (param_col2[0] <= 0.0 && param_col1[0] >= 1.0) {
                    return 2;
                } else {
                    return 6;
                }
            }
        }
    }
}

template <class T>
MU_FORCE_INLINE MU_HOST_DEV_FUNC int edge_edge_distance_type(
    const T* ea0, const T* ea1, const T* eb0, const T* eb1)
{
    T u[3], v[3], w[3];
    for (int i = 0; i < 3; i++) {
        u[i] = ea1[i] - ea0[i];
        v[i] = eb1[i] - eb0[i];
        w[i] = ea0[i] - eb0[i];
    }

    T a = math::norm_sqr<3>(u);
    T b = math::dot<3>(u, v);
    T c = math::norm_sqr<3>(v);
    T d = math::dot<3>(u, w);
    T e = math::dot<3>(v, w);
    T D = a * c - b * b;
    T tD = D;
    T sN, tN;

    int defaultCase = 8;

    sN = (b * e - c * d);
    if (sN <= 0.0) {
        tN = e;
        tD = c;
        defaultCase = 2;
    } else if (sN >= D) {
        tN = e + b;
        tD = c;
        defaultCase = 5;
    } else {
        tN = (a * e - b * d);
        T uv[3];
        math::cross_product3(u, v, uv);

        if (tN > 0.0 && tN < tD
            && (math::dot<3>(uv, w) == 0.0 || math::norm_sqr<3>(uv) < 1.0e-20 * a * c)) {
            if (sN < D / 2) {
                tN = e;
                tD = c;
                defaultCase = 2;
            } else {
                tN = e + b;
                tD = c;
                defaultCase = 5;
            }
        }
    }

    if (tN <= 0.0) {
        if (-d <= 0.0) {
            return 0;
        } else if (-d >= a) {
            return 3;
        } else {
            return 6;
        }
    } else if (tN >= tD) {
        if ((-d + b) <= 0.0) {
            return 1;
        } else if ((-d + b) >= a) {
            return 4;
        } else {
            return 7;
        }
    }

    return defaultCase;
}

}
