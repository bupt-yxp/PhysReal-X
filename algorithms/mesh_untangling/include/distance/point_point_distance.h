#pragma once

#include "distance_common.cuh"

namespace mu::distance
{
template <class T>
MU_FORCE_INLINE MU_HOST_DEV_FUNC T point_point_distance(const T* a, const T* b, const int dim = 3)
{
    T dist = 0;
    for (int i = 0; i < dim; i++)
        dist += (a[i] - b[i]) * (a[i] - b[i]);
    return dist;
}

template <class T>
MU_FORCE_INLINE MU_HOST_DEV_FUNC void point_point_distance_gradient(
    const T* a, const T* b, T* grad, const int dim = 3)
{
    for (int i = 0; i < dim; i++) {
        grad[i] = 2 * (a[i] - b[i]);
        grad[i + dim] = -grad[i];
    }
}

template <class T>
MU_FORCE_INLINE MU_HOST_DEV_FUNC void point_point_distance_hessian(
    const T* a, const T* b, T* hess, const int dim = 3)
{
    int size = 2 * dim;
    size *= size;
    for (int i = 0; i < size; i++)
        hess[i] = 0.;

    if (dim == 2) {
        hess[0] = hess[5] = hess[10] = hess[15] = 2.;
        hess[2] = hess[7] = hess[8] = hess[13] = -2.;
    } else {
        hess[0] = hess[7] = hess[14] = hess[21] = hess[28] = hess[35] = 2.;
        hess[3] = hess[10] = hess[17] = hess[18] = hess[25] = hess[32] = -2.;
    }
}

}
