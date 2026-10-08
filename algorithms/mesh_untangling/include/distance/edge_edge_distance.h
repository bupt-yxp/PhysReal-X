#pragma once

#include "distance_type.h"
#include "line_line_distance.h"
#include "point_line_distance.h"
#include "point_point_distance.h"
#include "distance_common.cuh"

namespace mu::distance
{
template <class T>
MU_FORCE_INLINE MU_HOST_DEV_FUNC T edge_edge_distance(const T* ea0, const T* ea1, const T* eb0, const T* eb1)
{
    switch (edge_edge_distance_type(ea0, ea1, eb0, eb1)) {
        case 0:
            return point_point_distance(ea0, eb0);
        case 1:
            return point_point_distance(ea0, eb1);
        case 2:
            return point_line_distance(ea0, eb0, eb1);
        case 3:
            return point_point_distance(ea1, eb0);
        case 4:
            return point_point_distance(ea1, eb1);
        case 5:
            return point_line_distance(ea1, eb0, eb1);
        case 6:
            return point_line_distance(eb0, ea0, ea1);
        case 7:
            return point_line_distance(eb1, ea0, ea1);
        case 8:
            return line_line_distance(ea0, ea1, eb0, eb1);
        default:
            return 1e20;
    }
}

template <class T>
MU_FORCE_INLINE MU_HOST_DEV_FUNC void edge_edge_distance_gradient(
    const T* ea0, const T* ea1, const T* eb0, const T* eb1, T* grad)
{
    for (int i = 0; i < 12; ++i)
        grad[i] = 0;
    T grad_compact[9];
    switch (edge_edge_distance_type(ea0, ea1, eb0, eb1)) {
        case 0: {
            point_point_distance_gradient(ea0, eb0, grad_compact);
            for (int i = 0; i < 3; ++i)
                grad[i] = grad_compact[i];
            for (int i = 0; i < 3; ++i)
                grad[i + 6] = grad_compact[i + 3];
            break;
        }
        case 1: {
            point_point_distance_gradient(ea0, eb1, grad_compact);
            for (int i = 0; i < 3; ++i)
                grad[i] = grad_compact[i];
            for (int i = 0; i < 3; ++i)
                grad[i + 9] = grad_compact[i + 3];
            break;
        }
        case 2: {
            point_line_distance_gradient(ea0, eb0, eb1, grad_compact);
            for (int i = 0; i < 3; ++i)
                grad[i] = grad_compact[i];
            for (int i = 0; i < 6; ++i)
                grad[i + 6] = grad_compact[i + 3];
            break;
        }
        case 3: {
            point_point_distance_gradient(ea1, eb0, grad_compact);
            for (int i = 0; i < 3; ++i)
                grad[i + 3] = grad_compact[i];
            for (int i = 0; i < 3; ++i)
                grad[i + 6] = grad_compact[i + 3];
            break;
        }
        case 4: {
            point_point_distance_gradient(ea1, eb1, grad_compact);
            for (int i = 0; i < 3; ++i)
                grad[i + 3] = grad_compact[i];
            for (int i = 0; i < 3; ++i)
                grad[i + 9] = grad_compact[i + 3];
            break;
        }
        case 5: {
            point_line_distance_gradient(ea1, eb0, eb1, grad_compact);
            for (int i = 0; i < 3; ++i)
                grad[i + 3] = grad_compact[i];
            for (int i = 0; i < 6; ++i)
                grad[i + 6] = grad_compact[i + 3];
            break;
        }
        case 6: {
            point_line_distance_gradient(eb0, ea0, ea1, grad_compact);
            for (int i = 0; i < 3; ++i)
                grad[i + 6] = grad_compact[i];
            for (int i = 0; i < 6; ++i)
                grad[i] = grad_compact[i + 3];
            break;
        }
        case 7: {
            point_line_distance_gradient(eb1, ea0, ea1, grad_compact);
            for (int i = 0; i < 3; ++i)
                grad[i + 9] = grad_compact[i];
            for (int i = 0; i < 6; ++i)
                grad[i] = grad_compact[i + 3];
            break;
        }
        case 8: {
            line_line_distance_gradient(ea0, ea1, eb0, eb1, grad);
            break;
        }
        default:
            break;
    }
}

template <class T>
MU_FORCE_INLINE MU_HOST_DEV_FUNC void edge_edge_distance_hessian(
    const T* ea0, const T* ea1, const T* eb0, const T* eb1, T* hess, T* hess_compact)
{
    for (int i = 0; i < 144; ++i)
        hess[i] = 0;
    switch (edge_edge_distance_type(ea0, ea1, eb0, eb1)) {
        case 0: {
            point_point_distance_hessian(ea0, eb0, hess_compact);
            for (int i = 0; i < 6; ++i)
                for (int j = 0; j < 6; ++j) {
                    int ii = (i < 3 ? i : i + 3);
                    int jj = (j < 3 ? j : j + 3);
                    hess[ii * 12 + jj] = hess_compact[i * 6 + j];
                }
            break;
        }
        case 1: {
            point_point_distance_hessian(ea0, eb1, hess_compact);
            for (int i = 0; i < 6; ++i)
                for (int j = 0; j < 6; ++j) {
                    int ii = (i < 3 ? i : i + 6);
                    int jj = (j < 3 ? j : j + 6);
                    hess[ii * 12 + jj] = hess_compact[i * 6 + j];
                }
            break;
        }
        case 2: {
            point_line_distance_hessian(ea0, eb0, eb1, hess_compact);
            for (int i = 0; i < 9; ++i)
                for (int j = 0; j < 9; ++j) {
                    int ii = (i < 3 ? i : i + 3);
                    int jj = (j < 3 ? j : j + 3);
                    hess[ii * 12 + jj] = hess_compact[i * 9 + j];
                }
            break;
        }
        case 3: {
            point_point_distance_hessian(ea1, eb0, hess_compact);
            for (int i = 0; i < 6; ++i)
                for (int j = 0; j < 6; ++j) {
                    int ii = (i < 3 ? i + 3 : i + 3);
                    int jj = (j < 3 ? j + 3 : j + 3);
                    hess[ii * 12 + jj] = hess_compact[i * 6 + j];
                }
            break;
        }
        case 4: {
            point_point_distance_hessian(ea1, eb1, hess_compact);
            for (int i = 0; i < 6; ++i)
                for (int j = 0; j < 6; ++j) {
                    int ii = (i < 3 ? i + 3 : i + 6);
                    int jj = (j < 3 ? j + 3 : j + 6);
                    hess[ii * 12 + jj] = hess_compact[i * 6 + j];
                }
            break;
        }
        case 5: {
            point_line_distance_hessian(ea1, eb0, eb1, hess_compact);
            for (int i = 0; i < 9; ++i)
                for (int j = 0; j < 9; ++j) {
                    int ii = (i < 3 ? i + 3 : i + 3);
                    int jj = (j < 3 ? j + 3 : j + 3);
                    hess[ii * 12 + jj] = hess_compact[i * 9 + j];
                }
            break;
        }
        case 6: {
            point_line_distance_hessian(eb0, ea0, ea1, hess_compact);
            for (int i = 0; i < 9; ++i)
                for (int j = 0; j < 9; ++j) {
                    int ii = (i < 3 ? i + 6 : i - 3);
                    int jj = (j < 3 ? j + 6 : j - 3);
                    hess[ii * 12 + jj] = hess_compact[i * 9 + j];
                }
            break;
        }
        case 7: {
            point_line_distance_hessian(eb1, ea0, ea1, hess_compact);
            for (int i = 0; i < 9; ++i)
                for (int j = 0; j < 9; ++j) {
                    int ii = (i < 3 ? i + 9 : i - 3);
                    int jj = (j < 3 ? j + 9 : j - 3);
                    hess[ii * 12 + jj] = hess_compact[i * 9 + j];
                }
            break;
        }
        case 8: {
            line_line_distance_hessian(ea0, ea1, eb0, eb1, hess);
            break;
        }
        default:
            break;
    }
}

}
