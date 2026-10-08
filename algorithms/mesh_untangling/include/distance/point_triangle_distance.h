#pragma once

#include "distance_type.h"
#include "point_line_distance.h"
#include "point_plane_distance.h"
#include "point_point_distance.h"
#include "distance_common.cuh"

namespace mu::distance
{
template <class T>
MU_FORCE_INLINE MU_HOST_DEV_FUNC void g_PT_signed(
    T v01, T v02, T v03, T v11, T v12, T v13, T v21, T v22, T v23, T v31, T v32, T v33, T* g)
{
    T g_tmp;
    T t11;
    T t12;
    T t13;
    T t14;
    T t15;
    T t16;
    T t17;
    T t18;
    T t19;
    T t20;
    T t21;
    T t22;
    T t32;
    T t33;
    T t34;
    T t35;
    T t36;
    T t37;
    T t38;
    T t39;
    T t40;
    T t49;
    t11 = -v11 + v01;
    t12 = -v12 + v02;
    t13 = -v13 + v03;
    t14 = -v21 + v11;
    t15 = -v22 + v12;
    t16 = -v23 + v13;
    t17 = -v31 + v11;
    t18 = -v32 + v12;
    t19 = -v33 + v13;
    t20 = -v31 + v21;
    t21 = -v32 + v22;
    t22 = -v33 + v23;
    t32 = t14 * t18 + -(t15 * t17);
    t33 = t14 * t19 + -(t16 * t17);
    t34 = t15 * t19 + -(t16 * t18);
    t35 = math::abs(t32);
    t36 = math::abs(t33);
    t37 = math::abs(t34);
    t38 = t32;
    if (t32 < 0.0) {
        t38 = -1.0;
    } else if (t32 > 0.0) {
        t38 = 1.0;
    }
    t39 = t33;
    if (t33 < 0.0) {
        t39 = -1.0;
    } else if (t33 > 0.0) {
        t39 = 1.0;
    }
    t40 = t34;
    if (t34 < 0.0) {
        t40 = -1.0;
    } else if (t34 > 0.0) {
        t40 = 1.0;
    }
    t49 = 1.0 / math::sqrt((t35 * t35 + t36 * t36) + t37 * t37);
    g[0] = t34 * t49;
    g[1] = -t33 * t49;
    g[2] = t32 * t49;
    g_tmp = math::pow(t49, 3.0) * ((t13 * t32 + t11 * t34) + -(t12 * t33));
    g[3] = -t49 * ((t34 + t12 * t22) - t13 * t21)
        - g_tmp * (t21 * t35 * t38 * 2.0 + t22 * t36 * t39 * 2.0) / 2.0;
    g[4] = t49 * ((t33 + t11 * t22) - t13 * t20)
        + g_tmp * (t20 * t35 * t38 * 2.0 - t22 * t37 * t40 * 2.0) / 2.0;
    g[5] = -t49 * ((t32 + t11 * t21) - t12 * t20)
        + g_tmp * (t20 * t36 * t39 * 2.0 + t21 * t37 * t40 * 2.0) / 2.0;
    g[6] = t49 * (t12 * t19 - t13 * t18)
        + g_tmp * (t18 * t35 * t38 * 2.0 + t19 * t36 * t39 * 2.0) / 2.0;
    g[7] = -t49 * (t11 * t19 - t13 * t17)
        - g_tmp * (t17 * t35 * t38 * 2.0 - t19 * t37 * t40 * 2.0) / 2.0;
    g[8] = t49 * (t11 * t18 - t12 * t17)
        - g_tmp * (t17 * t36 * t39 * 2.0 + t18 * t37 * t40 * 2.0) / 2.0;
    g[9] = -t49 * (t12 * t16 - t13 * t15)
        - g_tmp * (t15 * t35 * t38 * 2.0 + t16 * t36 * t39 * 2.0) / 2.0;
    g[10] = t49 * (t11 * t16 - t13 * t14)
        + g_tmp * (t14 * t35 * t38 * 2.0 - t16 * t37 * t40 * 2.0) / 2.0;
    g[11] = -t49 * (t11 * t15 - t12 * t14)
        + g_tmp * (t14 * t36 * t39 * 2.0 + t15 * t37 * t40 * 2.0) / 2.0;
}

template <class T>
MU_FORCE_INLINE MU_HOST_DEV_FUNC T point_triangle_distance(const T* p, const T* t0, const T* t1, const T* t2)
{
    switch (point_triangle_distance_type(p, t0, t1, t2)) {
        case 0:
            return point_point_distance(p, t0);
        case 1:
            return point_point_distance(p, t1);
        case 2:
            return point_point_distance(p, t2);
        case 3:
            return point_line_distance(p, t0, t1);
        case 4:
            return point_line_distance(p, t1, t2);
        case 5:
            return point_line_distance(p, t2, t0);
        case 6:
            return point_plane_distance(p, t0, t1, t2);
        default:
            return 1e20;
    }
}

template <class T>
MU_FORCE_INLINE MU_HOST_DEV_FUNC void point_triangle_distance_gradient(
    const T* p, const T* t0, const T* t1, const T* t2, T* grad)
{
    for (int i = 0; i < 12; ++i)
        grad[i] = 0;
    T grad_compact[9];
    switch (point_triangle_distance_type(p, t0, t1, t2)) {
        case 0: {
            point_point_distance_gradient(p, t0, grad_compact);
            for (int i = 0; i < 6; ++i)
                grad[i] = grad_compact[i];
            break;
        }
        case 1: {
            point_point_distance_gradient(p, t1, grad_compact);
            for (int i = 0; i < 3; ++i)
                grad[i] = grad_compact[i];
            for (int i = 0; i < 3; ++i)
                grad[i + 6] = grad_compact[i + 3];
            break;
        }
        case 2: {
            point_point_distance_gradient(p, t2, grad_compact);
            for (int i = 0; i < 3; ++i)
                grad[i] = grad_compact[i];
            for (int i = 0; i < 3; ++i)
                grad[i + 9] = grad_compact[i + 3];
            break;
        }
        case 3: {
            point_line_distance_gradient(p, t0, t1, grad_compact);
            for (int i = 0; i < 9; ++i)
                grad[i] = grad_compact[i];
            break;
        }
        case 4: {
            point_line_distance_gradient(p, t1, t2, grad_compact);
            for (int i = 0; i < 9; ++i) {
                int ii = (i < 3 ? i : i + 3);
                grad[ii] = grad_compact[i];
            }
            break;
        }
        case 5: {
            point_line_distance_gradient(p, t0, t2, grad_compact);
            for (int i = 0; i < 9; ++i) {
                int ii = (i < 6 ? i : i + 3);
                grad[ii] = grad_compact[i];
            }
            break;
        }
        case 6: {
            point_plane_distance_gradient(p, t0, t1, t2, grad);
            break;
        }
        default:
            break;
    }
}

template <class T>
MU_FORCE_INLINE MU_HOST_DEV_FUNC void point_triangle_distance_hessian(
    const T* p, const T* t0, const T* t1, const T* t2, T* hess, T* hess_compact)
{
    for (int i = 0; i < 144; ++i)
        hess[i] = 0;
    switch (point_triangle_distance_type(p, t0, t1, t2)) {
        case 0: {
            point_point_distance_hessian(p, t0, hess_compact);
            for (int i = 0; i < 6; ++i)
                for (int j = 0; j < 6; ++j)
                    hess[i * 12 + j] = hess_compact[i * 6 + j];
            break;
        }
        case 1: {
            point_point_distance_hessian(p, t1, hess_compact);
            for (int i = 0; i < 6; ++i)
                for (int j = 0; j < 6; ++j) {
                    int ii = (i < 3 ? i : i + 3);
                    int jj = (j < 3 ? j : j + 3);
                    hess[ii * 12 + jj] = hess_compact[i * 6 + j];
                }
            break;
        }
        case 2: {
            point_point_distance_hessian(p, t2, hess_compact);
            for (int i = 0; i < 6; ++i)
                for (int j = 0; j < 6; ++j) {
                    int ii = (i < 3 ? i : i + 6);
                    int jj = (j < 3 ? j : j + 6);
                    hess[ii * 12 + jj] = hess_compact[i * 6 + j];
                }
            break;
        }
        case 3: {
            point_line_distance_hessian(p, t0, t1, hess_compact);
            for (int i = 0; i < 9; ++i)
                for (int j = 0; j < 9; ++j)
                    hess[i * 12 + j] = hess_compact[i * 9 + j];
            break;
        }
        case 4: {
            point_line_distance_hessian(p, t1, t2, hess_compact);
            for (int i = 0; i < 9; ++i)
                for (int j = 0; j < 9; ++j) {
                    int ii = (i < 3 ? i : i + 3);
                    int jj = (j < 3 ? j : j + 3);
                    hess[ii * 12 + jj] = hess_compact[i * 9 + j];
                }
            break;
        }
        case 5: {
            point_line_distance_hessian(p, t0, t2, hess_compact);
            for (int i = 0; i < 9; ++i)
                for (int j = 0; j < 9; ++j) {
                    int ii = (i < 6 ? i : i + 3);
                    int jj = (j < 6 ? j : j + 3);
                    hess[ii * 12 + jj] = hess_compact[i * 9 + j];
                }
            break;
        }
        case 6: {
            point_plane_distance_hessian(p, t0, t1, t2, hess);
            break;
        }
        default:
            break;
    }
}

template <class T>
MU_FORCE_INLINE MU_HOST_DEV_FUNC void H_PT_signed(
    T v01, T v02, T v03, T v11, T v12, T v13, T v21, T v22, T v23, T v31, T v32, T v33, T* h)
{
    double b_t583_tmp;
    double b_t584_tmp;
    double b_t585_tmp;
    double b_t586_tmp;
    double b_t589_tmp;
    double b_t590_tmp;
    double c_t583_tmp;
    double c_t584_tmp;
    double c_t589_tmp;
    double ct_idx_117;
    double ct_idx_119;
    double ct_idx_127;
    double ct_idx_128;
    double ct_idx_133;
    double ct_idx_51;
    double ct_idx_52;
    double ct_idx_54;
    double ct_idx_60;
    double ct_idx_66;
    double ct_idx_71;
    double ct_idx_72;
    double ct_idx_77;
    double ct_idx_78;
    double ct_idx_81;
    double ct_idx_82;
    double ct_idx_87;
    double ct_idx_98;
    double d_t583_tmp;
    double e_t583_tmp;
    double t11;
    double t12;
    double t13;
    double t150;
    double t151;
    double t152;
    double t18;
    double t20;
    double t22;
    double t223;
    double t224;
    double t226;
    double t227;
    double t228;
    double t229;
    double t23;
    double t230;
    double t231;
    double t232;
    double t233;
    double t234;
    double t236;
    double t238;
    double t239;
    double t24;
    double t240;
    double t241;
    double t242;
    double t25;
    double t26;
    double t262;
    double t27;
    double t272;
    double t273;
    double t274;
    double t275;
    double t276;
    double t277;
    double t278;
    double t279;
    double t28;
    double t280;
    double t467;
    double t468;
    double t469;
    double t470;
    double t471;
    double t472;
    double t473;
    double t474;
    double t475;
    double t476;
    double t477;
    double t478;
    double t479;
    double t480;
    double t481;
    double t482;
    double t483;
    double t484;
    double t56;
    double t57;
    double t58;
    double t583;
    double t583_tmp;
    double t584;
    double t584_tmp;
    double t585;
    double t585_tmp;
    double t586;
    double t586_tmp;
    double t587;
    double t588;
    double t589;
    double t589_tmp;
    double t59;
    double t590;
    double t590_tmp;
    double t591;
    double t592;
    double t593;
    double t594;
    double t595;
    double t596;
    double t597;
    double t60;
    double t607;
    double t609;
    double t61;
    double t611;
    double t80;
    double t81;
    double t82;
    double t83;
    double t84;
    double t85;
    double t86;
    double t87;
    double t88;
    double t89;
    double t90;
    double t91;
    double t95;
    double t96;
    double t97;
    t11 = -v11 + v01;
    t12 = -v12 + v02;
    t13 = -v13 + v03;
    t18 = -v21 + v11;
    t20 = -v22 + v12;
    t22 = -v23 + v13;
    t23 = -v31 + v11;
    t24 = -v32 + v12;
    t25 = -v33 + v13;
    t26 = -v31 + v21;
    t27 = -v32 + v22;
    t28 = -v33 + v23;
    t56 = t18 * t24;
    t57 = t20 * t23;
    t58 = t18 * t25;
    t59 = t22 * t23;
    t60 = t20 * t25;
    t61 = t22 * t24;
    t80 = t56 + -t57;
    t81 = t58 + -t59;
    t82 = t60 + -t61;
    t83 = fabs(t80);
    t84 = fabs(t81);
    t85 = fabs(t82);
    t86 = (double)(t80 == 0.0) * 1.0E+10;
    t87 = (double)(t81 == 0.0) * 1.0E+10;
    t88 = (double)(t82 == 0.0) * 1.0E+10;
    t89 = t80;
    if (t80 < 0.0) {
        t89 = -1.0;
    } else if (t80 > 0.0) {
        t89 = 1.0;
    }
    t90 = t81;
    if (t81 < 0.0) {
        t90 = -1.0;
    } else if (t81 > 0.0) {
        t90 = 1.0;
    }
    t91 = t82;
    if (t82 < 0.0) {
        t91 = -1.0;
    } else if (t82 > 0.0) {
        t91 = 1.0;
    }
    t95 = t89 * t89;
    t96 = t90 * t90;
    t97 = t91 * t91;
    t150 = t83 * t89 * 2.0;
    t151 = t84 * t90 * 2.0;
    t152 = t85 * t91 * 2.0;
    t262 = (t13 * t80 + t11 * t82) + -(t12 * t81);
    t223 = 1.0 / sqrt((t83 * t83 + t84 * t84) + t85 * t85);
    t272 = t20 * t150 + t22 * t151;
    t273 = t18 * t151 + t20 * t152;
    t274 = t24 * t150 + t25 * t151;
    t275 = t23 * t151 + t24 * t152;
    t276 = t27 * t150 + t28 * t151;
    t277 = t26 * t151 + t27 * t152;
    t278 = t18 * t150 + t22 * t85 * t91 * -2.0;
    t279 = t23 * t150 + t25 * t85 * t91 * -2.0;
    t280 = t26 * t150 + t28 * t85 * t91 * -2.0;
    t224 = pow(t223, 3.0);
    t226 = t11 * t223;
    t227 = t12 * t223;
    t228 = t13 * t223;
    t229 = (-v21 + v01) * t223;
    t230 = (-v22 + v02) * t223;
    t231 = (-v23 + v03) * t223;
    t232 = (-v31 + v01) * t223;
    t233 = t18 * t223;
    t234 = (-v32 + v02) * t223;
    t91 = t20 * t223;
    t236 = (-v33 + v03) * t223;
    t89 = t22 * t223;
    t238 = t23 * t223;
    t239 = t24 * t223;
    t240 = t25 * t223;
    t241 = t26 * t223;
    t242 = t27 * t223;
    t90 = t28 * t223;
    ct_idx_51 = t18 * t18;
    ct_idx_52 = t20 * t20;
    ct_idx_54 = t22 * t22;
    ct_idx_60 = t23 * t23;
    ct_idx_66 = t24 * t24;
    ct_idx_71 = t25 * t25;
    t609 = t80 * t224;
    ct_idx_72 = t609 * t273 / 2.0;
    t607 = t82 * t224;
    ct_idx_77 = t607 * t274 / 2.0;
    ct_idx_78 = t26 * t26;
    ct_idx_81 = t607 * t276 / 2.0;
    ct_idx_82 = t609 * t277 / 2.0;
    ct_idx_87 = t27 * t27;
    ct_idx_98 = t28 * t28;
    ct_idx_117 = -(t607 * t272 / 2.0);
    t611 = t81 * t224;
    ct_idx_119 = t611 * t278 / 2.0;
    ct_idx_127 = t611 * t279 / 2.0;
    ct_idx_128 = -(t609 * t275 / 2.0);
    ct_idx_133 = t611 * t280 / 2.0;
    t469 = t89 + t607 * t278 / 2.0;
    t474 = t90 + t607 * t280 / 2.0;
    t475 = -t89 + t611 * t272 / 2.0;
    t476 = -t91 + t607 * t273 / 2.0;
    t477 = -t239 + t609 * t274 / 2.0;
    t479 = -t238 + t611 * t275 / 2.0;
    t480 = -t90 + t611 * t276 / 2.0;
    t481 = -t242 + t607 * t277 / 2.0;
    t482 = -t233 + t609 * t278 / 2.0;
    t483 = -t241 + t609 * t280 / 2.0;
    t467 = t91 + -(t609 * t272 / 2.0);
    t468 = t233 + -(t611 * t273 / 2.0);
    t470 = t240 + -(t611 * t274 / 2.0);
    t471 = t239 + -(t607 * t275 / 2.0);
    t472 = t242 + -(t609 * t276 / 2.0);
    t473 = t241 + -(t611 * t277 / 2.0);
    t478 = t238 + -(t609 * t279 / 2.0);
    t484 = -t240 + -(t607 * t279 / 2.0);
    t89 = t18 * t22;
    t583_tmp = t224 * t262;
    b_t583_tmp = pow(t223, 5.0) * t262;
    c_t583_tmp = (t11 * t20 + -(t12 * t18)) * t224;
    d_t583_tmp = (t12 * t22 + -(t13 * t20)) * t224;
    e_t583_tmp = b_t583_tmp * t272;
    t583 = ((c_t583_tmp * t272 / 2.0 + -(d_t583_tmp * t273 / 2.0))
               + t583_tmp * (t89 * t96 * 2.0 + t89 * t84 * t87 * 4.0) / 2.0)
        + -(e_t583_tmp * t273 * 0.75);
    t89 = t23 * t25;
    t584_tmp = (t11 * t24 + -(t12 * t23)) * t224;
    b_t584_tmp = (t12 * t25 + -(t13 * t24)) * t224;
    c_t584_tmp = b_t583_tmp * t274;
    t584 = ((t584_tmp * t274 / 2.0 + -(b_t584_tmp * t275 / 2.0))
               + t583_tmp * (t89 * t96 * 2.0 + t89 * t84 * t87 * 4.0) / 2.0)
        + -(c_t584_tmp * t275 * 0.75);
    t89 = t20 * t22;
    t585_tmp = (t11 * t22 + -(t13 * t18)) * t224;
    b_t585_tmp = b_t583_tmp * t273;
    t585 = ((t585_tmp * t273 / 2.0 + -(c_t583_tmp * t278 / 2.0))
               + t583_tmp * (t89 * t97 * 2.0 + t89 * t85 * t88 * 4.0) / 2.0)
        + b_t585_tmp * t278 * 0.75;
    t89 = t24 * t25;
    t586_tmp = (t11 * t25 + -(t13 * t23)) * t224;
    b_t586_tmp = b_t583_tmp * t275;
    t586 = ((t586_tmp * t275 / 2.0 + -(t584_tmp * t279 / 2.0))
               + t583_tmp * (t89 * t97 * 2.0 + t89 * t85 * t88 * 4.0) / 2.0)
        + b_t586_tmp * t279 * 0.75;
    t89 = t18 * t20;
    t587 = ((-(t585_tmp * t272 / 2.0) + -(d_t583_tmp * t278 / 2.0))
               + t583_tmp * (t89 * t95 * 2.0 + t89 * t83 * t86 * 4.0) / 2.0)
        + -(e_t583_tmp * t278 * 0.75);
    t89 = t23 * t24;
    t588 = ((-(t586_tmp * t274 / 2.0) + -(b_t584_tmp * t279 / 2.0))
               + t583_tmp * (t89 * t95 * 2.0 + t89 * t83 * t86 * 4.0) / 2.0)
        + -(c_t584_tmp * t279 * 0.75);
    t89 = t26 * t28;
    t589_tmp = ((t11 * t27 + -(t12 * t26)) + t80) * t224;
    b_t589_tmp = ((t12 * t28 + -(t13 * t27)) + t82) * t224;
    c_t589_tmp = b_t583_tmp * t276;
    t589 = ((t589_tmp * t276 / 2.0 + -(b_t589_tmp * t277 / 2.0))
               + t583_tmp * (t89 * t96 * 2.0 + t89 * t84 * t87 * 4.0) / 2.0)
        + -(c_t589_tmp * t277 * 0.75);
    t89 = t27 * t28;
    t590_tmp = ((t11 * t28 + -(t13 * t26)) + t81) * t224;
    b_t590_tmp = b_t583_tmp * t277;
    t590 = ((t590_tmp * t277 / 2.0 + -(t589_tmp * t280 / 2.0))
               + t583_tmp * (t89 * t97 * 2.0 + t89 * t85 * t88 * 4.0) / 2.0)
        + b_t590_tmp * t280 * 0.75;
    t89 = t26 * t27;
    t591 = ((-(t590_tmp * t276 / 2.0) + -(b_t589_tmp * t280 / 2.0))
               + t583_tmp * (t89 * t95 * 2.0 + t89 * t83 * t86 * 4.0) / 2.0)
        + -(c_t589_tmp * t280 * 0.75);
    t90 = t18 * t23;
    t89 = t20 * t24;
    t592 = ((t584_tmp * t273 / 2.0 + c_t583_tmp * t275 / 2.0)
               + t583_tmp
                   * (((t90 * t96 * 2.0 + t89 * t97 * 2.0) + t90 * t84 * t87 * 4.0)
                       + t89 * t85 * t88 * 4.0)
                   / 2.0)
        + -(b_t585_tmp * t275 * 0.75);
    t91 = t22 * t25;
    t593 = ((-(b_t584_tmp * t272 / 2.0) + -(d_t583_tmp * t274 / 2.0))
               + t583_tmp
                   * (((t89 * t95 * 2.0 + t91 * t96 * 2.0) + t89 * t83 * t86 * 4.0)
                       + t91 * t84 * t87 * 4.0)
                   / 2.0)
        + -(e_t583_tmp * t274 * 0.75);
    t242 = b_t583_tmp * t278;
    t594 = ((-(t586_tmp * t278 / 2.0) + -(t585_tmp * t279 / 2.0))
               + t583_tmp
                   * (((t90 * t95 * 2.0 + t91 * t97 * 2.0) + t90 * t83 * t86 * 4.0)
                       + t91 * t85 * t88 * 4.0)
                   / 2.0)
        + -(t242 * t279 * 0.75);
    t89 = t20 * t27;
    t91 = t22 * t28;
    t595 = ((d_t583_tmp * t276 / 2.0 + b_t589_tmp * t272 / 2.0)
               + -(t583_tmp
                   * (((t89 * t95 * 2.0 + t91 * t96 * 2.0) + t89 * t83 * t86 * 4.0)
                       + t91 * t84 * t87 * 4.0)
                   / 2.0))
        + e_t583_tmp * t276 * 0.75;
    t239 = t23 * t26;
    t90 = t24 * t27;
    t596 = ((t584_tmp * t277 / 2.0 + t589_tmp * t275 / 2.0)
               + t583_tmp
                   * (((t239 * t96 * 2.0 + t90 * t97 * 2.0) + t239 * t84 * t87 * 4.0)
                       + t90 * t85 * t88 * 4.0)
                   / 2.0)
        + -(b_t586_tmp * t277 * 0.75);
    t233 = t18 * t26;
    t597 = ((-(c_t583_tmp * t277 / 2.0) + -(t589_tmp * t273 / 2.0))
               + -(t583_tmp
                   * (((t233 * t96 * 2.0 + t89 * t97 * 2.0) + t233 * t84 * t87 * 4.0)
                       + t89 * t85 * t88 * 4.0)
                   / 2.0))
        + b_t585_tmp * t277 * 0.75;
    t89 = t25 * t28;
    t81 = ((-(b_t584_tmp * t276 / 2.0) + -(b_t589_tmp * t274 / 2.0))
              + t583_tmp
                  * (((t90 * t95 * 2.0 + t89 * t96 * 2.0) + t90 * t83 * t86 * 4.0)
                      + t89 * t84 * t87 * 4.0)
                  / 2.0)
        + -(c_t584_tmp * t276 * 0.75);
    t13 = ((t585_tmp * t280 / 2.0 + t590_tmp * t278 / 2.0)
              + -(t583_tmp
                  * (((t233 * t95 * 2.0 + t91 * t97 * 2.0) + t233 * t83 * t86 * 4.0)
                      + t91 * t85 * t88 * 4.0)
                  / 2.0))
        + t242 * t280 * 0.75;
    t11 = ((-(t586_tmp * t280 / 2.0) + -(t590_tmp * t279 / 2.0))
              + t583_tmp
                  * (((t239 * t95 * 2.0 + t89 * t97 * 2.0) + t239 * t83 * t86 * 4.0)
                      + t89 * t85 * t88 * 4.0)
                  / 2.0)
        + -(b_t583_tmp * t279 * t280 * 0.75);
    t224 = (((t228 + t585_tmp * t274 / 2.0) + b_t584_tmp * t278 / 2.0)
               + -(t583_tmp * ((t56 * t95 * 2.0 + t150) + t56 * t83 * t86 * 4.0) / 2.0))
        + c_t584_tmp * t278 * 0.75;
    t82 = (((t227 + d_t583_tmp * t275 / 2.0) + -(t584_tmp * t272 / 2.0))
              + -(t583_tmp * ((t59 * t96 * 2.0 + -t151) + t59 * t84 * t87 * 4.0) / 2.0))
        + e_t583_tmp * t275 * 0.75;
    t12 = (((-t227 + b_t584_tmp * t273 / 2.0) + -(c_t583_tmp * t274 / 2.0))
              + -(t583_tmp * ((t58 * t96 * 2.0 + t151) + t58 * t84 * t87 * 4.0) / 2.0))
        + b_t585_tmp * t274 * 0.75;
    t80 = (((t226 + c_t583_tmp * t279 / 2.0) + -(t586_tmp * t273 / 2.0))
              + -(t583_tmp * ((t60 * t97 * 2.0 + t152) + t60 * t85 * t88 * 4.0) / 2.0))
        + -(b_t585_tmp * t279 * 0.75);
    t223 = (((-t228 + t586_tmp * t272 / 2.0) + d_t583_tmp * t279 / 2.0)
               + -(t583_tmp * ((t57 * t95 * 2.0 + -t150) + t57 * t83 * t86 * 4.0) / 2.0))
        + e_t583_tmp * t279 * 0.75;
    t262 = (((-t226 + t584_tmp * t278 / 2.0) + -(t585_tmp * t275 / 2.0))
               + -(t583_tmp * ((t61 * t97 * 2.0 + -t152) + t61 * t85 * t88 * 4.0) / 2.0))
        + -(b_t586_tmp * t278 * 0.75);
    t89 = t18 * t28;
    t607 = (((t230 + c_t583_tmp * t276 / 2.0) + -(b_t589_tmp * t273 / 2.0))
               + t583_tmp * ((t89 * t96 * 2.0 + t151) + t89 * t84 * t87 * 4.0) / 2.0)
        + -(b_t585_tmp * t276 * 0.75);
    t89 = t23 * t27;
    t240 = (((t236 + t586_tmp * t276 / 2.0) + b_t589_tmp * t279 / 2.0)
               + -(t583_tmp * ((t89 * t95 * 2.0 + t150) + t89 * t83 * t86 * 4.0) / 2.0))
        + c_t589_tmp * t279 * 0.75;
    t89 = t25 * t26;
    t609 = (((t234 + b_t584_tmp * t277 / 2.0) + -(t589_tmp * t274 / 2.0))
               + -(t583_tmp * ((t89 * t96 * 2.0 + -t151) + t89 * t84 * t87 * 4.0) / 2.0))
        + c_t584_tmp * t277 * 0.75;
    t89 = t22 * t27;
    t238 = (((t229 + t585_tmp * t277 / 2.0) + -(t589_tmp * t278 / 2.0))
               + t583_tmp * ((t89 * t97 * 2.0 + -t152) + t89 * t85 * t88 * 4.0) / 2.0)
        + b_t590_tmp * t278 * 0.75;
    t89 = t23 * t28;
    t611 = (((-t234 + -(t584_tmp * t276 / 2.0)) + b_t589_tmp * t275 / 2.0)
               + -(t583_tmp * ((t89 * t96 * 2.0 + t151) + t89 * t84 * t87 * 4.0) / 2.0))
        + b_t586_tmp * t276 * 0.75;
    t89 = t22 * t26;
    t241 = (((-t230 + -(d_t583_tmp * t277 / 2.0)) + t589_tmp * t272 / 2.0)
               + t583_tmp * ((t89 * t96 * 2.0 + -t151) + t89 * t84 * t87 * 4.0) / 2.0)
        + -(e_t583_tmp * t277 * 0.75);
    t89 = t20 * t28;
    t242 = (((-t229 + -(c_t583_tmp * t280 / 2.0)) + t590_tmp * t273 / 2.0)
               + t583_tmp * ((t89 * t97 * 2.0 + t152) + t89 * t85 * t88 * 4.0) / 2.0)
        + b_t585_tmp * t280 * 0.75;
    t89 = t24 * t28;
    t239 = (((t232 + t584_tmp * t280 / 2.0) + -(t590_tmp * t275 / 2.0))
               + -(t583_tmp * ((t89 * t97 * 2.0 + t152) + t89 * t85 * t88 * 4.0) / 2.0))
        + -(b_t586_tmp * t280 * 0.75);
    t89 = t20 * t26;
    t233 = (((t231 + -(d_t583_tmp * t280 / 2.0)) + -(t590_tmp * t272 / 2.0))
               + t583_tmp * ((t89 * t95 * 2.0 + -t150) + t89 * t83 * t86 * 4.0) / 2.0)
        + -(e_t583_tmp * t280 * 0.75);
    t89 = t24 * t26;
    t91 = (((-t236 + b_t584_tmp * t280 / 2.0) + t590_tmp * t274 / 2.0)
              + -(t583_tmp * ((t89 * t95 * 2.0 + -t150) + t89 * t83 * t86 * 4.0) / 2.0))
        + c_t584_tmp * t280 * 0.75;
    t89 = t18 * t27;
    t90 = (((-t231 + -(t585_tmp * t276 / 2.0)) + -(b_t589_tmp * t278 / 2.0))
              + t583_tmp * ((t89 * t95 * 2.0 + t150) + t89 * t83 * t86 * 4.0) / 2.0)
        + -(c_t589_tmp * t278 * 0.75);
    t89 = t25 * t27;
    t89 = (((-t232 + -(t586_tmp * t277 / 2.0)) + t589_tmp * t279 / 2.0)
              + -(t583_tmp * ((t89 * t97 * 2.0 + -t152) + t89 * t85 * t88 * 4.0) / 2.0))
        + -(b_t590_tmp * t279 * 0.75);
    h[0] = 0.0;
    h[1] = 0.0;
    h[2] = 0.0;
    h[3] = -ct_idx_81;
    h[4] = t474;
    h[5] = t481;
    h[6] = ct_idx_77;
    h[7] = t484;
    h[8] = t471;
    h[9] = ct_idx_117;
    h[10] = t469;
    h[11] = t476;
    h[12] = 0.0;
    h[13] = 0.0;
    h[14] = 0.0;
    h[15] = t480;
    h[16] = -ct_idx_133;
    h[17] = t473;
    h[18] = t470;
    h[19] = ct_idx_127;
    h[20] = t479;
    h[21] = t475;
    h[22] = -ct_idx_119;
    h[23] = t468;
    h[24] = 0.0;
    h[25] = 0.0;
    h[26] = 0.0;
    h[27] = t472;
    h[28] = t483;
    h[29] = ct_idx_82;
    h[30] = t477;
    h[31] = t478;
    h[32] = ct_idx_128;
    h[33] = t467;
    h[34] = t482;
    h[35] = ct_idx_72;
    h[36] = -ct_idx_81;
    h[37] = t480;
    h[38] = t472;
    h[39]
        = (b_t583_tmp * (t276 * t276) * 0.75
              - t583_tmp
                  * (((ct_idx_87 * t95 * 2.0 + ct_idx_98 * t96 * 2.0) + ct_idx_87 * t83 * t86 * 4.0)
                      + ct_idx_98 * t84 * t87 * 4.0)
                  / 2.0)
        + b_t589_tmp * t276;
    h[40] = t591;
    h[41] = t589;
    h[42] = t81;
    h[43] = t240;
    h[44] = t611;
    h[45] = t595;
    h[46] = t90;
    h[47] = t607;
    h[48] = t474;
    h[49] = -ct_idx_133;
    h[50] = t483;
    h[51] = t591;
    h[52]
        = (b_t583_tmp * (t280 * t280) * 0.75
              - t583_tmp
                  * (((ct_idx_78 * t95 * 2.0 + ct_idx_98 * t97 * 2.0) + ct_idx_78 * t83 * t86 * 4.0)
                      + ct_idx_98 * t85 * t88 * 4.0)
                  / 2.0)
        + t590_tmp * t280;
    h[53] = t590;
    h[54] = t91;
    h[55] = t11;
    h[56] = t239;
    h[57] = t233;
    h[58] = t13;
    h[59] = t242;
    h[60] = t481;
    h[61] = t473;
    h[62] = ct_idx_82;
    h[63] = t589;
    h[64] = t590;
    h[65]
        = (b_t583_tmp * (t277 * t277) * 0.75
              - t583_tmp
                  * (((ct_idx_78 * t96 * 2.0 + ct_idx_87 * t97 * 2.0) + ct_idx_78 * t84 * t87 * 4.0)
                      + ct_idx_87 * t85 * t88 * 4.0)
                  / 2.0)
        - t589_tmp * t277;
    h[66] = t609;
    h[67] = t89;
    h[68] = t596;
    h[69] = t241;
    h[70] = t238;
    h[71] = t597;
    h[72] = ct_idx_77;
    h[73] = t470;
    h[74] = t477;
    h[75] = t81;
    h[76] = t91;
    h[77] = t609;
    h[78]
        = (b_t583_tmp * (t274 * t274) * 0.75
              - t583_tmp
                  * (((ct_idx_66 * t95 * 2.0 + ct_idx_71 * t96 * 2.0) + ct_idx_66 * t83 * t86 * 4.0)
                      + ct_idx_71 * t84 * t87 * 4.0)
                  / 2.0)
        + b_t584_tmp * t274;
    h[79] = t588;
    h[80] = t584;
    h[81] = t593;
    h[82] = t224;
    h[83] = t12;
    h[84] = t484;
    h[85] = ct_idx_127;
    h[86] = t478;
    h[87] = t240;
    h[88] = t11;
    h[89] = t89;
    h[90] = t588;
    h[91]
        = (b_t583_tmp * (t279 * t279) * 0.75
              - t583_tmp
                  * (((ct_idx_60 * t95 * 2.0 + ct_idx_71 * t97 * 2.0) + ct_idx_60 * t83 * t86 * 4.0)
                      + ct_idx_71 * t85 * t88 * 4.0)
                  / 2.0)
        + t586_tmp * t279;
    h[92] = t586;
    h[93] = t223;
    h[94] = t594;
    h[95] = t80;
    h[96] = t471;
    h[97] = t479;
    h[98] = ct_idx_128;
    h[99] = t611;
    h[100] = t239;
    h[101] = t596;
    h[102] = t584;
    h[103] = t586;
    h[104]
        = (b_t583_tmp * (t275 * t275) * 0.75
              - t583_tmp
                  * (((ct_idx_60 * t96 * 2.0 + ct_idx_66 * t97 * 2.0) + ct_idx_60 * t84 * t87 * 4.0)
                      + ct_idx_66 * t85 * t88 * 4.0)
                  / 2.0)
        - t584_tmp * t275;
    h[105] = t82;
    h[106] = t262;
    h[107] = t592;
    h[108] = ct_idx_117;
    h[109] = t475;
    h[110] = t467;
    h[111] = t595;
    h[112] = t233;
    h[113] = t241;
    h[114] = t593;
    h[115] = t223;
    h[116] = t82;
    h[117]
        = (b_t583_tmp * (t272 * t272) * 0.75
              - t583_tmp
                  * (((ct_idx_52 * t95 * 2.0 + ct_idx_54 * t96 * 2.0) + ct_idx_52 * t83 * t86 * 4.0)
                      + ct_idx_54 * t84 * t87 * 4.0)
                  / 2.0)
        + d_t583_tmp * t272;
    h[118] = t587;
    h[119] = t583;
    h[120] = t469;
    h[121] = -ct_idx_119;
    h[122] = t482;
    h[123] = t90;
    h[124] = t13;
    h[125] = t238;
    h[126] = t224;
    h[127] = t594;
    h[128] = t262;
    h[129] = t587;
    h[130]
        = (b_t583_tmp * (t278 * t278) * 0.75
              - t583_tmp
                  * (((ct_idx_51 * t95 * 2.0 + ct_idx_54 * t97 * 2.0) + ct_idx_51 * t83 * t86 * 4.0)
                      + ct_idx_54 * t85 * t88 * 4.0)
                  / 2.0)
        + t585_tmp * t278;
    h[131] = t585;
    h[132] = t476;
    h[133] = t468;
    h[134] = ct_idx_72;
    h[135] = t607;
    h[136] = t242;
    h[137] = t597;
    h[138] = t12;
    h[139] = t80;
    h[140] = t592;
    h[141] = t583;
    h[142] = t585;
    h[143]
        = (b_t583_tmp * (t273 * t273) * 0.75
              - t583_tmp
                  * (((ct_idx_51 * t96 * 2.0 + ct_idx_52 * t97 * 2.0) + ct_idx_51 * t84 * t87 * 4.0)
                      + ct_idx_52 * t85 * t88 * 4.0)
                  / 2.0)
        - c_t583_tmp * t273;
}

}
