#pragma once

#include "point_triangle_distance.h"
#include "distance_common.cuh"

namespace mu::distance
{
template <class T>
MU_FORCE_INLINE MU_HOST_DEV_FUNC bool point_triangle_cd_broadphase(
    const T* p, const T* t0, const T* t1, const T* t2, T dist)
{
    T cd = true;
    T tri_max, tri_min;
    for (int k = 0; k < 3; k++) {
        tri_min = math::min(t0[k], t1[k], t2[k]);
        tri_max = math::max(t0[k], t1[k], t2[k]);
        if ((p[k] - tri_max) > dist || (tri_min - p[k]) > dist) {
            cd = false;
            break;
        }
    }
    return cd;
}

template <class T>
MU_FORCE_INLINE MU_HOST_DEV_FUNC bool point_triangle_ccd_broadphase(const T* p, const T* t0, const T* t1,
    const T* t2, const T* dp, const T* dt0, const T* dt1, const T* dt2, T dist)
{
    bool ccd = true;
    T p_max, p_min, tri_max, tri_min;
    for (int i = 0; i < 3; i++) {
        tri_min = math::min(t0[i], t1[i], t2[i]);
        tri_min = math::min(tri_min, t0[i] + dt0[i], t1[i] + dt1[i]);
        tri_min = math::min(tri_min, t2[i] + dt2[i]);

        tri_max = math::max(t0[i], t1[i], t2[i]);
        tri_max = math::max(tri_max, t0[i] + dt0[i], t1[i] + dt1[i]);
        tri_max = math::max(tri_max, t2[i] + dt2[i]);

        p_min = math::min(p[i], p[i] + dp[i]);
        p_max = math::max(p[i], p[i] + dp[i]);

        if ((p_min - tri_max) > dist || (tri_min - p_max) > dist) {
            ccd = false;
            break;
        }
    }
    return ccd;
}

template <class T>
MU_FORCE_INLINE MU_HOST_DEV_FUNC T point_triangle_ccd(const T* _p, const T* _t0, const T* _t1, const T* _t2,
    const T* _dp, const T* _dt0, const T* _dt1, const T* _dt2, T eta, T thickness)
{
    T p[3], t0[3], t1[3], t2[3], dp[3], dt0[3], dt1[3], dt2[3];
    T mov;
    for (int i = 0; i < 3; i++) {
        p[i] = _p[i];
        dp[i] = _dp[i];
        t0[i] = _t0[i];
        dt0[i] = _dt0[i];
        t1[i] = _t1[i];
        dt1[i] = _dt1[i];
        t2[i] = _t2[i];
        dt2[i] = _dt2[i];
        mov = (dt0[i] + dt1[i] + dt2[i] + dp[i]) / 4;
        dt0[i] -= mov;
        dt1[i] -= mov;
        dt2[i] -= mov;
        dp[i] -= mov;
    }

    T disp_mag2_vec[3];
    disp_mag2_vec[0] = math::norm_sqr<3>(dt0);
    disp_mag2_vec[1] = math::norm_sqr<3>(dt1);
    disp_mag2_vec[2] = math::norm_sqr<3>(dt2);

    T max_disp_mag = math::norm<3>(dp)
        + math::sqrt(math::max(disp_mag2_vec[0], disp_mag2_vec[1], disp_mag2_vec[2]));
    if (max_disp_mag == 0)
        return 1.0;

    T dist2_cur = point_triangle_distance(p, t0, t1, t2);
    T dist_cur = math::sqrt(dist2_cur);
    T gap = eta * (dist2_cur - thickness * thickness) / (dist_cur + thickness);
    T toc = 0.0;

    while (true) {
        T toc_lower_bound = (1 - eta) * (dist2_cur - thickness * thickness)
            / ((dist_cur + thickness) * max_disp_mag);
        for (int i = 0; i < 3; i++) {
            p[i] += toc_lower_bound * dp[i];
            t0[i] += toc_lower_bound * dt0[i];
            t1[i] += toc_lower_bound * dt1[i];
            t2[i] += toc_lower_bound * dt2[i];
        }

        dist2_cur = point_triangle_distance(p, t0, t1, t2);
        dist_cur = math::sqrt(dist2_cur);
        if (toc && ((dist2_cur - thickness * thickness) / (dist_cur + thickness) < gap)) {
            break;
        }

        toc += toc_lower_bound;
        if (toc > 1.0) {
            return 1.0;
        }
    }
    return toc;
}

}
