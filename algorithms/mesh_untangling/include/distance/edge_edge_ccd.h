#pragma once

#include "edge_edge_distance.h"
#include "distance_common.cuh"

namespace mu::distance
{
template <class T>
MU_FORCE_INLINE MU_HOST_DEV_FUNC bool edge_edge_cd_broadphase(
    const T* ea0, const T* ea1, const T* eb0, const T* eb1, T dist)
{
    bool cd = true;
    T max_a, min_a, max_b, min_b;
    for (int k = 0; k < 3; k++) {
        min_a = math::min(ea0[k], ea1[k]);
        max_a = math::max(ea0[k], ea1[k]);
        min_b = math::min(eb0[k], eb1[k]);
        max_b = math::max(eb0[k], eb1[k]);
        if ((min_a - max_b) > dist || (min_b - max_a) > dist) {
            cd = false;
            break;
        }
    }
    return cd;
}

template <class T>
MU_FORCE_INLINE MU_HOST_DEV_FUNC bool edge_edge_ccd_broadphase(const T* ea0, const T* ea1, const T* eb0,
    const T* eb1, const T* dea0, const T* dea1, const T* deb0, const T* deb1, T dist)
{
    bool ccd = true;
    T max_a, min_a, max_b, min_b;
    for (int k = 0; k < 3; k++) {
        min_a = math::min(math::min(ea0[k], ea0[k] + dea0[k]), math::min(ea1[k], ea1[k] + dea1[k]));
        max_a = math::max(math::max(ea0[k], ea0[k] + dea0[k]), math::max(ea1[k], ea1[k] + dea1[k]));
        min_b = math::min(math::min(eb0[k], eb0[k] + deb0[k]), math::min(eb1[k], eb1[k] + deb1[k]));
        max_b = math::max(math::max(eb0[k], eb0[k] + deb0[k]), math::max(eb1[k], eb1[k] + deb1[k]));
        if ((min_a - max_b) > dist || (min_b - max_a) > dist) {
            ccd = false;
            break;
        }
    }
    return ccd;
}

template <class T>
MU_FORCE_INLINE MU_HOST_DEV_FUNC T edge_edge_ccd(const T* _ea0, const T* _ea1, const T* _eb0, const T* _eb1,
    const T* _dea0, const T* _dea1, const T* _deb0, const T* _deb1, T eta, T thickness)
{
    T ea0[3], ea1[3], eb0[3], eb1[3], dea0[3], dea1[3], deb0[3], deb1[3];
    T mov;
    for (int i = 0; i < 3; i++) {
        ea0[i] = _ea0[i];
        dea0[i] = _dea0[i];
        ea1[i] = _ea1[i];
        dea1[i] = _dea1[i];
        eb0[i] = _eb0[i];
        deb0[i] = _deb0[i];
        eb1[i] = _eb1[i];
        deb1[i] = _deb1[i];
        mov = (dea0[i] + dea1[i] + deb0[i] + deb1[i]) / 4;
        dea0[i] -= mov;
        dea1[i] -= mov;
        deb0[i] -= mov;
        deb1[i] -= mov;
    }

    T max_disp_mag = math::sqrt(math::max(math::norm_sqr<3>(dea0), math::norm_sqr<3>(dea1)))
        + math::sqrt(math::max(math::norm_sqr<3>(deb0), math::norm_sqr<3>(deb1)));
    if (max_disp_mag == 0)
        return 1.0;

    T dist2_cur = edge_edge_distance(ea0, ea1, eb0, eb1);

    T dFunc = dist2_cur - thickness * thickness;

    T dists[4], eb0a0[3], eb1a0[3], eb0a1[3], eb1a1[3];
    if (dFunc <= 0) {
        for (int i = 0; i < 3; i++) {
            eb0a0[i] = ea0[i] - eb0[i];
            eb1a0[i] = ea0[i] - eb1[i];
            eb0a1[i] = ea1[i] - eb0[i];
            eb1a1[i] = ea1[i] - eb1[i];
        }
        dists[0] = math::norm_sqr<3>(eb0a0);
        dists[1] = math::norm_sqr<3>(eb1a0);
        dists[2] = math::norm_sqr<3>(eb0a1);
        dists[3] = math::norm_sqr<3>(eb1a1);
        dist2_cur = math::min(math::min(dists[0], dists[1]), math::min(dists[2], dists[3]));
        dFunc = dist2_cur - thickness * thickness;
    }
    
    if (dFunc <= 0) return 0.0;

    T dist_cur = math::sqrt(dist2_cur);
    T gap = eta * dFunc / (dist_cur + thickness);
    T toc = 0.0;

    int iter = 0;
    while (true) {
        if (iter++ > 1000000) return toc;
        
        T toc_lower_bound = (1 - eta) * dFunc / ((dist_cur + thickness) * max_disp_mag);
        for (int i = 0; i < 3; i++) {
            ea0[i] += toc_lower_bound * dea0[i];
            ea1[i] += toc_lower_bound * dea1[i];
            eb0[i] += toc_lower_bound * deb0[i];
            eb1[i] += toc_lower_bound * deb1[i];
        }

        dist2_cur = edge_edge_distance(ea0, ea1, eb0, eb1);
        dFunc = dist2_cur - thickness * thickness;
        
        if (dFunc <= 0) {
            for (int i = 0; i < 3; i++) {
                eb0a0[i] = ea0[i] - eb0[i];
                eb1a0[i] = ea0[i] - eb1[i];
                eb0a1[i] = ea1[i] - eb0[i];
                eb1a1[i] = ea1[i] - eb1[i];
            }
            dists[0] = math::norm_sqr<3>(eb0a0);
            dists[1] = math::norm_sqr<3>(eb1a0);
            dists[2] = math::norm_sqr<3>(eb0a1);
            dists[3] = math::norm_sqr<3>(eb1a1);
            dist2_cur = math::min(math::min(dists[0], dists[1]), math::min(dists[2], dists[3]));
            dFunc = dist2_cur - thickness * thickness;
        }
        
        dist_cur = math::sqrt(dist2_cur);
        if (toc && (dFunc / (dist_cur + thickness) < gap)) {
            break;
        }
        toc += toc_lower_bound;
        if (toc > 1.0)
            return 1.0;
    }
    return toc;
}

}
