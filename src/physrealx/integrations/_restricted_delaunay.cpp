// Restricted Delaunay surface extraction with exact CGAL constructions.
// Input/output is a private numeric stream; no file paths or shell expressions.
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <CGAL/Delaunay_triangulation_cell_base_3.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_triangle_primitive.h>
#include <iostream>
#include <vector>
#include <array>

using K = CGAL::Exact_predicates_exact_constructions_kernel;
using VB = CGAL::Triangulation_vertex_base_with_info_3<std::size_t, K>;
using TDS = CGAL::Triangulation_data_structure_3<VB, CGAL::Delaunay_triangulation_cell_base_3<K>>;
using DT = CGAL::Delaunay_triangulation_3<K, TDS>;
using Triangles = std::vector<K::Triangle_3>;
using Primitive = CGAL::AABB_triangle_primitive<K, Triangles::const_iterator>;
using Tree = CGAL::AABB_tree<CGAL::AABB_traits<K, Primitive>>;

int main() {
    std::size_t nv, nf;
    if (!(std::cin >> nv >> nf) || nv < 4 || nf == 0) return 2;
    std::vector<std::pair<K::Point_3, std::size_t>> points;
    points.reserve(nv);
    for (std::size_t i = 0; i < nv; ++i) {
        double x, y, z;
        if (!(std::cin >> x >> y >> z)) return 2;
        points.emplace_back(K::Point_3(x, y, z), i);
    }
    Triangles triangles;
    triangles.reserve(nf);
    for (std::size_t i = 0; i < nf; ++i) {
        std::size_t a, b, c;
        if (!(std::cin >> a >> b >> c) || a >= nv || b >= nv || c >= nv) return 2;
        K::Triangle_3 triangle(points[a].first, points[b].first, points[c].first);
        if (!triangle.is_degenerate()) triangles.push_back(triangle);
    }
    if (triangles.empty()) return 2;
    Tree tree(triangles.cbegin(), triangles.cend());
    tree.build();
    DT dt(points.begin(), points.end());
    if (dt.dimension() != 3) return 3;
    std::vector<std::array<std::size_t, 3>> faces;
    for (auto f = dt.finite_facets_begin(); f != dt.finite_facets_end(); ++f) {
        auto dual = dt.dual(*f);
        bool hit = false;
        if (auto s = CGAL::object_cast<K::Segment_3>(&dual))
            hit = s->is_degenerate() ? tree.squared_distance(s->source()) == 0 : tree.do_intersect(*s);
        else if (auto r = CGAL::object_cast<K::Ray_3>(&dual)) hit = tree.do_intersect(*r);
        else if (auto l = CGAL::object_cast<K::Line_3>(&dual)) hit = tree.do_intersect(*l);
        if (hit) {
            std::array<std::size_t, 3> face;
            int n = 0;
            for (int j = 0; j < 4; ++j)
                if (j != f->second) face[n++] = f->first->vertex(j)->info();
            faces.push_back(face);
        }
    }
    std::cout << faces.size() << '\n';
    for (const auto& f : faces) std::cout << f[0] << ' ' << f[1] << ' ' << f[2] << '\n';
}
