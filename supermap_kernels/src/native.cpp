// Compiled kernels for SuperMap's semantic_mapping package.
//
// Each function reimplements one NumPy/SciPy routine of semantic_mapping
// (named in its docstring) with the same inputs, outputs and semantics; the
// Python code stays the reference implementation and the fallback when this
// module is not installed. Neighbour searches use a uniform grid instead of a
// KD-tree, run one query per point without temporaries, and release the GIL.
//
// Neighbour semantics follow scipy.spatial.cKDTree.query(k=..., distance_upper_bound=r):
// a point is a neighbour when its squared distance is strictly below r^2, and the
// k nearest are kept. cKDTree orders equal distances by its tree layout; here they
// are ordered by point index, so results are deterministic and independent of the
// thread count, and equal SciPy's wherever no tie decides which point is k-th.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

namespace {

using Index = std::int64_t;
using Points = py::array_t<double, py::array::c_style | py::array::forcecast>;
using Indices = py::array_t<Index, py::array::c_style | py::array::forcecast>;
using Labels = py::array_t<std::int32_t, py::array::c_style | py::array::forcecast>;
using Flags = py::array_t<bool, py::array::c_style | py::array::forcecast>;

int thread_count(int workers) {
#ifdef _OPENMP
  return workers > 0 ? workers : omp_get_max_threads();
#else
  (void)workers;
  return 1;
#endif
}

// Threads for `work` items: one below `min_work`. Waking a thread team for a
// small input costs more than it saves, and its spinning workers then compete
// with NumPy's BLAS threads for the cores (a per-detection crop, a VGA frame).
int thread_count(int workers, Index work, Index min_work) { return work < min_work ? 1 : thread_count(workers); }

constexpr Index kParallelQueries = 16384;       // neighbour searches, normals, edges, within
constexpr Index kParallelPixels = Index{1} << 19;  // image kernels (about 1024 x 512)
constexpr Index kParallelEdges = Index{1} << 20;   // edge remapping

Index rows_of(const Points& points, const char* name) {
  if (points.ndim() != 2 || points.shape(1) != 3) {
    throw std::invalid_argument(std::string(name) + " must have shape (N, 3)");
  }
  return points.shape(0);
}

void check_indices(const Indices& index, Index n) {
  if (index.ndim() != 1) throw std::invalid_argument("index must be one-dimensional");
  const Index* data = index.data();
  for (Index i = 0; i < index.shape(0); ++i) {
    if (data[i] < 0 || data[i] >= n) throw std::out_of_range("index out of range");
  }
}

// Every coordinate finite and small enough that its grid cell index fits in
// int64. Checked before any parallel region, where exceptions cannot propagate.
void check_coordinates(const double* p, Index n, double radius, const char* name) {
  const double limit = 1e18 * radius;
  for (Index i = 0; i < 3 * n; ++i) {
    if (!(std::fabs(p[i]) < limit)) throw std::invalid_argument(std::string(name) + " must be finite");
  }
}

// ----------------------------------------------------------------------------- grid

struct Cell {
  std::int64_t x, y, z;
  bool operator==(const Cell& other) const { return x == other.x && y == other.y && z == other.z; }
  bool operator<(const Cell& other) const {
    return x != other.x ? x < other.x : (y != other.y ? y < other.y : z < other.z);
  }
};

struct CellHash {
  std::size_t operator()(const Cell& c) const noexcept {
    std::uint64_t h = static_cast<std::uint64_t>(c.x) * 0x9E3779B97F4A7C15ull;
    h ^= static_cast<std::uint64_t>(c.y) + 0x7F4A7C159E3779B9ull + (h << 6) + (h >> 2);
    h ^= static_cast<std::uint64_t>(c.z) + 0x94D049BB133111EBull + (h << 6) + (h >> 2);
    h ^= h >> 31;
    h *= 0xBF58476D1CE4E5B9ull;
    h ^= h >> 27;
    return static_cast<std::size_t>(h);
  }
};

// Points bucketed into cubic cells slightly larger than the search radius, so
// every point within the radius of a query lies in the query's cell or one of
// its 26 neighbours. Points are stored cell by cell for sequential scans.
class Grid {
 public:
  Grid(const double* points, Index n, double radius) : cell_size_(radius * (1.0 + 1e-6)) {
    if (!(radius > 0) || !std::isfinite(radius)) throw std::invalid_argument("radius must be finite and positive");
    std::vector<std::pair<Cell, Index>> keyed(static_cast<std::size_t>(n));
    for (Index i = 0; i < n; ++i) keyed[i] = {cell_of(points + 3 * i), i};
    std::sort(keyed.begin(), keyed.end(), [](const auto& a, const auto& b) {
      return a.first == b.first ? a.second < b.second : a.first < b.first;
    });
    index_.resize(keyed.size());
    coords_.resize(3 * keyed.size());
    ranges_.reserve(keyed.size() / 4 + 1);
    std::size_t start = 0;
    for (std::size_t i = 0; i < keyed.size(); ++i) {
      const Index source = keyed[i].second;
      index_[i] = source;
      for (int d = 0; d < 3; ++d) coords_[3 * i + d] = points[3 * source + d];
      if (i + 1 == keyed.size() || !(keyed[i + 1].first == keyed[i].first)) {
        ranges_.emplace(keyed[i].first, std::make_pair(start, i + 1));
        start = i + 1;
      }
    }
  }

  Cell cell_of(const double* p) const {
    return {static_cast<std::int64_t>(std::floor(p[0] / cell_size_)),
            static_cast<std::int64_t>(std::floor(p[1] / cell_size_)),
            static_cast<std::int64_t>(std::floor(p[2] / cell_size_))};
  }

  // Storage ranges of the 27 cells around ``cell`` that hold points.
  void neighbourhood(const Cell& cell, std::vector<std::pair<std::size_t, std::size_t>>& out) const {
    out.clear();
    for (std::int64_t dx = -1; dx <= 1; ++dx)
      for (std::int64_t dy = -1; dy <= 1; ++dy)
        for (std::int64_t dz = -1; dz <= 1; ++dz) {
          auto it = ranges_.find({cell.x + dx, cell.y + dy, cell.z + dz});
          if (it != ranges_.end()) out.push_back(it->second);
        }
  }

  const double* coords(std::size_t slot) const { return coords_.data() + 3 * slot; }
  Index point_index(std::size_t slot) const { return index_[slot]; }

 private:
  double cell_size_;
  std::vector<Index> index_;
  std::vector<double> coords_;
  std::unordered_map<Cell, std::pair<std::size_t, std::size_t>, CellHash> ranges_;
};

inline double squared_distance(const double* a, const double* b) {
  const double dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
  return dx * dx + dy * dy + dz * dz;
}

using Neighbours = std::vector<std::pair<double, Index>>;

// Calls visit(i, neighbours) for every query position i in [0, m): the (at most)
// k points with d^2 < r2 around points[index[i]], ordered by (d^2, point index).
// Queries are grouped by grid cell so each neighbourhood is looked up once per
// cell; groups run in parallel, and visit must only write data owned by i.
template <class Visit>
void visit_nearest(const Grid& grid, const double* points, const Index* index, Index m, double r2, std::size_t k,
                   int threads, Visit visit) {
  std::vector<std::pair<Cell, Index>> queries(static_cast<std::size_t>(m));
  for (Index i = 0; i < m; ++i) queries[i] = {grid.cell_of(points + 3 * index[i]), i};
  std::sort(queries.begin(), queries.end(), [](const auto& a, const auto& b) {
    return a.first == b.first ? a.second < b.second : a.first < b.first;
  });
  std::vector<std::size_t> starts;
  for (std::size_t j = 0; j < queries.size(); ++j) {
    if (j == 0 || !(queries[j].first == queries[j - 1].first)) starts.push_back(j);
  }
  starts.push_back(queries.size());
  const Index groups = static_cast<Index>(starts.size()) - 1;
#pragma omp parallel num_threads(threads)
  {
    std::vector<std::pair<std::size_t, std::size_t>> ranges;
    Neighbours found;
#pragma omp for schedule(dynamic, 8)
    for (Index g = 0; g < groups; ++g) {
      grid.neighbourhood(queries[starts[g]].first, ranges);
      for (std::size_t j = starts[g]; j < starts[g + 1]; ++j) {
        const Index i = queries[j].second;
        const double* q = points + 3 * index[i];
        found.clear();
        for (const auto& range : ranges) {
          for (std::size_t slot = range.first; slot < range.second; ++slot) {
            const double d2 = squared_distance(grid.coords(slot), q);
            if (d2 < r2) found.emplace_back(d2, grid.point_index(slot));
          }
        }
        if (found.size() > k) {
          std::nth_element(found.begin(), found.begin() + static_cast<std::ptrdiff_t>(k), found.end());
          found.resize(k);
        }
        std::sort(found.begin(), found.end());
        visit(i, found);
      }
    }
  }
}

// ---------------------------------------------------------------- 3x3 symmetric eigen

// Eigenvalues (ascending) and unit eigenvectors (columns of v) of a symmetric
// 3x3 matrix by cyclic Jacobi rotations, accurate to machine precision.
void symmetric_eigen3(const double m[3][3], double w[3], double v[3][3]) {
  double a[3][3];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      a[i][j] = m[i][j];
      v[i][j] = i == j ? 1.0 : 0.0;
    }
  static const int pairs[3][2] = {{0, 1}, {0, 2}, {1, 2}};
  const double scale = std::fabs(a[0][0]) + std::fabs(a[1][1]) + std::fabs(a[2][2]) + std::fabs(a[0][1]) +
                       std::fabs(a[0][2]) + std::fabs(a[1][2]);
  for (int sweep = 0; sweep < 64; ++sweep) {
    // Rotations shrink the off-diagonal quadratically; stop at rounding level.
    const double off = std::fabs(a[0][1]) + std::fabs(a[0][2]) + std::fabs(a[1][2]);
    if (off <= 1e-15 * scale) break;
    for (const auto& pq : pairs) {
      const int p = pq[0], q = pq[1];
      const double apq = a[p][q];
      if (apq == 0.0) continue;
      const double diff = a[q][q] - a[p][p];
      if (sweep > 3 && std::fabs(apq) * 1e18 <= std::fabs(a[p][p]) && std::fabs(apq) * 1e18 <= std::fabs(a[q][q])) {
        a[p][q] = a[q][p] = 0.0;  // negligible next to both diagonal entries
        continue;
      }
      double t;
      if (std::fabs(apq) * 1e18 <= std::fabs(diff)) {
        t = apq / diff;
      } else {
        const double theta = 0.5 * diff / apq;
        t = 1.0 / (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
        if (theta < 0.0) t = -t;
      }
      const double c = 1.0 / std::sqrt(t * t + 1.0), s = t * c, tau = s / (1.0 + c);
      a[p][p] -= t * apq;
      a[q][q] += t * apq;
      a[p][q] = a[q][p] = 0.0;
      const int r = 3 - p - q;
      const double arp = a[r][p], arq = a[r][q];
      a[r][p] = a[p][r] = arp - s * (arq + tau * arp);
      a[r][q] = a[q][r] = arq + s * (arp - tau * arq);
      for (int i = 0; i < 3; ++i) {
        const double vip = v[i][p], viq = v[i][q];
        v[i][p] = vip - s * (viq + tau * vip);
        v[i][q] = viq + s * (vip - tau * viq);
      }
    }
  }
  int order[3] = {0, 1, 2};
  std::stable_sort(order, order + 3, [&](int x, int y) { return a[x][x] < a[y][y]; });
  double vs[3][3];
  for (int c = 0; c < 3; ++c) {
    w[c] = a[order[c]][order[c]];
    for (int i = 0; i < 3; ++i) vs[i][c] = v[i][order[c]];
  }
  for (int i = 0; i < 3; ++i)
    for (int c = 0; c < 3; ++c) v[i][c] = vs[i][c];
}

// --------------------------------------------------------------------- kernels

// PCA normal and planarity flag of one point from its neighbours (point
// indices, nearest first; neighbour(j) gives the j-th), exactly as
// semantic_mapping.dense_cloud._surface_normals computes them.
template <class Neighbour>
void normal_from(const double* p, std::size_t count, Neighbour neighbour, double* normal, bool* reliable) {
  const double denominator = static_cast<double>(std::max<std::size_t>(count, 1));
  double mean[3] = {0.0, 0.0, 0.0};
  for (std::size_t j = 0; j < count; ++j)
    for (int d = 0; d < 3; ++d) mean[d] += p[3 * neighbour(j) + d];
  for (int d = 0; d < 3; ++d) mean[d] /= denominator;
  double cov[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
  for (std::size_t j = 0; j < count; ++j) {
    double c[3];
    for (int d = 0; d < 3; ++d) c[d] = p[3 * neighbour(j) + d] - mean[d];
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) cov[a][b] += c[a] * c[b];
  }
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b) cov[a][b] /= denominator;
  double w[3], v[3][3];
  symmetric_eigen3(cov, w, v);
  for (int d = 0; d < 3; ++d) normal[d] = v[d][0];
  *reliable = count >= 3 && w[1] > 1e-10 && w[0] <= 0.1 * std::max(w[0] + w[1] + w[2], 1e-12);
}

// Whether the graph keeps edge a-b (semantic_mapping.dense_cloud._surface_edges):
// always unless both ends are planar, then only between aligned, coplanar normals.
inline bool keep_edge(const double* p, const double* nv, const bool* rel, Index a, Index b, double cos_angle,
                      double plane_tolerance) {
  if (!(rel[a] && rel[b])) return true;
  const double* na = nv + 3 * a;
  const double* nb = nv + 3 * b;
  const double delta[3] = {p[3 * b] - p[3 * a], p[3 * b + 1] - p[3 * a + 1], p[3 * b + 2] - p[3 * a + 2]};
  const bool aligned = std::fabs(na[0] * nb[0] + na[1] * nb[1] + na[2] * nb[2]) >= cos_angle;
  const bool flat_a = std::fabs(na[0] * delta[0] + na[1] * delta[1] + na[2] * delta[2]) <= plane_tolerance;
  const bool flat_b = std::fabs(nb[0] * delta[0] + nb[1] * delta[1] + nb[2] * delta[2]) <= plane_tolerance;
  return aligned && flat_a && flat_b;
}

void check_radius(double radius) {
  if (!(radius > 0) || !std::isfinite(radius)) throw std::invalid_argument("radius must be finite and positive");
}

// Concatenates per-query edge lists (query i: kept[i * k + j], j < counts[i]) into rows/cols, in query order.
py::tuple collect_edges(const Index* rows_of_query, Index m, std::size_t k, const std::vector<std::int32_t>& kept,
                        const std::vector<std::uint32_t>& counts) {
  std::size_t total = 0;
  for (auto c : counts) total += c;
  py::array_t<std::int32_t> rows(static_cast<Index>(total)), cols(static_cast<Index>(total));
  std::int32_t* out_r = rows.mutable_data();
  std::int32_t* out_c = cols.mutable_data();
  std::size_t e = 0;
  for (Index i = 0; i < m; ++i) {
    for (std::uint32_t j = 0; j < counts[i]; ++j, ++e) {
      out_r[e] = static_cast<std::int32_t>(rows_of_query[i]);
      out_c[e] = kept[static_cast<std::size_t>(i) * k + j];
    }
  }
  return py::make_tuple(rows, cols);
}

py::tuple surface_normals(Points points, Indices index, double radius, int max_neighbors, int workers) {
  const Index n = rows_of(points, "points");
  check_indices(index, n);
  if (max_neighbors < 1) throw std::invalid_argument("max_neighbors must be positive");
  check_radius(radius);
  const Index m = index.shape(0);
  py::array_t<double> normals({m, static_cast<Index>(3)});
  py::array_t<bool> reliable(m);
  if (m == 0) return py::make_tuple(normals, reliable);
  const double* p = points.data();
  const Index* idx = index.data();
  double* out_n = normals.mutable_data();
  bool* out_r = reliable.mutable_data();
  const std::size_t k = static_cast<std::size_t>(std::min<Index>(n, max_neighbors));
  check_coordinates(p, n, radius, "points");
  {
    py::gil_scoped_release release;
    const Grid grid(p, n, radius);
    visit_nearest(grid, p, idx, m, radius * radius, k, thread_count(workers, m, kParallelQueries), [&](Index i, const Neighbours& found) {
      normal_from(p, found.size(), [&](std::size_t j) { return found[j].second; }, out_n + 3 * i, out_r + i);
    });
  }
  return py::make_tuple(normals, reliable);
}

py::tuple surface_edges(Points points, Points normals, Flags reliable, Indices index, double radius, int max_neighbors,
                        double cos_angle, double plane_tolerance, int workers) {
  const Index n = rows_of(points, "points");
  if (rows_of(normals, "normals") != n || reliable.ndim() != 1 || reliable.shape(0) != n) {
    throw std::invalid_argument("normals and reliable must have one row per point");
  }
  if (n > std::numeric_limits<std::int32_t>::max()) throw std::invalid_argument("too many points for int32 edges");
  check_indices(index, n);
  if (max_neighbors < 1) throw std::invalid_argument("max_neighbors must be positive");
  check_radius(radius);
  const Index m = index.shape(0);
  const double* p = points.data();
  const double* nv = normals.data();
  const bool* rel = reliable.data();
  const Index* idx = index.data();
  const std::size_t k = static_cast<std::size_t>(std::min<Index>(n, max_neighbors));
  check_coordinates(p, n, radius, "points");
  std::vector<std::int32_t> kept(static_cast<std::size_t>(m) * k);
  std::vector<std::uint32_t> counts(static_cast<std::size_t>(m), 0);
  {
    py::gil_scoped_release release;
    if (m > 0) {
      const Grid grid(p, n, radius);
      visit_nearest(grid, p, idx, m, radius * radius, k, thread_count(workers, m, kParallelQueries), [&](Index i, const Neighbours& found) {
        const Index a = idx[i];
        std::uint32_t c = 0;
        for (const auto& f : found) {
          if (f.second != a && keep_edge(p, nv, rel, a, f.second, cos_angle, plane_tolerance)) {
            kept[static_cast<std::size_t>(i) * k + c++] = static_cast<std::int32_t>(f.second);
          }
        }
        counts[i] = c;
      });
    }
  }
  return collect_edges(idx, m, k, kept, counts);
}

// Normals of points[normal_index] and edges from points[edge_index] from one
// neighbour search: the k nearest within the larger radius, ordered by
// (distance, index), start with the k nearest within the smaller one, so each
// result equals surface_normals / surface_edges run separately. normals_in /
// reliable_in give every other point's normal (an incremental update keeps
// them); returns (normals, reliable, rows, cols) for all points.
py::tuple surface_graph(Points points, Indices normal_index, Indices edge_index, double normal_radius,
                        double neighbor_radius, int max_neighbors, double cos_angle, double plane_tolerance,
                        Points normals_in, Flags reliable_in, int workers) {
  const Index n = rows_of(points, "points");
  if (rows_of(normals_in, "normals") != n || reliable_in.ndim() != 1 || reliable_in.shape(0) != n) {
    throw std::invalid_argument("normals and reliable must have one row per point");
  }
  if (n > std::numeric_limits<std::int32_t>::max()) throw std::invalid_argument("too many points for int32 edges");
  check_indices(normal_index, n);
  check_indices(edge_index, n);
  if (max_neighbors < 1) throw std::invalid_argument("max_neighbors must be positive");
  check_radius(normal_radius);
  check_radius(neighbor_radius);
  const double radius = std::max(normal_radius, neighbor_radius);
  const double* p = points.data();
  check_coordinates(p, n, radius, "points");
  const Index mn = normal_index.shape(0), me = edge_index.shape(0);
  const Index* nidx = normal_index.data();
  const Index* eidx = edge_index.data();
  py::array_t<double> normals({n, static_cast<Index>(3)});
  py::array_t<bool> reliable(n);
  double* out_n = normals.mutable_data();
  bool* out_r = reliable.mutable_data();
  std::copy(normals_in.data(), normals_in.data() + 3 * n, out_n);
  std::copy(reliable_in.data(), reliable_in.data() + n, out_r);
  const std::size_t k = static_cast<std::size_t>(std::min<Index>(n, max_neighbors));
  std::vector<std::int32_t> kept(static_cast<std::size_t>(me) * k);
  std::vector<std::uint32_t> edge_counts(static_cast<std::size_t>(me), 0);
  {
    py::gil_scoped_release release;
    // One neighbour list (point indices, nearest first) per distinct query point.
    std::vector<Index> slot(static_cast<std::size_t>(n), -1), queries;
    for (const Index* list : {nidx, eidx}) {
      const Index count = list == nidx ? mn : me;
      for (Index i = 0; i < count; ++i) {
        if (slot[list[i]] < 0) {
          slot[list[i]] = static_cast<Index>(queries.size());
          queries.push_back(list[i]);
        }
      }
    }
    const Index q = static_cast<Index>(queries.size());
    std::vector<std::int32_t> lists(static_cast<std::size_t>(q) * k);
    std::vector<std::uint32_t> list_counts(static_cast<std::size_t>(q), 0);
    const int threads = thread_count(workers, q, kParallelQueries);
    if (q > 0) {
      const Grid grid(p, n, radius);
      visit_nearest(grid, p, queries.data(), q, radius * radius, k, threads, [&](Index i, const Neighbours& found) {
        for (std::size_t j = 0; j < found.size(); ++j) lists[static_cast<std::size_t>(i) * k + j] = static_cast<std::int32_t>(found[j].second);
        list_counts[i] = static_cast<std::uint32_t>(found.size());
      });
    }
    // The neighbours within r of a: the list's prefix with the same squared distances the search compared.
    const auto within = [&](Index a, double r) {
      const std::int32_t* list = lists.data() + static_cast<std::size_t>(slot[a]) * k;
      const double r2 = r * r;
      std::size_t c = 0;
      while (c < list_counts[slot[a]]) {
        const double* b = p + 3 * static_cast<Index>(list[c]);
        const double dx = b[0] - p[3 * a], dy = b[1] - p[3 * a + 1], dz = b[2] - p[3 * a + 2];
        if (!(dx * dx + dy * dy + dz * dz < r2)) break;
        ++c;
      }
      return std::make_pair(list, c);
    };
#pragma omp parallel for num_threads(threads) schedule(dynamic, 256)
    for (Index i = 0; i < mn; ++i) {
      const Index a = nidx[i];
      const auto [list, count] = within(a, normal_radius);
      normal_from(p, count, [&](std::size_t j) { return static_cast<Index>(list[j]); }, out_n + 3 * a, out_r + a);
    }
#pragma omp parallel for num_threads(threads) schedule(dynamic, 256)
    for (Index i = 0; i < me; ++i) {
      const Index a = eidx[i];
      const auto [list, count] = within(a, neighbor_radius);
      std::uint32_t c = 0;
      for (std::size_t j = 0; j < count; ++j) {
        const Index b = list[j];
        if (b != a && keep_edge(p, out_n, out_r, a, b, cos_angle, plane_tolerance)) {
          kept[static_cast<std::size_t>(i) * k + c++] = static_cast<std::int32_t>(b);
        }
      }
      edge_counts[i] = c;
    }
  }
  py::tuple edges = collect_edges(eidx, me, k, kept, edge_counts);
  return py::make_tuple(normals, reliable, edges[0], edges[1]);
}

// Edges renumbered through old_to_new, dropping those with a removed endpoint
// or a dirty (renumbered) row, in order, followed by fresh_rows / fresh_cols:
// the incremental update's edge list, written once into its final arrays.
py::tuple remap_edges(Labels rows, Labels cols, Labels old_to_new, Flags dirty, Labels fresh_rows, Labels fresh_cols,
                      int workers) {
  if (rows.ndim() != 1 || cols.ndim() != 1 || rows.shape(0) != cols.shape(0) || old_to_new.ndim() != 1 ||
      dirty.ndim() != 1 || fresh_rows.ndim() != 1 || fresh_cols.ndim() != 1 ||
      fresh_rows.shape(0) != fresh_cols.shape(0)) {
    throw std::invalid_argument("edge arrays must be one-dimensional, rows and cols equally long");
  }
  const Index e = rows.shape(0), old_n = old_to_new.shape(0), new_n = dirty.shape(0), fresh = fresh_rows.shape(0);
  const std::int32_t* r = rows.data();
  const std::int32_t* c = cols.data();
  const std::int32_t* map = old_to_new.data();
  const bool* drop = dirty.data();
  for (Index i = 0; i < old_n; ++i) {
    if (map[i] < -1 || map[i] >= new_n) throw std::out_of_range("old_to_new entry out of range");
  }
  const int threads = thread_count(workers, e, kParallelEdges);
  const Index chunk = 1 << 16;
  const Index chunks = (e + chunk - 1) / chunk;
  std::vector<Index> kept(static_cast<std::size_t>(chunks) + 1, 0);
  bool invalid = false;
  {
    py::gil_scoped_release release;
#pragma omp parallel for num_threads(threads) schedule(static) reduction(|| : invalid)
    for (Index b = 0; b < chunks; ++b) {
      Index count = 0;
      for (Index i = b * chunk; i < std::min(e, (b + 1) * chunk); ++i) {
        if (r[i] < 0 || r[i] >= old_n || c[i] < 0 || c[i] >= old_n) {
          invalid = true;
          continue;
        }
        const std::int32_t a = map[r[i]], d = map[c[i]];
        count += a >= 0 && d >= 0 && !drop[a];
      }
      kept[b + 1] = count;
    }
  }
  if (invalid) throw std::out_of_range("edge endpoint out of range");
  for (Index b = 0; b < chunks; ++b) kept[b + 1] += kept[b];
  const Index total = kept[chunks] + fresh;
  py::array_t<std::int32_t> new_rows(total), new_cols(total);
  std::int32_t* out_r = new_rows.mutable_data();
  std::int32_t* out_c = new_cols.mutable_data();
  {
    py::gil_scoped_release release;
#pragma omp parallel for num_threads(threads) schedule(static)
    for (Index b = 0; b < chunks; ++b) {
      Index o = kept[b];
      for (Index i = b * chunk; i < std::min(e, (b + 1) * chunk); ++i) {
        const std::int32_t a = map[r[i]], d = map[c[i]];
        if (a >= 0 && d >= 0 && !drop[a]) {
          out_r[o] = a;
          out_c[o] = d;
          ++o;
        }
      }
    }
    std::copy(fresh_rows.data(), fresh_rows.data() + fresh, out_r + kept[chunks]);
    std::copy(fresh_cols.data(), fresh_cols.data() + fresh, out_c + kept[chunks]);
  }
  return py::make_tuple(new_rows, new_cols);
}

Index find_root(std::vector<Index>& parent, Index x) {
  while (parent[x] != x) {
    parent[x] = parent[parent[x]];
    x = parent[x];
  }
  return x;
}

py::array_t<std::int32_t> connected_components(Index n, Labels rows, Labels cols) {
  if (n < 0 || n > std::numeric_limits<std::int32_t>::max()) throw std::invalid_argument("invalid node count");
  if (rows.ndim() != 1 || cols.ndim() != 1 || rows.shape(0) != cols.shape(0)) {
    throw std::invalid_argument("rows and cols must be one-dimensional and equally long");
  }
  const Index e = rows.shape(0);
  const std::int32_t* r = rows.data();
  const std::int32_t* c = cols.data();
  for (Index i = 0; i < e; ++i) {
    if (r[i] < 0 || r[i] >= n || c[i] < 0 || c[i] >= n) throw std::out_of_range("edge endpoint out of range");
  }
  py::array_t<std::int32_t> labels(n);
  std::int32_t* out = labels.mutable_data();
  {
    py::gil_scoped_release release;
    std::vector<Index> parent(static_cast<std::size_t>(n));
    for (Index i = 0; i < n; ++i) parent[i] = i;
    for (Index i = 0; i < e; ++i) {
      Index a = find_root(parent, r[i]), b = find_root(parent, c[i]);
      if (a != b) parent[std::max(a, b)] = std::min(a, b);  // the root is the smallest node
    }
    // Components numbered in order of their smallest node, as scipy.sparse.csgraph does.
    std::vector<std::int32_t> label_of_root(static_cast<std::size_t>(n), -1);
    std::int32_t next = 0;
    for (Index i = 0; i < n; ++i) {
      const Index root = find_root(parent, i);
      if (label_of_root[root] < 0) label_of_root[root] = next++;
      out[i] = label_of_root[root];
    }
  }
  return labels;
}

py::array_t<bool> within(Points points, Points seeds, double radius, int workers) {
  const Index n = rows_of(points, "points");
  const Index s = rows_of(seeds, "seeds");
  py::array_t<bool> result(n);
  bool* out = result.mutable_data();
  const double* p = points.data();
  const double r2 = radius * radius;
  if (!(radius > 0) || !std::isfinite(radius)) throw std::invalid_argument("radius must be finite and positive");
  check_coordinates(p, n, radius, "points");
  check_coordinates(seeds.data(), s, radius, "seeds");
  {
    py::gil_scoped_release release;
    if (s == 0) {
      std::fill(out, out + n, false);
    } else {
      const Grid grid(seeds.data(), s, radius);
      const int threads = thread_count(workers, n, kParallelQueries);
#pragma omp parallel num_threads(threads)
      {
        std::vector<std::pair<std::size_t, std::size_t>> ranges;
        Cell last{0, 0, 0};
        bool has_last = false;
#pragma omp for schedule(dynamic, 1024)
        for (Index i = 0; i < n; ++i) {
          const double* q = p + 3 * i;
          const Cell cell = grid.cell_of(q);
          if (!has_last || !(cell == last)) {
            grid.neighbourhood(cell, ranges);
            last = cell;
            has_last = true;
          }
          bool hit = false;
          for (const auto& range : ranges) {
            for (std::size_t slot = range.first; slot < range.second && !hit; ++slot) {
              hit = squared_distance(grid.coords(slot), q) < r2;
            }
            if (hit) break;
          }
          out[i] = hit;
        }
      }
    }
  }
  return result;
}

// semantic_mapping.geometry_utils.fit_ground_plane: every step but the least
// squares is reproduced operation for operation; the least-squares plane (NumPy:
// LAPACK lstsq) and the collinearity test (NumPy: eigvalsh of np.cov) agree to
// rounding. Returns (plane or None, fitted), or None for coordinates too large
// for integer cells (the caller then uses NumPy).
py::object fit_ground_plane(Points points_in, double tolerance_m, double max_slope, double cell_m, int iterations) {
  const Index n_in = rows_of(points_in, "points");
  const double* raw = points_in.data();
  std::vector<std::array<double, 3>> points;
  points.reserve(static_cast<std::size_t>(n_in));
  for (Index i = 0; i < n_in; ++i) {
    const double* p = raw + 3 * i;
    if (std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2])) points.push_back({p[0], p[1], p[2]});
  }
  const std::size_t n = points.size();
  if (n == 0) return py::make_tuple(py::none(), false);
  std::vector<std::int64_t> cx(n), cy(n);
  for (std::size_t i = 0; i < n; ++i) {
    const double fx = std::floor(points[i][0] / cell_m), fy = std::floor(points[i][1] / cell_m);
    if (!(std::fabs(fx) < 1e18) || !(std::fabs(fy) < 1e18)) return py::none();
    cx[i] = static_cast<std::int64_t>(fx);
    cy[i] = static_cast<std::int64_t>(fy);
  }
  const std::int64_t min_x = *std::min_element(cx.begin(), cx.end()), min_y = *std::min_element(cy.begin(), cy.end());
  std::int64_t max_y = 0;
  for (std::size_t i = 0; i < n; ++i) max_y = std::max(max_y, cy[i] - min_y);
  // The same integer cell key NumPy computes (wrapping like int64 arrays do).
  std::vector<std::int64_t> cell(n);
  for (std::size_t i = 0; i < n; ++i) {
    cell[i] = static_cast<std::int64_t>(static_cast<std::uint64_t>(cx[i] - min_x) * static_cast<std::uint64_t>(max_y + 1) +
                                        static_cast<std::uint64_t>(cy[i] - min_y));
  }
  // The first row of np.lexsort((z, cell)) per cell: its lowest return, the
  // earliest point on ties. Seeds are in ascending cell order, as NumPy's are.
  const auto lower = [&](std::size_t a, std::size_t b) {
    return points[a][2] < points[b][2] || (points[a][2] == points[b][2] && a < b);
  };
  std::vector<std::pair<std::int64_t, std::size_t>> lowest;
  std::int64_t max_x = 0;
  for (std::size_t i = 0; i < n; ++i) max_x = std::max(max_x, cx[i] - min_x);
  const double cells_spanned = (static_cast<double>(max_x) + 1) * (static_cast<double>(max_y) + 1);
  if (cells_spanned <= std::max<double>(4.0 * static_cast<double>(n), 4096.0)) {
    // Compact footprint (the usual case): one slot per cell, no sort of all points.
    std::vector<std::int64_t> best(static_cast<std::size_t>(cells_spanned), -1);
    for (std::size_t i = 0; i < n; ++i) {
      std::int64_t& slot = best[static_cast<std::size_t>(cell[i])];
      if (slot < 0 || lower(i, static_cast<std::size_t>(slot))) slot = static_cast<std::int64_t>(i);
    }
    for (std::size_t c = 0; c < best.size(); ++c) {
      if (best[c] >= 0) lowest.emplace_back(static_cast<std::int64_t>(c), static_cast<std::size_t>(best[c]));
    }
  } else {
    std::unordered_map<std::int64_t, std::size_t> best;
    for (std::size_t i = 0; i < n; ++i) {
      auto it = best.emplace(cell[i], i).first;
      if (lower(i, it->second)) it->second = i;
    }
    lowest.assign(best.begin(), best.end());
    std::sort(lowest.begin(), lowest.end());
  }
  std::vector<std::array<double, 3>> seeds;
  seeds.reserve(lowest.size());
  for (const auto& entry : lowest) seeds.push_back(points[entry.second]);
  const std::size_t s = seeds.size();
  std::vector<std::size_t> by_height(s);
  for (std::size_t i = 0; i < s; ++i) by_height[i] = i;
  std::stable_sort(by_height.begin(), by_height.end(), [&](std::size_t a, std::size_t b) { return seeds[a][2] < seeds[b][2]; });
  const std::array<double, 3> anchor = seeds[by_height[static_cast<std::size_t>(0.1 * static_cast<double>(s - 1))]];
  py::array_t<double> level(3);
  level.mutable_data()[0] = 0.0;
  level.mutable_data()[1] = 0.0;
  level.mutable_data()[2] = anchor[2];
  std::vector<char> inliers(s);
  for (std::size_t i = 0; i < s; ++i) {
    const double dx = seeds[i][0] - anchor[0], dy = seeds[i][1] - anchor[1];
    const double reach = tolerance_m + max_slope * std::sqrt(dx * dx + dy * dy);
    inliers[i] = std::fabs(seeds[i][2] - anchor[2]) <= reach;
  }
  const double min_spread = (cell_m / 2) * (cell_m / 2);
  double plane[3] = {0.0, 0.0, anchor[2]};
  for (int iteration = 0; iteration < iterations; ++iteration) {
    std::size_t m = 0;
    double mx = 0, my = 0, mz = 0;
    for (std::size_t i = 0; i < s; ++i) {
      if (!inliers[i]) continue;
      ++m;
      mx += seeds[i][0];
      my += seeds[i][1];
      mz += seeds[i][2];
    }
    if (m < 3) return py::make_tuple(level, false);
    mx /= static_cast<double>(m);
    my /= static_cast<double>(m);
    mz /= static_cast<double>(m);
    double sxx = 0, sxy = 0, syy = 0, sxz = 0, syz = 0;
    for (std::size_t i = 0; i < s; ++i) {
      if (!inliers[i]) continue;
      const double dx = seeds[i][0] - mx, dy = seeds[i][1] - my, dz = seeds[i][2] - mz;
      sxx += dx * dx;
      sxy += dx * dy;
      syy += dy * dy;
      sxz += dx * dz;
      syz += dy * dz;
    }
    // Smaller eigenvalue of the sample covariance (np.cov, N - 1): too thin a spread is no plane.
    const double scale = 1.0 / static_cast<double>(m - 1);
    const double cxx = sxx * scale, cyy = syy * scale, cxy = sxy * scale;
    const double half_gap = (cxx - cyy) / 2;
    const double smallest = (cxx + cyy) / 2 - std::sqrt(half_gap * half_gap + cxy * cxy);
    if (smallest < min_spread) return py::make_tuple(level, false);
    const double det = sxx * syy - sxy * sxy;
    plane[0] = (syy * sxz - sxy * syz) / det;
    plane[1] = (sxx * syz - sxy * sxz) / det;
    plane[2] = mz - plane[0] * mx - plane[1] * my;
    if (std::hypot(plane[0], plane[1]) > max_slope) return py::make_tuple(level, false);
    for (std::size_t i = 0; i < s; ++i) {
      const double fitted = seeds[i][0] * plane[0] + seeds[i][1] * plane[1];
      inliers[i] = std::fabs(seeds[i][2] - fitted - plane[2]) <= tolerance_m;
    }
  }
  py::array_t<double> result(3);
  std::copy(plane, plane + 3, result.mutable_data());
  return py::make_tuple(result, true);
}

// np.minimum semantics: NaN in either operand gives NaN.
inline double nan_min(double a, double b) {
  if (std::isnan(a) || std::isnan(b)) return std::numeric_limits<double>::quiet_NaN();
  return b < a ? b : a;
}

void check_pixels(const Index* u, const Index* v, Index n, Index width, Index height) {
  for (Index i = 0; i < n; ++i) {
    if (u[i] < 0 || u[i] >= width || v[i] < 0 || v[i] >= height) throw std::out_of_range("pixel outside the image");
  }
}

// One bit per pixel: which pixels hold a point. Lets a per-pixel buffer stay
// uninitialized (and its memory untouched) except where points land, which at
// 5 MP and a LiDAR's tens of thousands of points is most of the cost otherwise.
class Occupancy {
 public:
  explicit Occupancy(std::size_t pixels) : bits_((pixels + 63) / 64, 0) {}
  bool test(std::size_t p) const { return (bits_[p >> 6] >> (p & 63)) & 1u; }
  // Marks p; returns whether it was already marked.
  bool mark(std::size_t p) {
    const std::uint64_t bit = std::uint64_t{1} << (p & 63);
    const bool was = bits_[p >> 6] & bit;
    bits_[p >> 6] |= bit;
    return was;
  }

 private:
  std::vector<std::uint64_t> bits_;
};

// np.minimum.at(values, v * width + u, z) on a +inf buffer, stored sparsely:
// ``values[p]`` is meaningful only where ``occupied`` is marked (+inf elsewhere).
void scatter_nearest(double* values, Occupancy& occupied, const Index* u, const Index* v, const double* z, Index n,
                     Index width) {
  for (Index i = 0; i < n; ++i) {
    const std::size_t p = static_cast<std::size_t>(v[i] * width + u[i]);
    values[p] = occupied.mark(p) ? nan_min(values[p], z[i]) : nan_min(std::numeric_limits<double>::infinity(), z[i]);
  }
}

void splat_into(double* out, const Index* u, const Index* v, const double* z, Index count, double focal, Index width,
                Index height, double radius_m, Index max_px) {
  std::fill(out, out + width * height, std::numeric_limits<double>::infinity());
  if (!(radius_m > 0) || count == 0) return;
  const double scale = focal * radius_m;
  for (Index i = 0; i < count; ++i) {
    // Footprint radius exactly as NumPy computes it; a NaN radius never writes.
    const double r = std::min(std::floor(scale / z[i] + 0.5), static_cast<double>(max_px));
    if (std::isnan(r) || r < 0) continue;
    const Index radius = static_cast<Index>(r);
    const Index x0 = std::max<Index>(u[i] - radius, 0), x1 = std::min<Index>(u[i] + radius, width - 1);
    const Index y0 = std::max<Index>(v[i] - radius, 0), y1 = std::min<Index>(v[i] + radius, height - 1);
    const double depth = z[i];
    for (Index y = y0; y <= y1; ++y) {
      double* row = out + y * width;
      for (Index x = x0; x <= x1; ++x) row[x] = std::min(row[x], depth);
    }
  }
}

// semantic_mapping.geometry_utils.occlusion_visible, step for step.
py::array_t<bool> occlusion_visible(Indices us, Indices vs, Points z_in, double focal, Index width, Index height,
                                    double radius_m, Index max_px, double gap_m, Index grid_px, bool keep_dense_surfaces) {
  if (us.ndim() != 1 || vs.ndim() != 1 || us.shape(0) != vs.shape(0) || z_in.size() != us.shape(0)) {
    throw std::invalid_argument("us, vs and z must be one-dimensional and equally long");
  }
  const Index n = us.shape(0);
  py::array_t<bool> visible(n);
  bool* out = visible.mutable_data();
  if (!(radius_m > 0) || n == 0) {
    std::fill(out, out + n, true);
    return visible;
  }
  const Index* u = us.data();
  const Index* v = vs.data();
  const double* z = z_in.data();
  check_pixels(u, v, n, width, height);
  {
    py::gil_scoped_release release;
    std::vector<char> dense(static_cast<std::size_t>(n), 0);
    if (keep_dense_surfaces) {
      const std::size_t pixels = static_cast<std::size_t>(width * height);
      std::unique_ptr<double[]> nearest(new double[pixels]);  // uninitialized: only occupied pixels are read
      Occupancy occupied(pixels);
      scatter_nearest(nearest.get(), occupied, u, v, z, n, width);
      const auto found = [&](Index i, Index du, Index dv) {
        const Index uu = u[i] + du, vv = v[i] + dv;
        if (uu < 0 || uu >= width || vv < 0 || vv >= height) return false;
        const std::size_t p = static_cast<std::size_t>(vv * width + uu);
        return occupied.test(p) && std::fabs(nearest[p] - z[i]) <= gap_m;  // |inf - z| <= gap never holds
      };
      for (Index i = 0; i < n; ++i) {
        dense[i] = (found(i, -1, 0) || found(i, 1, 0)) && (found(i, 0, -1) || found(i, 0, 1));
      }
    }
    const Index grid = std::max<Index>(grid_px, 1);
    Index w = width, h = height, cap = max_px;
    double f = focal;
    std::vector<Index> gu, gv;
    const Index *su = u, *sv = v;
    if (grid > 1) {
      gu.resize(static_cast<std::size_t>(n));
      gv.resize(static_cast<std::size_t>(n));
      for (Index i = 0; i < n; ++i) {
        gu[i] = u[i] / grid;  // non-negative: floor division
        gv[i] = v[i] / grid;
      }
      su = gu.data();
      sv = gv.data();
      w = (width + grid - 1) / grid;
      h = (height + grid - 1) / grid;
      f = focal / static_cast<double>(grid);
      cap = std::max<Index>((max_px + grid - 1) / grid, 1);
    }
    std::vector<double> buffer(static_cast<std::size_t>(w * h));
    splat_into(buffer.data(), su, sv, z, n, f, w, h, radius_m, cap);
    for (Index i = 0; i < n; ++i) {
      out[i] = dense[i] || z[i] <= buffer[static_cast<std::size_t>(sv[i] * w + su[i])] + gap_m;
    }
  }
  return visible;
}

// The z-buffer image of rasterize_depth: nearest depth per pixel, 0 where no point (or not finite).
py::array_t<double> depth_image(Indices us, Indices vs, Points z_in, Index width, Index height) {
  if (us.ndim() != 1 || vs.ndim() != 1 || us.shape(0) != vs.shape(0) || z_in.size() != us.shape(0)) {
    throw std::invalid_argument("us, vs and z must be one-dimensional and equally long");
  }
  const Index n = us.shape(0);
  const Index* u = us.data();
  const Index* v = vs.data();
  check_pixels(u, v, n, width, height);
  // numpy.zeros: the OS provides zero pages lazily, so pixels without a point cost nothing.
  py::array_t<double> image = py::module_::import("numpy").attr("zeros")(py::make_tuple(height, width));
  double* out = image.mutable_data();
  {
    py::gil_scoped_release release;
    Occupancy occupied(static_cast<std::size_t>(width * height));
    scatter_nearest(out, occupied, u, v, z_in.data(), n, width);
    for (Index i = 0; i < n; ++i) {
      double& value = out[static_cast<std::size_t>(v[i] * width + u[i])];
      if (!std::isfinite(value)) value = 0.0;
    }
  }
  return image;
}

// semantic_mapping.geometry_utils.fill_sparse_depth: readings (finite, > 0) are kept;
// every other pixel takes the minimum reading within radius_px (a square window,
// clipped at the border like mode "nearest"), or 0 when there is none. Bands of
// rows run in parallel, each keeping the horizontal minima of its rows (plus a
// radius of rows on each side) in a small buffer rather than a full-size image.
py::array_t<double> fill_sparse_depth(Points depth_in, Index radius_px, int workers) {
  if (depth_in.ndim() != 2) throw std::invalid_argument("depth must be two-dimensional");
  if (radius_px < 0) throw std::invalid_argument("radius_px must be non-negative");
  const Index h = depth_in.shape(0), w = depth_in.shape(1);
  const double* depth = depth_in.data();
  py::array_t<double> filled({h, w});
  double* out = filled.mutable_data();
  {
    py::gil_scoped_release release;
    const double inf = std::numeric_limits<double>::infinity();
    const Index band = 64;
    const Index bands = (h + band - 1) / band;
    const int threads = thread_count(workers, h * w, kParallelPixels);
#pragma omp parallel num_threads(threads)
    {
      std::vector<double> minima, candidates(static_cast<std::size_t>(w));
#pragma omp for schedule(dynamic, 1)
      for (Index b = 0; b < bands; ++b) {
        const Index y_begin = b * band, y_end = std::min(y_begin + band, h);
        const Index r0 = std::max<Index>(y_begin - radius_px, 0), r1 = std::min<Index>(y_end - 1 + radius_px, h - 1);
        minima.resize(static_cast<std::size_t>((r1 - r0 + 1) * w));
        // Horizontal minimum of the candidates (a reading, else +inf): the
        // running minimum of the row shifted by 1..radius each way, in
        // branch-free loops the compiler vectorizes.
        const Index reach = std::min<Index>(radius_px, w - 1);
        for (Index y = r0; y <= r1; ++y) {
          const double* src = depth + y * w;
          double* c = candidates.data();
          for (Index x = 0; x < w; ++x) c[x] = (std::isfinite(src[x]) && src[x] > 0) ? src[x] : inf;
          double* dst = minima.data() + (y - r0) * w;
          std::copy(c, c + w, dst);
          for (Index o = 1; o <= reach; ++o) {
            for (Index x = 0; x + o < w; ++x) dst[x] = std::min(dst[x], c[x + o]);
            for (Index x = o; x < w; ++x) dst[x] = std::min(dst[x], c[x - o]);
          }
        }
        // Vertical minimum over the rows within the radius, then keep readings as they are.
        for (Index y = y_begin; y < y_end; ++y) {
          const Index k0 = std::max<Index>(y - radius_px, 0), k1 = std::min<Index>(y + radius_px, h - 1);
          double* row = out + y * w;
          std::copy(minima.data() + (k0 - r0) * w, minima.data() + (k0 - r0 + 1) * w, row);
          for (Index k = k0 + 1; k <= k1; ++k) {
            const double* m = minima.data() + (k - r0) * w;
            for (Index x = 0; x < w; ++x) row[x] = std::min(row[x], m[x]);
          }
          const double* src = depth + y * w;
          for (Index x = 0; x < w; ++x) {
            const double d = src[x];
            row[x] = (std::isfinite(d) && d > 0) ? d : (std::isfinite(row[x]) ? row[x] : 0.0);
          }
        }
      }
    }
  }
  return filled;
}

py::array_t<double> splat_depth_buffer(Indices us, Indices vs, Points z_in, double focal, Index width, Index height,
                                       double radius_m, Index max_px) {
  if (us.ndim() != 1 || vs.ndim() != 1 || us.shape(0) != vs.shape(0)) {
    throw std::invalid_argument("us and vs must be one-dimensional and equally long");
  }
  if (width < 0 || height < 0) throw std::invalid_argument("image size must be non-negative");
  const Index count = us.shape(0);
  if (z_in.size() != count) throw std::invalid_argument("z must hold one depth per point");
  py::array_t<double> buffer(width * height);
  double* out = buffer.mutable_data();
  const Index* u = us.data();
  const Index* v = vs.data();
  const double* z = z_in.data();
  {
    py::gil_scoped_release release;
    splat_into(out, u, v, z, count, focal, width, height, radius_m, max_px);
  }
  return buffer;
}

}  // namespace

PYBIND11_MODULE(_native, m) {
  m.doc() = "Compiled kernels for semantic_mapping; the NumPy/SciPy implementations are the reference.";
  m.def("surface_normals", &surface_normals, py::arg("points"), py::arg("index"), py::arg("radius"),
        py::arg("max_neighbors"), py::arg("workers") = -1,
        "PCA normals and planarity flags of points[index] (semantic_mapping.dense_cloud._surface_normals).");
  m.def("surface_edges", &surface_edges, py::arg("points"), py::arg("normals"), py::arg("reliable"), py::arg("index"),
        py::arg("radius"), py::arg("max_neighbors"), py::arg("cos_angle"), py::arg("plane_tolerance"),
        py::arg("workers") = -1,
        "Kept smooth-surface edges from points[index] (semantic_mapping.dense_cloud._surface_edges).");
  m.def("surface_graph", &surface_graph, py::arg("points"), py::arg("normal_index"), py::arg("edge_index"),
        py::arg("normal_radius"), py::arg("neighbor_radius"), py::arg("max_neighbors"), py::arg("cos_angle"),
        py::arg("plane_tolerance"), py::arg("normals"), py::arg("reliable"), py::arg("workers") = -1,
        "Normals of points[normal_index] and edges from points[edge_index] from one neighbour search: "
        "(normals, reliable, rows, cols).");
  m.def("remap_edges", &remap_edges, py::arg("rows"), py::arg("cols"), py::arg("old_to_new"), py::arg("dirty"),
        py::arg("fresh_rows"), py::arg("fresh_cols"), py::arg("workers") = -1,
        "Edges renumbered through old_to_new, without removed endpoints or dirty rows, in order, then the fresh "
        "edges (semantic_mapping.dense_cloud.update_surface_graph).");
  m.def("connected_components", &connected_components, py::arg("n"), py::arg("rows"), py::arg("cols"),
        "Undirected component label per node, numbered by smallest node (scipy.sparse.csgraph.connected_components).");
  m.def("within", &within, py::arg("points"), py::arg("seeds"), py::arg("radius"), py::arg("workers") = -1,
        "Whether each point has a seed closer than radius (semantic_mapping.dense_cloud._within).");
  m.def("fit_ground_plane", &fit_ground_plane, py::arg("points"), py::arg("tolerance_m"), py::arg("max_slope"),
        py::arg("cell_m"), py::arg("iterations"),
        "Local ground plane under world points (semantic_mapping.geometry_utils.fit_ground_plane): (plane, fitted).");
  m.def("occlusion_visible", &occlusion_visible, py::arg("us"), py::arg("vs"), py::arg("z"), py::arg("focal"),
        py::arg("width"), py::arg("height"), py::arg("radius_m"), py::arg("max_px"), py::arg("gap_m"),
        py::arg("grid_px"), py::arg("keep_dense_surfaces"),
        "Points no nearer footprint hides (semantic_mapping.geometry_utils.occlusion_visible).");
  m.def("depth_image", &depth_image, py::arg("us"), py::arg("vs"), py::arg("z"), py::arg("width"), py::arg("height"),
        "(H, W) nearest depth per pixel, 0 where none (semantic_mapping.geometry_utils.rasterize_depth).");
  m.def("fill_sparse_depth", &fill_sparse_depth, py::arg("depth"), py::arg("radius_px"), py::arg("workers") = -1,
        "Empty pixels take the minimum reading within radius_px (semantic_mapping.geometry_utils.fill_sparse_depth).");
  m.def("splat_depth_buffer", &splat_depth_buffer, py::arg("us"), py::arg("vs"), py::arg("z"), py::arg("focal"),
        py::arg("width"), py::arg("height"), py::arg("radius_m"), py::arg("max_px"),
        "Per-pixel nearest depth of square point footprints (semantic_mapping.geometry_utils.splat_depth_buffer).");
  m.attr("__version__") = "0.1.0";
  // Bumped whenever a signature or a result changes (semantic_mapping/native.py checks it).
  m.attr("API_VERSION") = 1;
}
