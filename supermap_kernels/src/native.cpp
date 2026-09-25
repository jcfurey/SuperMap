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
#include <cmath>
#include <cstdint>
#include <limits>
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

py::tuple surface_normals(Points points, Indices index, double radius, int max_neighbors, int workers) {
  const Index n = rows_of(points, "points");
  check_indices(index, n);
  if (max_neighbors < 1) throw std::invalid_argument("max_neighbors must be positive");
  const Index m = index.shape(0);
  py::array_t<double> normals({m, static_cast<Index>(3)});
  py::array_t<bool> reliable(m);
  if (m == 0) return py::make_tuple(normals, reliable);
  const double* p = points.data();
  const Index* idx = index.data();
  double* out_n = normals.mutable_data();
  bool* out_r = reliable.mutable_data();
  const std::size_t k = static_cast<std::size_t>(std::min<Index>(n, max_neighbors));
  const double r2 = radius * radius;
  if (!(radius > 0) || !std::isfinite(radius)) throw std::invalid_argument("radius must be finite and positive");
  check_coordinates(p, n, radius, "points");
  {
    py::gil_scoped_release release;
    const Grid grid(p, n, radius);
    visit_nearest(grid, p, idx, m, r2, k, thread_count(workers), [&](Index i, const Neighbours& found) {
        const double count = static_cast<double>(std::max<std::size_t>(found.size(), 1));
        double mean[3] = {0.0, 0.0, 0.0};
        for (const auto& f : found)
          for (int d = 0; d < 3; ++d) mean[d] += p[3 * f.second + d];
        for (int d = 0; d < 3; ++d) mean[d] /= count;
        double cov[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
        for (const auto& f : found) {
          double c[3];
          for (int d = 0; d < 3; ++d) c[d] = p[3 * f.second + d] - mean[d];
          for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b) cov[a][b] += c[a] * c[b];
        }
        for (int a = 0; a < 3; ++a)
          for (int b = 0; b < 3; ++b) cov[a][b] /= count;
        double w[3], v[3][3];
        symmetric_eigen3(cov, w, v);
        for (int d = 0; d < 3; ++d) out_n[3 * i + d] = v[d][0];
        out_r[i] = found.size() >= 3 && w[1] > 1e-10 && w[0] <= 0.1 * std::max(w[0] + w[1] + w[2], 1e-12);
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
  const Index m = index.shape(0);
  const double* p = points.data();
  const double* nv = normals.data();
  const bool* rel = reliable.data();
  const Index* idx = index.data();
  const std::size_t k = static_cast<std::size_t>(std::min<Index>(n, max_neighbors));
  const double r2 = radius * radius;
  if (!(radius > 0) || !std::isfinite(radius)) throw std::invalid_argument("radius must be finite and positive");
  check_coordinates(p, n, radius, "points");
  std::vector<std::int32_t> kept(static_cast<std::size_t>(m) * k);
  std::vector<std::uint32_t> counts(static_cast<std::size_t>(m), 0);
  {
    py::gil_scoped_release release;
    if (m > 0) {
      const Grid grid(p, n, radius);
      visit_nearest(grid, p, idx, m, r2, k, thread_count(workers), [&](Index i, const Neighbours& found) {
          const Index a = idx[i];
          std::uint32_t c = 0;
          for (const auto& f : found) {
            const Index b = f.second;
            if (b == a) continue;
            bool keep = true;
            if (rel[a] && rel[b]) {
              const double* na = nv + 3 * a;
              const double* nb = nv + 3 * b;
              const double delta[3] = {p[3 * b] - p[3 * a], p[3 * b + 1] - p[3 * a + 1], p[3 * b + 2] - p[3 * a + 2]};
              const bool aligned = std::fabs(na[0] * nb[0] + na[1] * nb[1] + na[2] * nb[2]) >= cos_angle;
              const bool flat_a = std::fabs(na[0] * delta[0] + na[1] * delta[1] + na[2] * delta[2]) <= plane_tolerance;
              const bool flat_b = std::fabs(nb[0] * delta[0] + nb[1] * delta[1] + nb[2] * delta[2]) <= plane_tolerance;
              keep = aligned && flat_a && flat_b;
            }
            if (keep) kept[static_cast<std::size_t>(i) * k + c++] = static_cast<std::int32_t>(b);
          }
          counts[i] = c;
      });
    }
  }
  std::size_t total = 0;
  for (auto c : counts) total += c;
  py::array_t<std::int32_t> rows(static_cast<Index>(total)), cols(static_cast<Index>(total));
  std::int32_t* out_r = rows.mutable_data();
  std::int32_t* out_c = cols.mutable_data();
  std::size_t e = 0;
  for (Index i = 0; i < m; ++i) {
    for (std::uint32_t j = 0; j < counts[i]; ++j, ++e) {
      out_r[e] = static_cast<std::int32_t>(idx[i]);
      out_c[e] = kept[static_cast<std::size_t>(i) * k + j];
    }
  }
  return py::make_tuple(rows, cols);
}

py::tuple remap_edges(Labels rows, Labels cols, Labels old_to_new, Flags dirty) {
  if (rows.ndim() != 1 || cols.ndim() != 1 || rows.shape(0) != cols.shape(0) || old_to_new.ndim() != 1 ||
      dirty.ndim() != 1) {
    throw std::invalid_argument("rows, cols, old_to_new and dirty must be one-dimensional; rows and cols equally long");
  }
  const Index e = rows.shape(0), old_n = old_to_new.shape(0), new_n = dirty.shape(0);
  const std::int32_t* r = rows.data();
  const std::int32_t* c = cols.data();
  const std::int32_t* map = old_to_new.data();
  const bool* drop = dirty.data();
  std::vector<std::int32_t> out_rows, out_cols;
  bool invalid = false;
  {
    py::gil_scoped_release release;
    out_rows.reserve(static_cast<std::size_t>(e));
    out_cols.reserve(static_cast<std::size_t>(e));
    for (Index i = 0; i < e && !invalid; ++i) {
      if (r[i] < 0 || r[i] >= old_n || c[i] < 0 || c[i] >= old_n) {
        invalid = true;
        break;
      }
      const std::int32_t a = map[r[i]], b = map[c[i]];
      if (a < -1 || a >= new_n || b < -1 || b >= new_n) {
        invalid = true;
        break;
      }
      if (a >= 0 && b >= 0 && !drop[a]) {
        out_rows.push_back(a);
        out_cols.push_back(b);
      }
    }
  }
  if (invalid) throw std::out_of_range("edge endpoint or remapped index out of range");
  py::array_t<std::int32_t> new_rows(static_cast<Index>(out_rows.size())), new_cols(static_cast<Index>(out_cols.size()));
  std::copy(out_rows.begin(), out_rows.end(), new_rows.mutable_data());
  std::copy(out_cols.begin(), out_cols.end(), new_cols.mutable_data());
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
      const int threads = thread_count(workers);
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
    std::fill(out, out + width * height, std::numeric_limits<double>::infinity());
    if (radius_m > 0 && count > 0) {
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
  m.def("remap_edges", &remap_edges, py::arg("rows"), py::arg("cols"), py::arg("old_to_new"), py::arg("dirty"),
        "Edges renumbered through old_to_new, without removed endpoints or dirty rows, in order "
        "(semantic_mapping.dense_cloud.update_surface_graph).");
  m.def("connected_components", &connected_components, py::arg("n"), py::arg("rows"), py::arg("cols"),
        "Undirected component label per node, numbered by smallest node (scipy.sparse.csgraph.connected_components).");
  m.def("within", &within, py::arg("points"), py::arg("seeds"), py::arg("radius"), py::arg("workers") = -1,
        "Whether each point has a seed closer than radius (semantic_mapping.dense_cloud._within).");
  m.def("splat_depth_buffer", &splat_depth_buffer, py::arg("us"), py::arg("vs"), py::arg("z"), py::arg("focal"),
        py::arg("width"), py::arg("height"), py::arg("radius_m"), py::arg("max_px"),
        "Per-pixel nearest depth of square point footprints (semantic_mapping.geometry_utils.splat_depth_buffer).");
  m.attr("__version__") = "0.1.0";
  // Bumped whenever a signature or a result changes (semantic_mapping/native.py checks it).
  m.attr("API_VERSION") = 1;
}
