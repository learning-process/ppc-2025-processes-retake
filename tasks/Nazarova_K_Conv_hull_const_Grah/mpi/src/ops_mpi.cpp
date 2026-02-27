#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "Nazarova_K_Conv_hull_const_Grah/common/include/common.hpp"
#include "Nazarova_K_Conv_hull_const_Grah/mpi/include/ops_mpi.hpp"

namespace nazarova_k_conv_hull_const_grah_processes {
namespace {

inline bool LessPivot(const Point& a, const Point& b) {
  if (a.y != b.y) { return a.y < b.y;
}
  return a.x < b.x;
}

inline bool LessXY(const Point& a, const Point& b) {
  if (a.x != b.x) { return a.x < b.x;
}
  return a.y < b.y;
}

inline std::int64_t Cross(const Point& o, const Point& a, const Point& b) {
  const std::int64_t ax = static_cast<std::int64_t>(a.x) - static_cast<std::int64_t>(o.x);
  const std::int64_t ay = static_cast<std::int64_t>(a.y) - static_cast<std::int64_t>(o.y);
  const std::int64_t bx = static_cast<std::int64_t>(b.x) - static_cast<std::int64_t>(o.x);
  const std::int64_t by = static_cast<std::int64_t>(b.y) - static_cast<std::int64_t>(o.y);
  return (ax * by) - (ay * bx);
}

inline std::int64_t Dist2(const Point& a, const Point& b) {
  const std::int64_t dx = static_cast<std::int64_t>(a.x) - static_cast<std::int64_t>(b.x);
  const std::int64_t dy = static_cast<std::int64_t>(a.y) - static_cast<std::int64_t>(b.y);
  return (dx * dx) + (dy * dy);
}

std::vector<Point> GrahamScan(std::vector<Point>& pts) {
  std::ranges::sort(pts, LessXY);
  pts.erase(std::ranges::unique(pts), pts.end());

  const std::size_t n = pts.size();
  if (n <= 1U) { return pts;
}

  const auto pivot_it = std::ranges::min_element(pts, LessPivot);
  std::iter_swap(pts.begin(), pivot_it);
  const Point& pivot = pts.front();

  auto angle_less = [&pivot, &pts](std::size_t i, std::size_t j) {
    const std::int64_t cr = Cross(pivot, pts[i], pts[j]);
    if (cr != 0) { return cr > 0;
}
    return Dist2(pivot, pts[i]) < Dist2(pivot, pts[j]);
  };

  if (n == 2U) { return pts;
}

  std::vector<std::size_t> idx;
  idx.reserve(n - 1U);
  for (std::size_t i = 1; i < n; ++i) { idx.push_back(i);
}
  std::ranges::sort(idx, angle_less);

  std::vector<Point> hull;
  hull.reserve(n);
  hull.push_back(pts[0]);
  hull.push_back(pts[idx[0]]);
  for (std::size_t k = 1; k < idx.size(); ++k) {
    const Point& p = pts[idx[k]];
    while (hull.size() >= 2U &&
           Cross(hull[hull.size() - 2U], hull[hull.size() - 1U], p) <= static_cast<std::int64_t>(0)) {
      hull.pop_back();
    }
    hull.push_back(p);
  }

  if (hull.size() == 2U && LessPivot(hull[1], hull[0])) { std::swap(hull[0], hull[1]);
}
  return hull;
}

}  // namespace

NazarovaKConvHullConstGrahMPI::NazarovaKConvHullConstGrahMPI(const InType& in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool NazarovaKConvHullConstGrahMPI::ValidationImpl() {
  return true;
}

bool NazarovaKConvHullConstGrahMPI::PreProcessingImpl() {
  points_ = GetInput().points;
  GetOutput().clear();
  return true;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
bool NazarovaKConvHullConstGrahMPI::RunImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const std::size_t n = points_.size();
  const std::size_t start = (static_cast<std::size_t>(rank) * n) / static_cast<std::size_t>(size);
  const std::size_t end = (static_cast<std::size_t>(rank + 1) * n) / static_cast<std::size_t>(size);

  std::vector<Point> local_points;
  local_points.reserve(end - start);
  for (std::size_t i = start; i < end; ++i) {
    local_points.push_back(points_[i]);
  }

  std::vector<Point> local_hull = GrahamScan(local_points);

  std::vector<int> sendbuf;
  sendbuf.reserve(local_hull.size() * 2U);
  for (const auto& p : local_hull) {
    sendbuf.push_back(p.x);
    sendbuf.push_back(p.y);
  }

  int sendcount = static_cast<int>(sendbuf.size());
  std::vector<int> recvcounts;
  if (rank == 0) {
    recvcounts.resize(static_cast<std::size_t>(size), 0);
  }
  MPI_Gather(&sendcount, 1, MPI_INT, (rank == 0 ? recvcounts.data() : nullptr), 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> displs;
  std::vector<int> recvbuf;
  int total = 0;
  if (rank == 0) {
    displs.resize(static_cast<std::size_t>(size), 0);
    for (int i = 0; i < size; i++) {
      displs[static_cast<std::size_t>(i)] = total;
      total += recvcounts[static_cast<std::size_t>(i)];
    }
    recvbuf.resize(static_cast<std::size_t>(total), 0);
  }

  MPI_Gatherv(sendcount > 0 ? sendbuf.data() : nullptr, sendcount, MPI_INT, (rank == 0 ? recvbuf.data() : nullptr),
              (rank == 0 ? recvcounts.data() : nullptr), (rank == 0 ? displs.data() : nullptr), MPI_INT, 0,
              MPI_COMM_WORLD);

  std::vector<Point> hull;
  if (rank == 0) {
    std::vector<Point> candidates;
    candidates.reserve(static_cast<std::size_t>(total / 2));
    for (int i = 0; i + 1 < total; i += 2) {
      const auto idx = static_cast<std::size_t>(i);
      candidates.push_back(Point{.x = recvbuf[idx], .y = recvbuf[idx + 1U]});
    }
    hull = GrahamScan(candidates);
  }

  int hull_count = (rank == 0) ? static_cast<int>(hull.size()) : 0;
  MPI_Bcast(&hull_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> hull_buf;
  hull_buf.resize(static_cast<std::size_t>(hull_count) * 2U, 0);
  if (rank == 0) {
    for (int i = 0; i < hull_count; i++) {
      const auto idx = static_cast<std::size_t>(i);
      hull_buf[2U * idx] = hull[idx].x;
      hull_buf[(2U * idx) + 1U] = hull[idx].y;
    }
  }
  MPI_Bcast(hull_buf.data(), static_cast<int>(hull_buf.size()), MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<Point> out;
  out.reserve(static_cast<std::size_t>(hull_count));
  for (int i = 0; i < hull_count; i++) {
    const auto idx = static_cast<std::size_t>(i);
    out.push_back(Point{.x = hull_buf[2U * idx], .y = hull_buf[(2U * idx) + 1U]});
  }
  GetOutput() = out;

  return true;
}

bool NazarovaKConvHullConstGrahMPI::PostProcessingImpl() {
  return GetOutput().size() <= GetInput().points.size();
}

}  // namespace nazarova_k_conv_hull_const_grah_processes

