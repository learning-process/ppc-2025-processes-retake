#include "Nazarova_K_Conv_hull_const_Grah/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "Nazarova_K_Conv_hull_const_Grah/common/include/common.hpp"

namespace nazarova_k_conv_hull_const_grah_processes {
namespace {

inline bool LessPivot(const Point &a, const Point &b) {
  if (a.y != b.y) {
    return a.y < b.y;
  }
  return a.x < b.x;
}

inline bool LessXY(const Point &a, const Point &b) {
  if (a.x != b.x) {
    return a.x < b.x;
  }
  return a.y < b.y;
}

inline std::int64_t Cross(const Point &o, const Point &a, const Point &b) {
  const std::int64_t ax = static_cast<std::int64_t>(a.x) - static_cast<std::int64_t>(o.x);
  const std::int64_t ay = static_cast<std::int64_t>(a.y) - static_cast<std::int64_t>(o.y);
  const std::int64_t bx = static_cast<std::int64_t>(b.x) - static_cast<std::int64_t>(o.x);
  const std::int64_t by = static_cast<std::int64_t>(b.y) - static_cast<std::int64_t>(o.y);
  return (ax * by) - (ay * bx);
}

inline std::int64_t Dist2(const Point &a, const Point &b) {
  const std::int64_t dx = static_cast<std::int64_t>(a.x) - static_cast<std::int64_t>(b.x);
  const std::int64_t dy = static_cast<std::int64_t>(a.y) - static_cast<std::int64_t>(b.y);
  return (dx * dx) + (dy * dy);
}

std::vector<Point> GrahamScan(std::vector<Point> &pts) {
  std::ranges::sort(pts, LessXY);
  const auto [new_end, last] = std::ranges::unique(pts);
  pts.erase(new_end, last);

  const std::size_t n = pts.size();
  if (n <= 1U) {
    return pts;
  }

  const auto pivot_it = std::ranges::min_element(pts, LessPivot);
  std::iter_swap(pts.begin(), pivot_it);
  const Point &pivot = pts.front();

  auto angle_less = [&pivot, &pts](std::size_t i, std::size_t j) {
    const std::int64_t cr = Cross(pivot, pts[i], pts[j]);
    if (cr != 0) {
      return cr > 0;
    }
    return Dist2(pivot, pts[i]) < Dist2(pivot, pts[j]);
  };

  if (n == 2U) {
    return pts;
  }

  std::vector<std::size_t> idx;
  idx.reserve(n - 1U);
  for (std::size_t i = 1; i < n; ++i) {
    idx.push_back(i);
  }
  std::ranges::sort(idx, angle_less);

  std::vector<Point> hull;
  hull.reserve(n);
  hull.push_back(pts[0]);
  hull.push_back(pts[idx[0]]);
  for (std::size_t k = 1; k < idx.size(); ++k) {
    const Point &p = pts[idx[k]];
    while (hull.size() >= 2U &&
           Cross(hull[hull.size() - 2U], hull[hull.size() - 1U], p) <= static_cast<std::int64_t>(0)) {
      hull.pop_back();
    }
    hull.push_back(p);
  }

  if (hull.size() == 2U && LessPivot(hull[1], hull[0])) {
    std::swap(hull[0], hull[1]);
  }
  return hull;
}

std::vector<Point> GetLocalPoints(const std::vector<Point> &points, int rank, int size) {
  const std::size_t n = points.size();
  const std::size_t start = (static_cast<std::size_t>(rank) * n) / static_cast<std::size_t>(size);
  const std::size_t end = (static_cast<std::size_t>(rank + 1) * n) / static_cast<std::size_t>(size);
  std::vector<Point> local;
  local.reserve(end - start);
  for (std::size_t i = start; i < end; ++i) {
    local.push_back(points[i]);
  }
  return local;
}

std::vector<int> HullToIntBuffer(const std::vector<Point> &hull) {
  std::vector<int> buf;
  buf.reserve(hull.size() * 2U);
  for (const auto &p : hull) {
    buf.push_back(p.x);
    buf.push_back(p.y);
  }
  return buf;
}

void FillGatherBuffers(int size, const std::vector<int> &recvcounts, std::vector<int> &displs,
                       std::vector<int> &recvbuf, int &total) {
  displs.resize(static_cast<std::size_t>(size), 0);
  total = 0;
  for (int i = 0; i < size; i++) {
    displs[static_cast<std::size_t>(i)] = total;
    total += recvcounts[static_cast<std::size_t>(i)];
  }
  recvbuf.resize(static_cast<std::size_t>(total), 0);
}

std::vector<Point> RecvbufToHull(const std::vector<int> &recvbuf, int total) {
  std::vector<Point> candidates;
  candidates.reserve(static_cast<std::size_t>(total / 2));
  for (int i = 0; i + 1 < total; i += 2) {
    const auto idx = static_cast<std::size_t>(i);
    candidates.push_back(Point{.x = recvbuf[idx], .y = recvbuf[idx + 1U]});
  }
  return GrahamScan(candidates);
}

std::vector<int> HullToBroadcastBuffer(const std::vector<Point> &hull) {
  std::vector<int> buf(static_cast<std::size_t>(hull.size()) * 2U, 0);
  for (std::size_t i = 0; i < hull.size(); i++) {
    buf[2U * i] = hull[i].x;
    buf[(2U * i) + 1U] = hull[i].y;
  }
  return buf;
}

std::vector<Point> HullBufToPoints(const std::vector<int> &hull_buf, int hull_count) {
  std::vector<Point> out;
  out.reserve(static_cast<std::size_t>(hull_count));
  for (int i = 0; i < hull_count; i++) {
    const auto idx = static_cast<std::size_t>(i);
    out.push_back(Point{.x = hull_buf[2U * idx], .y = hull_buf[(2U * idx) + 1U]});
  }
  return out;
}

}  // namespace

NazarovaKConvHullConstGrahMPI::NazarovaKConvHullConstGrahMPI(const InType &in) {
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

bool NazarovaKConvHullConstGrahMPI::RunImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::vector<Point> local_points = GetLocalPoints(points_, rank, size);
  std::vector<Point> local_hull = GrahamScan(local_points);
  std::vector<int> sendbuf = HullToIntBuffer(local_hull);
  int sendcount = static_cast<int>(sendbuf.size());

  std::vector<int> recvcounts(rank == 0 ? static_cast<std::size_t>(size) : 0U, 0);
  int *recvcounts_ptr = rank == 0 ? recvcounts.data() : nullptr;
  MPI_Gather(&sendcount, 1, MPI_INT, recvcounts_ptr, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> displs;
  std::vector<int> recvbuf;
  int total = 0;
  if (rank == 0) {
    FillGatherBuffers(size, recvcounts, displs, recvbuf, total);
  }

  int *recvbuf_ptr = rank == 0 ? recvbuf.data() : nullptr;
  int *recvcounts_data = rank == 0 ? recvcounts.data() : nullptr;
  int *displs_data = rank == 0 ? displs.data() : nullptr;
  int *sendbuf_ptr = sendcount > 0 ? sendbuf.data() : nullptr;
  MPI_Gatherv(sendbuf_ptr, sendcount, MPI_INT, recvbuf_ptr, recvcounts_data, displs_data, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<Point> hull = rank == 0 ? RecvbufToHull(recvbuf, total) : std::vector<Point>{};
  int hull_count = rank == 0 ? static_cast<int>(hull.size()) : 0;
  MPI_Bcast(&hull_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> hull_buf =
      rank == 0 ? HullToBroadcastBuffer(hull) : std::vector<int>(static_cast<std::size_t>(hull_count) * 2U, 0);
  MPI_Bcast(hull_buf.data(), static_cast<int>(hull_buf.size()), MPI_INT, 0, MPI_COMM_WORLD);

  GetOutput() = HullBufToPoints(hull_buf, hull_count);
  return true;
}

bool NazarovaKConvHullConstGrahMPI::PostProcessingImpl() {
  return GetOutput().size() <= GetInput().points.size();
}

}  // namespace nazarova_k_conv_hull_const_grah_processes
