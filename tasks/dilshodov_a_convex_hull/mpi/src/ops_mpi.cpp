#include "dilshodov_a_convex_hull/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

#include "dilshodov_a_convex_hull/common/include/common.hpp"

namespace dilshodov_a_convex_hull {

namespace {

int Cross(const Point &o, const Point &a, const Point &b) {
  return ((a.x - o.x) * (b.y - o.y)) - ((a.y - o.y) * (b.x - o.x));
}

bool PointLess(const Point &a, const Point &b) {
  return a.x < b.x || (a.x == b.x && a.y < b.y);
}

void GrahamScanInPlace(std::vector<Point> &pts) {
  if (pts.size() < 3) {
    return;
  }

  std::ranges::sort(pts, PointLess);
  pts.erase(std::ranges::unique(pts).begin(), pts.end());

  if (pts.size() < 3) {
    return;
  }

  std::vector<Point> hull;
  for (const auto &p : pts) {
    while (hull.size() >= 2 && Cross(hull[hull.size() - 2], hull.back(), p) <= 0) {
      hull.pop_back();
    }
    hull.push_back(p);
  }

  auto lower_size = hull.size();
  for (int i = static_cast<int>(pts.size()) - 2; i >= 0; --i) {
    const auto &p = pts[static_cast<std::size_t>(i)];
    while (hull.size() > lower_size && Cross(hull[hull.size() - 2], hull.back(), p) <= 0) {
      hull.pop_back();
    }
    hull.push_back(p);
  }

  hull.pop_back();
  pts = std::move(hull);
}

void ScatterPixels(const std::vector<int> &pixels, int n, int rank, int size, std::vector<int> &local_pixels,
                   std::vector<int> &sendcounts, std::vector<int> &displs) {
  int base = n / size;
  int rem = n % size;
  sendcounts.resize(static_cast<std::size_t>(size));
  displs.resize(static_cast<std::size_t>(size));
  for (int i = 0; i < size; ++i) {
    sendcounts[static_cast<std::size_t>(i)] = base + (i < rem ? 1 : 0);
    displs[static_cast<std::size_t>(i)] =
        (i == 0) ? 0 : displs[static_cast<std::size_t>(i) - 1] + sendcounts[static_cast<std::size_t>(i) - 1];
  }
  int local_n = sendcounts[static_cast<std::size_t>(rank)];
  local_pixels.resize(static_cast<std::size_t>(local_n));
  MPI_Scatterv(rank == 0 ? pixels.data() : nullptr, sendcounts.data(), displs.data(), MPI_INT, local_pixels.data(),
               local_n, MPI_INT, 0, MPI_COMM_WORLD);
}

void GatherLocalHulls(std::vector<Point> &local_hull, int rank, int size, MPI_Datatype mpi_point,
                      std::vector<Point> &gathered) {
  int local_size = static_cast<int>(local_hull.size());
  std::vector<int> recv_sizes(static_cast<std::size_t>(size));
  MPI_Gather(&local_size, 1, MPI_INT, recv_sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> displs_hull(static_cast<std::size_t>(size), 0);
  int total = 0;
  if (rank == 0) {
    for (int i = 0; i < size; ++i) {
      displs_hull[static_cast<std::size_t>(i)] = total;
      total += recv_sizes[static_cast<std::size_t>(i)];
    }
  }
  gathered.resize(static_cast<std::size_t>(total));
  MPI_Gatherv(local_hull.data(), local_size, mpi_point, gathered.data(), recv_sizes.data(), displs_hull.data(),
              mpi_point, 0, MPI_COMM_WORLD);
}

}  // namespace

ConvexHullMPI::ConvexHullMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  InType copy = in;
  GetInput() = std::move(copy);
  GetOutput().clear();
}

bool ConvexHullMPI::ValidationImpl() {
  const auto &input = GetInput();
  if (input.size() < 3) {
    return false;
  }
  width_ = input[0];
  height_ = input[1];
  return width_ > 0 && height_ > 0 && static_cast<int>(input.size()) == (width_ * height_) + 2;
}

bool ConvexHullMPI::PreProcessingImpl() {
  return true;
}

bool ConvexHullMPI::RunImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int n = 0;
  if (rank == 0) {
    n = width_ * height_;
  }
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&width_, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&height_, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> pixels;
  if (rank == 0) {
    const auto &input = GetInput();
    pixels.assign(input.begin() + 2, input.end());
  }

  std::vector<int> local_pixels;
  std::vector<int> sendcounts;
  std::vector<int> displs;
  ScatterPixels(pixels, n, rank, size, local_pixels, sendcounts, displs);

  int global_offset = displs[static_cast<std::size_t>(rank)];
  std::vector<Point> local_points;
  for (std::size_t i = 0; i < local_pixels.size(); ++i) {
    if (local_pixels[i] != 0) {
      int global_idx = global_offset + static_cast<int>(i);
      int px = global_idx % width_;
      int py = global_idx / width_;
      local_points.push_back({px, py});
    }
  }

  GrahamScanInPlace(local_points);

  MPI_Datatype mpi_point = MPI_DATATYPE_NULL;
  MPI_Type_contiguous(2, MPI_INT, &mpi_point);
  MPI_Type_commit(&mpi_point);

  std::vector<Point> gathered;
  GatherLocalHulls(local_points, rank, size, mpi_point, gathered);

  if (rank == 0) {
    GrahamScanInPlace(gathered);
    if (gathered.size() < 3) {
      gathered.clear();
    }
    GetOutput() = std::move(gathered);
  }

  MPI_Type_free(&mpi_point);
  return true;
}

bool ConvexHullMPI::PostProcessingImpl() {
  return true;
}

}  // namespace dilshodov_a_convex_hull
