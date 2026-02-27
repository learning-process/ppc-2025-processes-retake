#include "yushkova_p_hypercube/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstdint>

#include "yushkova_p_hypercube/common/include/common.hpp"

namespace yushkova_p_hypercube {

namespace {

bool IsPowerOfTwo(int value) {
  return value > 0 && (value & (value - 1)) == 0;
}

std::uint64_t CountLocalEdges(std::uint64_t start_vertex, std::uint64_t end_vertex, int dimension) {
  std::uint64_t local_edges = 0;
  for (std::uint64_t vertex = start_vertex; vertex < end_vertex; ++vertex) {
    for (int bit = 0; bit < dimension; ++bit) {
      const std::uint64_t neighbor = vertex ^ (static_cast<std::uint64_t>(1) << bit);
      if (vertex < neighbor) {
        ++local_edges;
      }
    }
  }
  return local_edges;
}

}  // namespace

YushkovaPHypercubeMPI::YushkovaPHypercubeMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool YushkovaPHypercubeMPI::ValidationImpl() {
  int world_size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  const InType n = GetInput();
  return IsPowerOfTwo(world_size) && n > 0 && n < 63;
}

bool YushkovaPHypercubeMPI::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool YushkovaPHypercubeMPI::RunImpl() {
  int rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const InType n = GetInput();

  if (!IsPowerOfTwo(world_size)) {
    return false;
  }
  const std::uint64_t vertices = static_cast<std::uint64_t>(1) << n;
  const auto u_rank = static_cast<std::uint64_t>(rank);
  const auto u_world = static_cast<std::uint64_t>(world_size);
  const std::uint64_t base = vertices / u_world;
  const std::uint64_t tail = vertices % u_world;
  const std::uint64_t local_count = base + (u_rank < tail ? 1ULL : 0ULL);
  const std::uint64_t local_start = (u_rank * base) + std::min(u_rank, tail);
  const std::uint64_t local_end = local_start + local_count;

  std::uint64_t sum_edges = CountLocalEdges(local_start, local_end, n);
  for (int mask = 1; mask < world_size; mask <<= 1) {
    const int partner = rank ^ mask;
    std::uint64_t from_partner = 0;
    MPI_Sendrecv(&sum_edges, 1, MPI_UINT64_T, partner, 700 + mask, &from_partner, 1, MPI_UINT64_T, partner, 700 + mask,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    if ((rank & mask) == 0) {
      sum_edges += from_partner;
    } else {
      break;
    }
  }

  MPI_Bcast(&sum_edges, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
  GetOutput() = static_cast<OutType>(sum_edges);
  return true;
}

bool YushkovaPHypercubeMPI::PostProcessingImpl() {
  return true;
}

}  // namespace yushkova_p_hypercube
