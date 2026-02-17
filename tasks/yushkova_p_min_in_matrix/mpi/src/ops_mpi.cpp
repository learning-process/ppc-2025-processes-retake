#include "yushkova_p_min_in_matrix/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "yushkova_p_min_in_matrix/common/include/common.hpp"

namespace yushkova_p_min_in_matrix {

namespace {

inline int GenerateValue(int64_t i, int64_t j) {
  constexpr int64_t kA = 1103515245LL;
  constexpr int64_t kC = 12345LL;
  constexpr int64_t kM = 2147483648LL;

  int64_t seed = ((i % kM) * (100000007LL % kM) + (j % kM) * (1000000009LL % kM)) % kM;
  seed = (seed ^ 42LL) % kM;
  int64_t val = ((kA % kM) * (seed % kM) + kC) % kM;

  return static_cast<int>((val % 2000001LL) - 1000000LL);
}

}  // namespace

YushkovaPMinInMatrixMPI::YushkovaPMinInMatrixMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool YushkovaPMinInMatrixMPI::ValidationImpl() {
  return GetInput() > 0;
}

bool YushkovaPMinInMatrixMPI::PreProcessingImpl() {
  GetOutput().assign(GetInput(), 0);
  return true;
}

bool YushkovaPMinInMatrixMPI::RunImpl() {
  int n = static_cast<int>(GetInput());
  int world_size = 0;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int base_rows = n / world_size;
  int extra = n % world_size;

  int my_start = rank * base_rows + std::min(rank, extra);
  int my_count = base_rows + (rank < extra ? 1 : 0);

  std::vector<int> local_results(my_count);
  for (int i = 0; i < my_count; ++i) {
    int current_row = my_start + i;
    int row_min = std::numeric_limits<int>::max();

    for (int j = 0; j < n; ++j) {
      int val = GenerateValue(current_row, j);
      if (val < row_min) {
        row_min = val;
      }
    }
    local_results[i] = row_min;
  }

  std::vector<int> recv_counts(world_size);
  std::vector<int> offsets(world_size);

  int current_offset = 0;
  for (int i = 0; i < world_size; ++i) {
    recv_counts[i] = base_rows + (i < extra ? 1 : 0);
    offsets[i] = current_offset;
    current_offset += recv_counts[i];
  }

  MPI_Allgatherv(local_results.data(), my_count, MPI_INT, GetOutput().data(), recv_counts.data(), offsets.data(),
                 MPI_INT, MPI_COMM_WORLD);

  return true;
}

bool YushkovaPMinInMatrixMPI::PostProcessingImpl() {
  return GetOutput().size() == static_cast<size_t>(GetInput());
}

}  // namespace yushkova_p_min_in_matrix
