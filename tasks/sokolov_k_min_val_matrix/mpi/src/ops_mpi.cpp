#include "sokolov_k_min_val_matrix/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

#include "sokolov_k_min_val_matrix/common/include/common.hpp"

namespace k_sokolov_min_val_matrix {

SokolovKMinValMatrixMPI::SokolovKMinValMatrixMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool SokolovKMinValMatrixMPI::ValidationImpl() {
  return GetInput() > 0;
}

bool SokolovKMinValMatrixMPI::PreProcessingImpl() {
  int n = GetInput();
  if (n <= 0) {
    rows_ = 0;
    cols_ = 0;
    return true;
  }
  rows_ = n;
  cols_ = n;

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    matrix_.resize(static_cast<std::size_t>(n) * n);
    for (int i = 0; i < n * n; i++) {
      matrix_[i] = i + 1;
    }
  }

  return true;
}

bool SokolovKMinValMatrixMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int total = rows_ * cols_;

  std::vector<int> send_counts(size);
  std::vector<int> displs(size);

  int base_count = total / size;
  int remainder = total % size;

  for (int i = 0; i < size; i++) {
    send_counts[i] = base_count + (i < remainder ? 1 : 0);
    displs[i] = (i == 0) ? 0 : displs[i - 1] + send_counts[i - 1];
  }

  std::vector<int> local_data(send_counts[rank]);
  MPI_Scatterv(matrix_.data(), send_counts.data(), displs.data(), MPI_INT, local_data.data(), send_counts[rank],
               MPI_INT, 0, MPI_COMM_WORLD);

  int local_min = std::numeric_limits<int>::max();
  if (!local_data.empty()) {
    local_min = *std::ranges::min_element(local_data);
  }

  int global_min = 0;
  MPI_Allreduce(&local_min, &global_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  min_val_ = global_min;

  return true;
}

bool SokolovKMinValMatrixMPI::PostProcessingImpl() {
  GetOutput() = min_val_;
  return true;
}

}  // namespace k_sokolov_min_val_matrix
