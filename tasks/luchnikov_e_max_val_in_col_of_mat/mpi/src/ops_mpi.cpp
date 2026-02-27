#include "luchnikov_e_max_val_in_col_of_mat/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <climits>
#include <vector>

namespace luchnikov_e_max_val_in_col_of_mat {

LuchnikovEMaxValInColOfMatMPI::LuchnikovEMaxValInColOfMatMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  matrix_ = in;
  result_.clear();
}

bool LuchnikovEMaxValInColOfMatMPI::ValidationImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &size_);

  if (rank_ == 0) {
    const auto &matrix = GetInput();

    if (matrix.empty()) {
      return false;
    }

    cols_ = static_cast<int>(matrix[0].size());
    for (const auto &row : matrix) {
      if (static_cast<int>(row.size()) != cols_) {
        return false;
      }
    }
  }

  return GetOutput().empty();
}

bool LuchnikovEMaxValInColOfMatMPI::PreProcessingImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &size_);

  if (rank_ == 0) {
    const auto &matrix = GetInput();
    rows_ = static_cast<int>(matrix.size());
    cols_ = static_cast<int>(matrix[0].size());
  }

  MPI_Bcast(&rows_, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols_, 1, MPI_INT, 0, MPI_COMM_WORLD);

  result_.assign(cols_, INT_MIN);
  return true;
}

bool LuchnikovEMaxValInColOfMatMPI::RunImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &size_);

  std::vector<int> flat_matrix;
  if (rank_ == 0) {
    const auto &matrix = GetInput();
    flat_matrix.reserve(rows_ * cols_);
    for (int i = 0; i < rows_; ++i) {
      for (int j = 0; j < cols_; ++j) {
        flat_matrix.push_back(matrix[i][j]);
      }
    }
  }

  std::vector<int> sendcounts(size_, 0);
  std::vector<int> displs(size_, 0);

  int base_rows = rows_ / size_;
  int extra_rows = rows_ % size_;

  int offset = 0;
  for (int i = 0; i < size_; ++i) {
    int current_rows = base_rows + (i < extra_rows ? 1 : 0);
    sendcounts[i] = current_rows * cols_;
    displs[i] = offset;
    offset += sendcounts[i];
  }

  std::vector<int> local_flat(sendcounts[rank_]);

  MPI_Scatterv(flat_matrix.data(), sendcounts.data(), displs.data(), MPI_INT, local_flat.data(), sendcounts[rank_],
               MPI_INT, 0, MPI_COMM_WORLD);

  int local_rows = sendcounts[rank_] / cols_;
  std::vector<int> local_max(cols_, INT_MIN);

  for (int i = 0; i < local_rows; ++i) {
    for (int j = 0; j < cols_; ++j) {
      int idx = i * cols_ + j;
      if (local_flat[idx] > local_max[j]) {
        local_max[j] = local_flat[idx];
      }
    }
  }

  MPI_Allreduce(local_max.data(), result_.data(), cols_, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool LuchnikovEMaxValInColOfMatMPI::PostProcessingImpl() {
  GetOutput() = result_;
  return !result_.empty();
}

}  // namespace luchnikov_e_max_val_in_col_of_mat
