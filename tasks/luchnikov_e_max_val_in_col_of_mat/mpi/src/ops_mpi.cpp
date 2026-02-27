#include "luchnikov_e_max_val_in_col_of_mat/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <limits>
#include <vector>

#include "luchnikov_e_max_val_in_col_of_mat/common/include/common.hpp"
#include "util/include/util.hpp"

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

    size_t cols = matrix[0].size();
    for (const auto &row : matrix) {
      if (row.size() != cols) {
        return false;
      }
    }
  }

  return GetOutput().empty();
}

bool LuchnikovEMaxValInColOfMatMPI::PreProcessingImpl() {
  const auto &matrix = GetInput();

  if (rank_ == 0) {
    if (!matrix.empty()) {
      rows_ = static_cast<int>(matrix.size());
      cols_ = static_cast<int>(matrix[0].size());
    }
  }

  MPI_Bcast(&rows_, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols_, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank_ != 0) {
    matrix_.resize(rows_, std::vector<int>(cols_));
  }

  for (int i = 0; i < rows_; ++i) {
    MPI_Bcast(matrix_[i].data(), cols_, MPI_INT, 0, MPI_COMM_WORLD);
  }

  if (rank_ == 0) {
    matrix_ = matrix;
  }

  result_.assign(cols_, std::numeric_limits<int>::min());

  return true;
}

bool LuchnikovEMaxValInColOfMatMPI::RunImpl() {
  if (matrix_.empty()) {
    return false;
  }

  std::vector<int> sendcounts(size_);
  std::vector<int> displs(size_);

  int rows_per_process = rows_ / size_;
  int remainder = rows_ % size_;

  int offset = 0;
  for (int i = 0; i < size_; ++i) {
    int current_rows = rows_per_process + (i < remainder ? 1 : 0);
    sendcounts[i] = current_rows * cols_;
    displs[i] = offset;
    offset += sendcounts[i];
  }

  std::vector<int> flat_matrix;
  if (rank_ == 0) {
    flat_matrix.reserve(rows_ * cols_);
    for (const auto &row : matrix_) {
      flat_matrix.insert(flat_matrix.end(), row.begin(), row.end());
    }
  }

  std::vector<int> local_flat(sendcounts[rank_]);
  MPI_Scatterv(flat_matrix.data(), sendcounts.data(), displs.data(), MPI_INT, local_flat.data(), sendcounts[rank_],
               MPI_INT, 0, MPI_COMM_WORLD);

  int local_rows = sendcounts[rank_] / cols_;
  std::vector<int> local_max(cols_, std::numeric_limits<int>::min());

  for (int i = 0; i < local_rows; ++i) {
    for (int j = 0; j < cols_; ++j) {
      int val = local_flat[i * cols_ + j];
      if (val > local_max[j]) {
        local_max[j] = val;
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
