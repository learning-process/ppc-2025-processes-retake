#include "luchnikov_e_max_val_in_col_of_mat/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <utility>
#include <vector>

#include "luchnikov_e_max_val_in_col_of_mat/common/include/common.hpp"

namespace luchnikov_e_max_val_in_col_of_mat {

LuchnikovEMaxValInColOfMatMPI::LuchnikovEMaxValInColOfMatMPI(const InType &in) : matrix_(in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  result_.clear();
}

bool LuchnikovEMaxValInColOfMatMPI::ValidationImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &size_);

  if (rank_ != 0) {
    return GetOutput().empty();
  }

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

std::vector<int> LuchnikovEMaxValInColOfMatMPI::RunSequential() const {
  std::vector<int> local_result(cols_, INT_MIN);
  for (int j = 0; j < cols_; ++j) {
    for (int i = 0; i < rows_; ++i) {
      local_result[j] = std::max(matrix_[i][j], local_result[j]);
    }
  }
  return local_result;
}

std::vector<int> LuchnikovEMaxValInColOfMatMPI::PrepareFlatMatrix() const {
  std::vector<int> flat;
  if (rank_ != 0) {
    return flat;
  }

  flat.reserve(static_cast<size_t>(rows_) * static_cast<size_t>(cols_));
  for (int i = 0; i < rows_; ++i) {
    flat.insert(flat.end(), matrix_[i].begin(), matrix_[i].end());
  }
  return flat;
}

std::pair<std::vector<int>, std::vector<int>> LuchnikovEMaxValInColOfMatMPI::CalculateDistribution() const {
  std::vector<int> sendcounts(size_, 0);
  std::vector<int> displs(size_, 0);

  int base_rows = rows_ / size_;
  int extra_rows = rows_ % size_;
  int offset = 0;

  for (int i = 0; i < size_; ++i) {
    int current_rows = base_rows + ((i < extra_rows) ? 1 : 0);
    sendcounts[i] = current_rows * cols_;
    displs[i] = offset;
    offset += sendcounts[i];
  }

  return {sendcounts, displs};
}

std::vector<int> LuchnikovEMaxValInColOfMatMPI::ComputeLocalMax(const std::vector<int> &local_flat, int local_rows) const {
  std::vector<int> local_max(cols_, INT_MIN);
  for (int i = 0; i < local_rows; ++i) {
    for (int j = 0; j < cols_; ++j) {
      int idx = (i * cols_) + j;
      local_max[j] = std::max(local_flat[idx], local_max[j]);
    }
  }
  return local_max;
}

bool LuchnikovEMaxValInColOfMatMPI::RunImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &size_);

  if (size_ == 1) {
    result_ = RunSequential();
    return true;
  }

  std::vector<int> flat_matrix = PrepareFlatMatrix();
  auto [sendcounts, displs] = CalculateDistribution();

  std::vector<int> local_flat(sendcounts[rank_]);
  const int *send_data = (rank_ == 0) ? flat_matrix.data() : nullptr;

  MPI_Scatterv(send_data, sendcounts.data(), displs.data(), MPI_INT,
               local_flat.data(), sendcounts[rank_], MPI_INT, 0, MPI_COMM_WORLD);

  int local_rows = sendcounts[rank_] / cols_;
  std::vector<int> local_max = ComputeLocalMax(local_flat, local_rows);

  MPI_Allreduce(local_max.data(), result_.data(), cols_, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

  return true;
}

bool LuchnikovEMaxValInColOfMatMPI::PostProcessingImpl() {
  GetOutput() = result_;
  return !result_.empty();
}

}  // namespace luchnikov_e_max_val_in_col_of_mat