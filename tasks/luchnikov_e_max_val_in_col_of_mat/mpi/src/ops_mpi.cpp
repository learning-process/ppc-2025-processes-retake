#include "luchnikov_e_max_val_in_col_of_mat/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <climits>
#include <vector>

namespace luchnikov_e_max_val_in_col_of_mat {

LuchnikovEMaxValInColOfMatMPI::LuchnikovEMaxValInColOfMatMPI(const InType& in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<int>();
}

bool LuchnikovEMaxValInColOfMatMPI::ValidationImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &size_);

  if (rank_ == 0) {
    if (GetInput().empty()) {
      return false;
    }
    rows_ = GetInput().size();
    cols_ = GetInput()[0].size();
    for (const auto& row : GetInput()) {
      if (row.size() != static_cast<size_t>(cols_)) {
        return false;
      }
    }
  }
  return true;
}

bool LuchnikovEMaxValInColOfMatMPI::PreProcessingImpl() {
  if (rank_ == 0) {
    rows_ = GetInput().size();
    cols_ = GetInput()[0].size();
  }

  MPI_Bcast(&rows_, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols_, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> matrix_data;
  if (rank_ == 0) {
    matrix_data.reserve(rows_ * cols_);
    for (const auto& row : GetInput()) {
      matrix_data.insert(matrix_data.end(), row.begin(), row.end());
    }
  }

  std::vector<int> sendcounts(size_);
  std::vector<int> displs(size_);

  int rows_per_process = rows_ / size_;
  int remainder = rows_ % size_;

  int offset = 0;
  for (int i = 0; i < size_; i++) {
    int current_rows = rows_per_process + (i < remainder ? 1 : 0);
    sendcounts[i] = current_rows * cols_;
    displs[i] = offset;
    offset += sendcounts[i];
  }

  local_matrix_data_.resize(sendcounts[rank_]);

  MPI_Scatterv(matrix_data.data(), sendcounts.data(), displs.data(), MPI_INT, local_matrix_data_.data(),
               sendcounts[rank_], MPI_INT, 0, MPI_COMM_WORLD);

  return true;
}

bool LuchnikovEMaxValInColOfMatMPI::RunImpl() {
  int local_rows = local_matrix_data_.size() / cols_;
  local_result_.resize(cols_, INT_MIN);

  for (int j = 0; j < cols_; j++) {
    for (int i = 0; i < local_rows; i++) {
      int val = local_matrix_data_[i * cols_ + j];
      if (val > local_result_[j]) {
        local_result_[j] = val;
      }
    }
  }

  return true;
}

bool LuchnikovEMaxValInColOfMatMPI::PostProcessingImpl() {
  GetOutput().resize(cols_);
  
  MPI_Allreduce(local_result_.data(), GetOutput().data(), cols_, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

  return true;
}

}  // namespace luchnikov_e_max_val_in_col_of_mat