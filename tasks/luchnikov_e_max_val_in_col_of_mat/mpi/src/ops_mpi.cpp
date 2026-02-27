#include "luchnikov_e_max_val_in_col_of_mat/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <climits>
#include <vector>

namespace luchnikov_e_max_val_in_col_of_mat {

LuchnikovEMaxValInColOfMatMPI::LuchnikovEMaxValInColOfMatMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  input_copy_ = in;
  GetOutput() = std::vector<int>();
}

bool LuchnikovEMaxValInColOfMatMPI::ValidationImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &size_);

  if (rank_ == 0) {
    if (GetInput().empty()) {
      return false;
    }
    rows_ = static_cast<int>(GetInput().size());
    cols_ = static_cast<int>(GetInput()[0].size());

    for (const auto &row : GetInput()) {
      if (static_cast<int>(row.size()) != cols_) {
        return false;
      }
    }
  }
  return GetOutput().empty();
}

bool LuchnikovEMaxValInColOfMatMPI::PreProcessingImpl() {
  if (rank_ == 0) {
    rows_ = static_cast<int>(GetInput().size());
    cols_ = static_cast<int>(GetInput()[0].size());
    input_copy_ = GetInput();
  }

  MPI_Bcast(&rows_, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols_, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank_ != 0) {
    input_copy_.resize(rows_, std::vector<int>(cols_));
  }

  std::vector<int> flat_matrix;
  if (rank_ == 0) {
    flat_matrix.reserve(rows_ * cols_);
    for (const auto &row : input_copy_) {
      flat_matrix.insert(flat_matrix.end(), row.begin(), row.end());
    }
  } else {
    flat_matrix.resize(rows_ * cols_);
  }

  MPI_Bcast(flat_matrix.data(), rows_ * cols_, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank_ != 0) {
    for (int i = 0; i < rows_; ++i) {
      for (int j = 0; j < cols_; ++j) {
        input_copy_[i][j] = flat_matrix[i * cols_ + j];
      }
    }
  }

  local_result_.assign(cols_, INT_MIN);
  return true;
}

bool LuchnikovEMaxValInColOfMatMPI::RunImpl() {
  if (input_copy_.empty()) {
    return false;
  }

  int chunk_size = rows_ / size_;
  int remainder = rows_ % size_;

  int start_idx = rank_ * chunk_size + (rank_ < remainder ? rank_ : remainder);
  int end_idx = start_idx + chunk_size + (rank_ < remainder ? 1 : 0);
  end_idx = std::min(end_idx, rows_);

  for (int i = start_idx; i < end_idx; ++i) {
    for (int j = 0; j < cols_; ++j) {
      int current_val = input_copy_[i][j];
      if (current_val > local_result_[j]) {
        local_result_[j] = current_val;
      }
    }
  }

  std::vector<int> global_result;
  if (rank_ == 0) {
    global_result.resize(cols_, INT_MIN);
  }

  MPI_Reduce(local_result_.data(), global_result.data(), cols_, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

  if (rank_ == 0) {
    local_result_ = global_result;
  }

  return true;
}

bool LuchnikovEMaxValInColOfMatMPI::PostProcessingImpl() {
  if (rank_ == 0) {
    GetOutput() = local_result_;
  }
  return !local_result_.empty();
}

}  // namespace luchnikov_e_max_val_in_col_of_mat
