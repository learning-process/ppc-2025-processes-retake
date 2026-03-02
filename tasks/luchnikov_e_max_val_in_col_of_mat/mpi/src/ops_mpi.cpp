#include "luchnikov_e_max_val_in_col_of_mat/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

namespace luchnikov_e_max_val_in_col_of_mat {

LuchnilkovEMaxValInColOfMatMpi::LuchnilkovEMaxValInColOfMatMpi(const InType &in) : matrix_(in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  result_.clear();
}

bool LuchnilkovEMaxValInColOfMatMpi::ValidationImpl() {
  const auto &matrix = GetInput();

  if (matrix.empty()) {
    return false;
  }

  const size_t cols = matrix[0].size();
  for (const auto &row : matrix) {
    if (row.size() != cols) {
      return false;
    }
  }

  return GetOutput().empty();
}

bool LuchnilkovEMaxValInColOfMatMpi::PreProcessingImpl() {
  const auto &matrix = GetInput();

  if (!matrix.empty()) {
    const size_t cols = matrix[0].size();
    result_.assign(cols, std::numeric_limits<int>::min());
  }

  return true;
}

bool LuchnilkovEMaxValInColOfMatMpi::RunImpl() {
  const auto &matrix = GetInput();

  if (matrix.empty()) {
    return false;
  }

  int rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const int total_rows = static_cast<int>(matrix.size());
  const int num_cols = static_cast<int>(matrix[0].size());

  std::vector<int> send_counts(world_size);
  std::vector<int> displacements(world_size);

  const int rows_portion = total_rows / world_size;
  const int extra_rows = total_rows % world_size;

  int current_position = 0;
  for (int proc = 0; proc < world_size; ++proc) {
    const int proc_rows = rows_portion + (proc < extra_rows ? 1 : 0);
    send_counts[proc] = proc_rows * num_cols;
    displacements[proc] = current_position;
    current_position += send_counts[proc];
  }

  std::vector<int> flat_matrix;
  if (rank == 0) {
    flat_matrix.reserve(static_cast<size_t>(total_rows) * static_cast<size_t>(num_cols));
    for (const auto &row : matrix) {
      flat_matrix.insert(flat_matrix.end(), row.begin(), row.end());
    }
  }

  const int local_elements = send_counts[rank];
  std::vector<int> local_data(local_elements);

  MPI_Scatterv(flat_matrix.data(), send_counts.data(), displacements.data(), MPI_INT, local_data.data(), local_elements,
               MPI_INT, 0, MPI_COMM_WORLD);

  const int local_rows = local_elements / num_cols;
  std::vector<int> local_max(num_cols, std::numeric_limits<int>::min());

  for (int i = 0; i < local_rows; ++i) {
    for (int j = 0; j < num_cols; ++j) {
      const int value = local_data[(i * num_cols) + j];
      local_max[j] = std::max(value, local_max[j]);
    }
  }

  std::vector<int> global_max(num_cols);
  MPI_Reduce(local_max.data(), global_max.data(), num_cols, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    result_ = global_max;
  }

  MPI_Bcast(result_.data(), num_cols, MPI_INT, 0, MPI_COMM_WORLD);

  return true;
}

bool LuchnilkovEMaxValInColOfMatMpi::PostProcessingImpl() {
  GetOutput() = result_;
  return !result_.empty();
}

}  // namespace luchnikov_e_max_val_in_col_of_mat
