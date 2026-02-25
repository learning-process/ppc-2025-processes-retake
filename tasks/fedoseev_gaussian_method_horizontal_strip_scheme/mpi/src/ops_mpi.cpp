#include "fedoseev_gaussian_method_horizontal_strip_scheme/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "fedoseev_gaussian_method_horizontal_strip_scheme/common/include/common.hpp"

namespace fedoseev_gaussian_method_horizontal_strip_scheme {

FedoseevGaussianMethodHorizontalStripSchemeMPI::FedoseevGaussianMethodHorizontalStripSchemeMPI(
    const InType &input_data) {
  SetTypeOfTask(GetStaticTypeOfTask());

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    GetInput() = input_data;
  }

  GetOutput().clear();
}

bool FedoseevGaussianMethodHorizontalStripSchemeMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int validation_result = 1;
  if (rank == 0) {
    validation_result = ValidateInputData(GetInput());
  }

  MPI_Bcast(&validation_result, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return validation_result != 0;
}

int FedoseevGaussianMethodHorizontalStripSchemeMPI::ValidateInputData(const InType &input_data) {
  if (input_data.empty()) {
    return 0;
  }

  const size_t n = input_data.size();
  const size_t cols = input_data[0].size();
  if (cols < n + 1) {
    return 0;
  }

  for (size_t i = 1; i < n; ++i) {
    if (input_data[i].size() != cols) {
      return 0;
    }
  }

  return 1;
}

bool FedoseevGaussianMethodHorizontalStripSchemeMPI::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

bool FedoseevGaussianMethodHorizontalStripSchemeMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  InType augmented_matrix;
  size_t n = 0;
  size_t cols = 0;

  if (rank == 0) {
    augmented_matrix = GetInput();
    if (!augmented_matrix.empty()) {
      n = augmented_matrix.size();
      cols = augmented_matrix[0].size();
    }
  }

  int n_int = static_cast<int>(n);
  int cols_int = static_cast<int>(cols);
  MPI_Bcast(&n_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols_int, 1, MPI_INT, 0, MPI_COMM_WORLD);

  n = static_cast<size_t>(n_int);
  cols = static_cast<size_t>(cols_int);

  if (n == 0 || cols < n + 1) {
    GetOutput() = std::vector<double>();
    return false;
  }

  InType local_matrix;
  std::vector<int> global_to_local_map(n, -1);

  DistributeRows(augmented_matrix, n, cols, rank, size, local_matrix, global_to_local_map);

  if (!ForwardEliminationMPI(local_matrix, global_to_local_map, n, cols, rank, size)) {
    return false;
  }

  GetOutput() = BackwardSubstitutionMPI(local_matrix, global_to_local_map, n, cols, rank, size);
  return true;
}

void FedoseevGaussianMethodHorizontalStripSchemeMPI::DistributeRows(const InType &matrix, size_t n, size_t cols,
                                                                    int rank, int size, InType &local_matrix,
                                                                    std::vector<int> &global_to_local_map) {
  int local_row_count = 0;
  for (size_t i = 0; i < n; ++i) {
    if (static_cast<int>(i) % size == rank) {
      ++local_row_count;
    }
  }

  if (local_row_count <= 0) {
    local_matrix.clear();
    return;
  }

  local_matrix.resize(static_cast<size_t>(local_row_count));
  for (auto &row : local_matrix) {
    row.resize(cols);
  }

  int local_idx = 0;

  for (size_t i = 0; i < n; ++i) {
    if (static_cast<int>(i) % size == rank) {
      global_to_local_map[i] = local_idx;
      if (rank == 0) {
        local_matrix[static_cast<size_t>(local_idx)] = matrix[i];
      } else {
        MPI_Recv(local_matrix[static_cast<size_t>(local_idx)].data(), static_cast<int>(cols), MPI_DOUBLE, 0,
                 static_cast<int>(i), MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
      ++local_idx;
    } else if (rank == 0) {
      MPI_Send(matrix[i].data(), static_cast<int>(cols), MPI_DOUBLE, static_cast<int>(i) % size, static_cast<int>(i),
               MPI_COMM_WORLD);
    }
  }
}

bool FedoseevGaussianMethodHorizontalStripSchemeMPI::ForwardEliminationMPI(InType &local_matrix,
                                                                           const std::vector<int> &global_to_local_map,
                                                                           size_t n, size_t cols, int rank, int size) {
  for (size_t k = 0; k < n; ++k) {
    const int pivot_owner = static_cast<int>(k) % size;

    std::vector<double> pivot_row(cols);
    if (rank == pivot_owner) {
      int local_k = global_to_local_map[k];
      if (local_k >= 0 && static_cast<size_t>(local_k) < local_matrix.size()) {
        pivot_row = local_matrix[static_cast<size_t>(local_k)];
      }
    }

    MPI_Bcast(pivot_row.data(), static_cast<int>(cols), MPI_DOUBLE, pivot_owner, MPI_COMM_WORLD);

    if (std::abs(pivot_row[k]) < 1e-10) {
      return false;
    }

    EliminateRowsMPI(local_matrix, global_to_local_map, k, n, cols, pivot_row);
  }
  return true;
}

void FedoseevGaussianMethodHorizontalStripSchemeMPI::EliminateRowsMPI(InType &local_matrix,
                                                                      const std::vector<int> &global_to_local_map,
                                                                      size_t pivot_idx, size_t n, size_t cols,
                                                                      const std::vector<double> &pivot_row) {
  for (size_t i = 0; i < local_matrix.size(); ++i) {
    const size_t global_index = GetGlobalIndex(global_to_local_map, i, n);
    if (global_index > pivot_idx && global_index < n && std::abs(local_matrix[i][pivot_idx]) > 1e-10) {
      const double factor = local_matrix[i][pivot_idx] / pivot_row[pivot_idx];
      for (size_t j = pivot_idx; j < cols; ++j) {
        local_matrix[i][j] -= factor * pivot_row[j];
      }
    }
  }
}

size_t FedoseevGaussianMethodHorizontalStripSchemeMPI::GetGlobalIndex(const std::vector<int> &global_to_local_map,
                                                                      size_t local_idx, size_t n) {
  const int local_idx_int = static_cast<int>(local_idx);
  for (size_t i = 0; i < n; ++i) {
    if (global_to_local_map[i] >= 0 && global_to_local_map[i] == local_idx_int) {
      return i;
    }
  }
  return n;
}

std::vector<double> FedoseevGaussianMethodHorizontalStripSchemeMPI::BackwardSubstitutionMPI(
    const InType &local_matrix, const std::vector<int> &global_to_local_map, size_t n, size_t cols, int rank,
    int size) {
  std::vector<double> solution(n, 0.0);

  for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
    double sum = 0.0;
    const int row_owner = i % size;

    if (rank == row_owner) {
      const int local_index = global_to_local_map[static_cast<size_t>(i)];
      if (local_index >= 0 && static_cast<size_t>(local_index) < local_matrix.size()) {
        for (size_t j = static_cast<size_t>(i) + 1; j < n; ++j) {
          sum += local_matrix[static_cast<size_t>(local_index)][j] * solution[j];
        }
        solution[static_cast<size_t>(i)] = (local_matrix[static_cast<size_t>(local_index)][cols - 1] - sum) /
                                           local_matrix[static_cast<size_t>(local_index)][static_cast<size_t>(i)];
      }
    }

    MPI_Bcast(&solution[static_cast<size_t>(i)], 1, MPI_DOUBLE, row_owner, MPI_COMM_WORLD);
  }

  return solution;
}

bool FedoseevGaussianMethodHorizontalStripSchemeMPI::PostProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  return rank == 0 ? !GetOutput().empty() : true;
}

}  // namespace fedoseev_gaussian_method_horizontal_strip_scheme
