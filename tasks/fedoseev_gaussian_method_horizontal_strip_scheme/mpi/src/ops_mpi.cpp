#include "fedoseev_gaussian_method_horizontal_strip_scheme/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "fedoseev_gaussian_method_horizontal_strip_scheme/common/include/common.hpp"

namespace fedoseev_gaussian_method_horizontal_strip_scheme {

FedoseevTestTaskMPI::FedoseevTestTaskMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<double>();
}

bool FedoseevTestTaskMPI::ValidationImpl() {
  const InType &augmented_matrix = GetInput();
  size_t n = augmented_matrix.size();

  if (n == 0) {
    return false;
  }

  return std::all_of(augmented_matrix.begin(), augmented_matrix.end(),
                     [n](const auto &row) { return row.size() == n + 1; });  // NOLINT(modernize-use-ranges)
}

bool FedoseevTestTaskMPI::PreProcessingImpl() {
  const InType &augmented_matrix = GetInput();
  size_t n = augmented_matrix.size();

  for (size_t i = 0; i < n; ++i) {
    if (std::abs(augmented_matrix[i][i]) < 1e-10) {
      bool found = false;
      for (size_t j = i + 1; j < n; ++j) {
        if (std::abs(augmented_matrix[j][i]) > 1e-10) {
          found = true;
          break;
        }
      }
      if (!found) {
        return false;
      }
    }
  }

  return true;
}

bool FedoseevTestTaskMPI::RunImpl() {
  const InType &full_matrix = GetInput();
  size_t n = full_matrix.size();

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (n > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return false;
  }
  int n_int = static_cast<int>(n);

  int rows_per_process = n_int / size;
  int remainder = n_int % size;
  int start_row = (rank * rows_per_process) + std::min(rank, remainder);
  int end_row = start_row + rows_per_process + (rank < remainder ? 1 : 0);
  int local_rows = end_row - start_row;

  std::vector<std::vector<double>> local_matrix(local_rows, std::vector<double>(n_int + 1));

  for (int i = 0; i < local_rows; ++i) {
    local_matrix[i] = full_matrix[start_row + i];
  }

  std::vector<double> pivot_row(n_int + 1);

  for (int k = 0; k < n_int; ++k) {
    int owner_of_k = 0;
    for (int proc = 0; proc < size; ++proc) {
      int p_start = (proc * rows_per_process) + std::min(proc, remainder);
      int p_end = p_start + rows_per_process + (proc < remainder ? 1 : 0);
      if (k >= p_start && k < p_end) {
        owner_of_k = proc;
        break;
      }
    }

    if (owner_of_k == rank) {
      int local_k = k - start_row;
      pivot_row = local_matrix[local_k];
    }

    MPI_Bcast(pivot_row.data(), n_int + 1, MPI_DOUBLE, owner_of_k, MPI_COMM_WORLD);
    for (int i = 0; i < local_rows; ++i) {
      int global_i = start_row + i;
      if (global_i > k) {
        double factor = local_matrix[i][k] / pivot_row[k];
        for (int j = k; j < n_int + 1; ++j) {
          local_matrix[i][j] -= factor * pivot_row[j];
        }
      }
    }
  }

  std::vector<int> recvcounts(size);
  std::vector<int> displs(size);

  for (int proc = 0; proc < size; ++proc) {
    int p_start = (proc * rows_per_process) + std::min(proc, remainder);
    int p_end = p_start + rows_per_process + (proc < remainder ? 1 : 0);
    recvcounts[proc] = (p_end - p_start) * (n_int + 1);
    displs[proc] = (proc == 0) ? 0 : displs[proc - 1] + recvcounts[proc - 1];
  }

  std::vector<double> gathered_data;
  std::vector<double> send_buffer;

  for (const auto &row : local_matrix) {
    send_buffer.insert(send_buffer.end(), row.begin(), row.end());
  }

  if (rank == 0) {
    gathered_data.resize(static_cast<size_t>(n_int) * (n_int + 1));
  }

  MPI_Gatherv(send_buffer.data(), local_rows * (n_int + 1), MPI_DOUBLE, gathered_data.data(), recvcounts.data(),
              displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  std::vector<double> x(static_cast<size_t>(n_int), 0.0);
  if (rank == 0) {
    std::vector<std::vector<double>> full_triangular_matrix(static_cast<size_t>(n_int),
                                                            std::vector<double>(static_cast<size_t>(n_int) + 1));

    int idx = 0;
    for (int proc = 0; proc < size; ++proc) {
      int p_start = (proc * rows_per_process) + std::min(proc, remainder);
      int p_end = p_start + rows_per_process + (proc < remainder ? 1 : 0);
      int p_rows = p_end - p_start;

      for (int i = 0; i < p_rows; ++i) {
        for (int j = 0; j < n_int + 1; ++j) {
          full_triangular_matrix[p_start + i][j] = gathered_data[idx++];
        }
      }
    }

    for (int i = n_int - 1; i >= 0; --i) {
      x[static_cast<size_t>(i)] = full_triangular_matrix[i][n_int];
      for (int j = i + 1; j < n_int; ++j) {
        x[static_cast<size_t>(i)] -= full_triangular_matrix[i][j] * x[static_cast<size_t>(j)];
      }
      x[static_cast<size_t>(i)] /= full_triangular_matrix[i][i];
    }
  }

  MPI_Bcast(x.data(), n_int, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  GetOutput() = x;
  return !GetOutput().empty();
}

bool FedoseevTestTaskMPI::PostProcessingImpl() {
  const InType &augmented_matrix = GetInput();
  const auto &x = GetOutput();
  size_t n = augmented_matrix.size();

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    double residual = 0.0;
    for (size_t i = 0; i < n; ++i) {
      double sum = 0.0;
      for (size_t j = 0; j < n; ++j) {
        sum += augmented_matrix[i][j] * x[j];
      }
      residual += std::abs(sum - augmented_matrix[i][n]);
    }
    return residual < 1e-6 * static_cast<double>(n);
  }

  return true;
}

}  // namespace fedoseev_gaussian_method_horizontal_strip_scheme
