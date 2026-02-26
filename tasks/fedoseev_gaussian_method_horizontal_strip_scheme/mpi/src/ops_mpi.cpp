#include "fedoseev_gaussian_method_horizontal_strip_scheme/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "fedoseev_gaussian_method_horizontal_strip_scheme/common/include/common.hpp"
#include "util/include/util.hpp"

namespace fedoseev_gaussian_method_horizontal_strip_scheme {

FedoseevTestTaskMPI::FedoseevTestTaskMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<double>();
}

bool FedoseevTestTaskMPI::ValidationImpl() {
  const InType &augmented_matrix = GetInput();
  int n = augmented_matrix.size();

  if (n == 0) {
    return false;
  }

  for (const auto &row : augmented_matrix) {
    if (static_cast<int>(row.size()) != n + 1) {
      return false;
    }
  }

  return true;
}

bool FedoseevTestTaskMPI::PreProcessingImpl() {
  const InType &augmented_matrix = GetInput();
  int n = augmented_matrix.size();

  for (int i = 0; i < n; ++i) {
    if (std::abs(augmented_matrix[i][i]) < 1e-10) {
      bool found = false;
      for (int j = i + 1; j < n; ++j) {
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
  int n = full_matrix.size();

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int rows_per_process = n / size;
  int remainder = n % size;
  int start_row = rank * rows_per_process + std::min(rank, remainder);
  int end_row = start_row + rows_per_process + (rank < remainder ? 1 : 0);
  int local_rows = end_row - start_row;

  std::vector<std::vector<double>> local_matrix(local_rows, std::vector<double>(n + 1));

  for (int i = 0; i < local_rows; ++i) {
    local_matrix[i] = full_matrix[start_row + i];
  }

  std::vector<double> pivot_row(n + 1);

  for (int k = 0; k < n; ++k) {
    int owner_of_k = 0;
    for (int p = 0; p < size; ++p) {
      int p_start = p * rows_per_process + std::min(p, remainder);
      int p_end = p_start + rows_per_process + (p < remainder ? 1 : 0);
      if (k >= p_start && k < p_end) {
        owner_of_k = p;
        break;
      }
    }

    if (owner_of_k == rank) {
      int local_k = k - start_row;

      pivot_row = local_matrix[local_k];
    }

    MPI_Bcast(pivot_row.data(), n + 1, MPI_DOUBLE, owner_of_k, MPI_COMM_WORLD);
    for (int i = 0; i < local_rows; ++i) {
      int global_i = start_row + i;
      if (global_i > k) {
        double factor = local_matrix[i][k] / pivot_row[k];
        for (int j = k; j < n + 1; ++j) {
          local_matrix[i][j] -= factor * pivot_row[j];
        }
      }
    }
  }

  std::vector<int> recvcounts(size);
  std::vector<int> displs(size);

  for (int p = 0; p < size; ++p) {
    int p_start = p * rows_per_process + std::min(p, remainder);
    int p_end = p_start + rows_per_process + (p < remainder ? 1 : 0);
    recvcounts[p] = (p_end - p_start) * (n + 1);
    displs[p] = (p == 0) ? 0 : displs[p - 1] + recvcounts[p - 1];
  }

  std::vector<double> gathered_data;
  std::vector<double> send_buffer;

  for (const auto &row : local_matrix) {
    send_buffer.insert(send_buffer.end(), row.begin(), row.end());
  }

  if (rank == 0) {
    gathered_data.resize(n * (n + 1));
  }

  MPI_Gatherv(send_buffer.data(), local_rows * (n + 1), MPI_DOUBLE, gathered_data.data(), recvcounts.data(),
              displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  std::vector<double> x(n, 0.0);
  if (rank == 0) {
    std::vector<std::vector<double>> full_triangular_matrix(n, std::vector<double>(n + 1));

    int idx = 0;
    for (int p = 0; p < size; ++p) {
      int p_start = p * rows_per_process + std::min(p, remainder);
      int p_end = p_start + rows_per_process + (p < remainder ? 1 : 0);
      int p_rows = p_end - p_start;

      for (int i = 0; i < p_rows; ++i) {
        for (int j = 0; j < n + 1; ++j) {
          full_triangular_matrix[p_start + i][j] = gathered_data[idx++];
        }
      }
    }

    for (int i = n - 1; i >= 0; --i) {
      x[i] = full_triangular_matrix[i][n];
      for (int j = i + 1; j < n; ++j) {
        x[i] -= full_triangular_matrix[i][j] * x[j];
      }
      x[i] /= full_triangular_matrix[i][i];
    }
  }

  MPI_Bcast(x.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  GetOutput() = x;
  return !GetOutput().empty();
}

bool FedoseevTestTaskMPI::PostProcessingImpl() {
  const InType &augmented_matrix = GetInput();
  const auto &x = GetOutput();
  int n = augmented_matrix.size();

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    double residual = 0.0;
    for (int i = 0; i < n; ++i) {
      double sum = 0.0;
      for (int j = 0; j < n; ++j) {
        sum += augmented_matrix[i][j] * x[j];
      }
      residual += std::abs(sum - augmented_matrix[i][n]);
    }
    return residual < 1e-6 * n;
  }

  return true;
}

}  // namespace fedoseev_gaussian_method_horizontal_strip_scheme
