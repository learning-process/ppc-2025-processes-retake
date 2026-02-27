#include "zyuzin_n_multiplication_matrix_horiz/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <array>
#include <cstddef>
#include <vector>

#include "zyuzin_n_multiplication_matrix_horiz/common/include/common.hpp"

namespace zyuzin_n_multiplication_matrix_horiz {

ZyuzinNMultiplicationMatrixMPI::ZyuzinNMultiplicationMatrixMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool ZyuzinNMultiplicationMatrixMPI::ValidationImpl() {
  const auto &matrix_a = GetInput().first;
  const auto &matrix_b = GetInput().second;

  if (matrix_a.empty() || matrix_b.empty()) {
    return false;
  }

  size_t cols_a = matrix_a[0].size();
  for (size_t i = 1; i < matrix_a.size(); i++) {
    if (matrix_a[i].size() != cols_a) {
      return false;
    }
  }

  size_t cols_b = matrix_b[0].size();
  for (size_t i = 1; i < matrix_b.size(); i++) {
    if (matrix_b[i].size() != cols_b) {
      return false;
    }
  }

  return cols_a == matrix_b.size();
}

bool ZyuzinNMultiplicationMatrixMPI::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

void ZyuzinNMultiplicationMatrixMPI::BroadcastMatricesInfo(int rank, size_t &rows_a, size_t &cols_a, size_t &rows_b,
                                                           size_t &cols_b, std::vector<double> &matrix_b_flat) {
  std::array<int, 4> sizes = {0, 0, 0, 0};
  if (rank == 0) {
    const auto &matrix_a = GetInput().first;
    const auto &matrix_b = GetInput().second;
    sizes[0] = static_cast<int>(matrix_a.size());
    sizes[1] = static_cast<int>(matrix_a[0].size());
    sizes[2] = static_cast<int>(matrix_b.size());
    sizes[3] = static_cast<int>(matrix_b[0].size());

    matrix_b_flat.resize(static_cast<size_t>(sizes[2]) * static_cast<size_t>(sizes[3]));
    for (int i = 0; i < sizes[2]; i++) {
      for (int j = 0; j < sizes[3]; j++) {
        matrix_b_flat[(i * sizes[3]) + j] = matrix_b[i][j];
      }
    }
  }

  MPI_Bcast(sizes.data(), 4, MPI_INT, 0, MPI_COMM_WORLD);
  rows_a = sizes[0];
  cols_a = sizes[1];
  rows_b = sizes[2];
  cols_b = sizes[3];

  if (rank != 0) {
    matrix_b_flat.resize(rows_b * cols_b);
  }
  MPI_Bcast(matrix_b_flat.data(), static_cast<int>(rows_b * cols_b), MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void ZyuzinNMultiplicationMatrixMPI::ScatterMatrixA(int rank, int size, size_t rows_a, size_t cols_a,
                                                    std::vector<double> &matrix_a_flat,
                                                    std::vector<double> &local_a_flat, int &actual_local_rows) {
  int base_rows = static_cast<int>(rows_a / size);
  int extra = static_cast<int>(rows_a % size);
  std::vector<int> rows_per_process(size);
  for (int i = 0; i < size; ++i) {
    rows_per_process[i] = base_rows + (i < extra ? 1 : 0);
  }
  actual_local_rows = rows_per_process[rank];

  if (rank == 0) {
    matrix_a_flat.resize(rows_a * cols_a);
    const auto &matrix_a = GetInput().first;
    for (size_t i = 0; i < rows_a; i++) {
      for (size_t j = 0; j < cols_a; j++) {
        matrix_a_flat[(i * cols_a) + j] = matrix_a[i][j];
      }
    }
  }

  std::vector<int> send_counts(size, 0);
  std::vector<int> displacements(size, 0);
  int offset = 0;
  for (int i = 0; i < size; i++) {
    send_counts[i] = static_cast<int>(rows_per_process[i] * cols_a);
    displacements[i] = offset;
    offset += send_counts[i];
  }

  local_a_flat.resize(static_cast<std::size_t>(actual_local_rows) * static_cast<std::size_t>(cols_a));
  MPI_Scatterv(matrix_a_flat.data(), send_counts.data(), displacements.data(), MPI_DOUBLE, local_a_flat.data(),
               static_cast<int>(actual_local_rows * cols_a), MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void ZyuzinNMultiplicationMatrixMPI::ComputeLocalMultiplication(const std::vector<double> &local_a_flat,
                                                                const std::vector<double> &matrix_b_flat,
                                                                std::vector<double> &local_result_flat,
                                                                int actual_local_rows, size_t cols_a, size_t cols_b) {
  local_result_flat.resize(actual_local_rows * cols_b, 0);
  for (int i = 0; i < actual_local_rows; i++) {
    for (size_t j = 0; j < cols_b; j++) {
      for (size_t k = 0; k < cols_a; k++) {
        local_result_flat[(i * cols_b) + j] += local_a_flat[(i * cols_a) + k] * matrix_b_flat[(k * cols_b) + j];
      }
    }
  }
}

void ZyuzinNMultiplicationMatrixMPI::GatherAndConvertResults(int size, size_t rows_a, size_t cols_b,
                                                             int actual_local_rows,
                                                             const std::vector<double> &local_result_flat) {
  int base_rows = static_cast<int>(rows_a / size);
  int extra = static_cast<int>(rows_a % size);
  std::vector<int> rows_per_process(size);
  for (int i = 0; i < size; ++i) {
    rows_per_process[i] = base_rows + (i < extra ? 1 : 0);
  }

  std::vector<int> result_send_counts(size, 0);
  std::vector<int> result_displacements(size, 0);
  int offset = 0;
  for (int i = 0; i < size; i++) {
    result_send_counts[i] = static_cast<int>(rows_per_process[i] * cols_b);
    result_displacements[i] = offset;
    offset += result_send_counts[i];
  }

  std::vector<double> all_result_flat(rows_a * cols_b, 0);
  MPI_Allgatherv(local_result_flat.data(), static_cast<int>(actual_local_rows * cols_b), MPI_DOUBLE,
                 all_result_flat.data(), result_send_counts.data(), result_displacements.data(), MPI_DOUBLE,
                 MPI_COMM_WORLD);

  GetOutput().resize(rows_a);
  for (size_t i = 0; i < rows_a; i++) {
    GetOutput()[i].resize(cols_b);
    for (size_t j = 0; j < cols_b; j++) {
      GetOutput()[i][j] = all_result_flat[(i * cols_b) + j];
    }
  }
}

bool ZyuzinNMultiplicationMatrixMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  size_t rows_a = 0;
  size_t cols_a = 0;
  size_t rows_b = 0;
  size_t cols_b = 0;
  std::vector<double> matrix_b_flat;

  BroadcastMatricesInfo(rank, rows_a, cols_a, rows_b, cols_b, matrix_b_flat);

  std::vector<double> matrix_a_flat;
  std::vector<double> local_a_flat;
  int actual_local_rows = 0;
  ScatterMatrixA(rank, size, rows_a, cols_a, matrix_a_flat, local_a_flat, actual_local_rows);

  std::vector<double> local_result_flat;
  ComputeLocalMultiplication(local_a_flat, matrix_b_flat, local_result_flat, actual_local_rows, cols_a, cols_b);

  GatherAndConvertResults(size, rows_a, cols_b, actual_local_rows, local_result_flat);

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool ZyuzinNMultiplicationMatrixMPI::PostProcessingImpl() {
  return true;
}

}  // namespace zyuzin_n_multiplication_matrix_horiz
