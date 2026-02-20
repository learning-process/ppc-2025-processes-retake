#include "sokolov_k_gauss_jordan/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "sokolov_k_gauss_jordan/common/include/common.hpp"

namespace sokolov_k_gauss_jordan {

namespace {

void GenerateSystem(int n, std::vector<double> &matrix) {
  int cols = n + 1;
  matrix.assign(static_cast<size_t>(n) * cols, 0.0);
  unsigned int seed = static_cast<unsigned int>(n) * 1234567U;
  auto gen = [&seed]() -> double {
    seed = (seed * 1103515245U) + 12345U;
    return static_cast<double>(((seed >> 16) % 100) + 1);
  };
  for (int i = 0; i < n; i++) {
    double row_sum = 0.0;
    for (int j = 0; j < n; j++) {
      if (i != j) {
        matrix[(i * cols) + j] = gen() / 100.0;
        row_sum += std::abs(matrix[(i * cols) + j]);
      }
    }
    matrix[(i * cols) + i] = row_sum + gen();
  }
  for (int i = 0; i < n; i++) {
    double b_val = 0.0;
    for (int j = 0; j < n; j++) {
      b_val += matrix[(i * cols) + j] * (j + 1.0);
    }
    matrix[(i * cols) + n] = b_val;
  }
}

void ComputeDistribution(int n, int cols, int world_size, std::vector<int> &rows_per_proc,
                         std::vector<int> &row_offsets, std::vector<int> &sendcounts, std::vector<int> &displs) {
  rows_per_proc.resize(world_size);
  row_offsets.assign(world_size, 0);
  sendcounts.resize(world_size);
  displs.resize(world_size);
  for (int i = 0; i < world_size; i++) {
    rows_per_proc[i] = (n / world_size) + (i < (n % world_size) ? 1 : 0);
    if (i > 0) {
      row_offsets[i] = row_offsets[i - 1] + rows_per_proc[i - 1];
    }
    sendcounts[i] = rows_per_proc[i] * cols;
    displs[i] = row_offsets[i] * cols;
  }
}

int FindRowOwner(int k, const std::vector<int> &row_offsets, const std::vector<int> &rows_per_proc, int world_size) {
  for (int pp = 0; pp < world_size; pp++) {
    if (k < row_offsets[pp] + rows_per_proc[pp]) {
      return pp;
    }
  }
  return world_size - 1;
}

void EliminationStep(std::vector<double> &local_data, std::vector<double> &pivot_row, int local_rows, int local_start,
                     int cols, int k, int rank, const std::vector<int> &row_offsets,
                     const std::vector<int> &rows_per_proc, int world_size) {
  int owner = FindRowOwner(k, row_offsets, rows_per_proc, world_size);

  if (rank == owner) {
    int li = k - local_start;
    double pivot_val = local_data[(li * cols) + k];
    for (int j = 0; j < cols; j++) {
      local_data[(li * cols) + j] /= pivot_val;
      pivot_row[j] = local_data[(li * cols) + j];
    }
  }

  MPI_Bcast(pivot_row.data(), cols, MPI_DOUBLE, owner, MPI_COMM_WORLD);

  for (int i = 0; i < local_rows; i++) {
    int gi = local_start + i;
    if (gi != k) {
      double factor = local_data[(i * cols) + k];
      for (int j = 0; j < cols; j++) {
        local_data[(i * cols) + j] -= factor * pivot_row[j];
      }
    }
  }
}

void CollectResults(std::vector<double> &local_data, std::vector<double> &matrix, int local_rows, int cols,
                    const std::vector<int> &sendcounts, const std::vector<int> &displs, int world_size, int rank) {
  if (rank != 0) {
    if (local_rows > 0) {
      MPI_Send(local_data.data(), local_rows * cols, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    return;
  }
  for (int i = 0; i < local_rows * cols; i++) {
    matrix[i] = local_data[i];
  }
  for (int pp = 1; pp < world_size; pp++) {
    if (sendcounts[pp] > 0) {
      MPI_Recv(matrix.data() + displs[pp], sendcounts[pp], MPI_DOUBLE, pp, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }
}

}  // namespace

SokolovKGaussJordanMPI::SokolovKGaussJordanMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool SokolovKGaussJordanMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int valid = 0;
  if (rank == 0) {
    valid = (GetInput() > 0 && GetOutput() == 0) ? 1 : 0;
  }
  MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return valid == 1;
}

bool SokolovKGaussJordanMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  n_ = GetInput();
  MPI_Bcast(&n_, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (n_ <= 0) {
    return false;
  }
  solution_.assign(n_, 0.0);
  if (rank == 0) {
    GenerateSystem(n_, matrix_);
    original_matrix_ = matrix_;
  }
  return true;
}

bool SokolovKGaussJordanMPI::RunImpl() {
  if (n_ <= 0) {
    return false;
  }

  int rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  if (rank == 0) {
    matrix_ = original_matrix_;
  }

  int cols = n_ + 1;
  std::vector<int> rows_per_proc;
  std::vector<int> row_offsets;
  std::vector<int> sendcounts;
  std::vector<int> displs;
  ComputeDistribution(n_, cols, world_size, rows_per_proc, row_offsets, sendcounts, displs);

  int local_rows = rows_per_proc[rank];
  int local_start = row_offsets[rank];
  std::vector<double> local_data(static_cast<size_t>(local_rows) * cols);

  MPI_Scatterv(matrix_.data(), sendcounts.data(), displs.data(), MPI_DOUBLE, local_data.data(), local_rows * cols,
               MPI_DOUBLE, 0, MPI_COMM_WORLD);

  std::vector<double> pivot_row(cols);
  for (int k = 0; k < n_; k++) {
    EliminationStep(local_data, pivot_row, local_rows, local_start, cols, k, rank, row_offsets, rows_per_proc,
                    world_size);
  }

  CollectResults(local_data, matrix_, local_rows, cols, sendcounts, displs, world_size, rank);

  if (rank == 0) {
    for (int i = 0; i < n_; i++) {
      solution_[i] = matrix_[(i * cols) + n_];
    }
  }

  return true;
}

bool SokolovKGaussJordanMPI::PostProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int valid = 1;
  if (rank == 0) {
    const double eps = 1e-6;
    for (int i = 0; i < n_; i++) {
      if (std::abs(solution_[i] - (i + 1.0)) > eps) {
        valid = 0;
        break;
      }
    }
  }
  MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (valid == 0) {
    return false;
  }
  GetOutput() = GetInput();
  return true;
}

}  // namespace sokolov_k_gauss_jordan
