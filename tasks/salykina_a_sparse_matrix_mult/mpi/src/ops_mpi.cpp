#include "salykina_a_sparse_matrix_mult/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <vector>

#include "salykina_a_sparse_matrix_mult/common/include/common.hpp"

namespace salykina_a_sparse_matrix_mult {

SalykinaASparseMatrixMultMPI::SalykinaASparseMatrixMultMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = SparseMatrixCRS{};
}

bool SalykinaASparseMatrixMultMPI::ValidationImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &input = GetInput();
  if (input.matrix_a.num_cols != input.matrix_b.num_rows) {
    if (rank == 0) {
      return false;
    }
    return false;
  }

  if (input.matrix_a.row_ptr.size() != static_cast<size_t>(static_cast<size_t>(input.matrix_a.num_rows) + 1U)) {
    return false;
  }
  if (input.matrix_b.row_ptr.size() != static_cast<size_t>(static_cast<size_t>(input.matrix_b.num_rows) + 1U)) {
    return false;
  }
  if (input.matrix_a.values.size() != input.matrix_a.col_indices.size()) {
    return false;
  }
  if (input.matrix_b.values.size() != input.matrix_b.col_indices.size()) {
    return false;
  }
  return true;
}

bool SalykinaASparseMatrixMultMPI::PreProcessingImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &input = GetInput();
  auto &output = GetOutput();
  output.num_rows = input.matrix_a.num_rows;
  output.num_cols = input.matrix_b.num_cols;
  output.row_ptr.resize(output.num_rows + 1, 0);

  return true;
}

bool SalykinaASparseMatrixMultMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  MultiplyHorizontalScheme();
  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

void SalykinaASparseMatrixMultMPI::MultiplyHorizontalScheme() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &input = GetInput();
  const auto &a = input.matrix_a;
  const auto &b = input.matrix_b;

  int rows_per_process = a.num_rows / size;
  int remainder = a.num_rows % size;
  int local_start_row = (rank * rows_per_process) + std::min(rank, remainder);
  int local_end_row = local_start_row + rows_per_process + (rank < remainder ? 1 : 0);
  int local_num_rows = local_end_row - local_start_row;

  std::vector<double> b_values;
  std::vector<int> b_col_indices;
  std::vector<int> b_row_ptr;
  int b_num_rows = 0;
  int b_num_cols = 0;

  BroadcastMatrixB(rank, size, b, b_values, b_col_indices, b_row_ptr, b_num_rows, b_num_cols);

  std::vector<double> local_values;
  std::vector<int> local_col_indices;
  std::vector<int> local_row_ptr(local_num_rows + 1, 0);

  ComputeLocalRows(a, b_values, b_col_indices, b_row_ptr, b_num_cols, local_start_row, local_num_rows, local_values,
                   local_col_indices, local_row_ptr);

  GatherResults(local_values, local_col_indices, local_row_ptr, local_start_row, local_num_rows);
}

void SalykinaASparseMatrixMultMPI::GatherResults(const std::vector<double> &local_values,
                                                 const std::vector<int> &local_col_indices,
                                                 const std::vector<int> &local_row_ptr,
                                                 [[maybe_unused]] int local_start_row, int local_num_rows) {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  auto &output = GetOutput();
  int local_nnz = static_cast<int>(local_values.size());
  std::vector<int> nnz_counts(size);
  std::vector<int> row_counts(size);

  MPI_Allgather(&local_nnz, 1, MPI_INT, nnz_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
  MPI_Allgather(&local_num_rows, 1, MPI_INT, row_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

  if (rank == 0) {
    AssembleResultsOnRank0(rank, size, local_values, local_col_indices, local_row_ptr, local_num_rows, nnz_counts,
                           row_counts);
  } else {
    std::vector<int> row_ptr_copy = local_row_ptr;
    std::vector<double> values_copy = local_values;
    std::vector<int> col_indices_copy = local_col_indices;

    MPI_Send(row_ptr_copy.data(), local_num_rows + 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    MPI_Send(values_copy.data(), local_nnz, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
    MPI_Send(col_indices_copy.data(), local_nnz, MPI_INT, 0, 2, MPI_COMM_WORLD);
  }

  int final_nnz = static_cast<int>(output.values.size());
  MPI_Bcast(&final_nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank != 0) {
    output.values.resize(final_nnz);
    output.col_indices.resize(final_nnz);
  }

  MPI_Bcast(output.values.data(), final_nnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(output.col_indices.data(), final_nnz, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(output.row_ptr.data(), output.num_rows + 1, MPI_INT, 0, MPI_COMM_WORLD);
}

bool SalykinaASparseMatrixMultMPI::PostProcessingImpl() {
  auto &output = GetOutput();
  if (output.values.size() != output.col_indices.size()) {
    return false;
  }
  if (output.row_ptr.size() != static_cast<size_t>(static_cast<size_t>(output.num_rows) + 1U)) {
    return false;
  }
  return true;
}

void SalykinaASparseMatrixMultMPI::BroadcastMatrixB(int rank, [[maybe_unused]] int size, const SparseMatrixCRS &b,
                                                    std::vector<double> &b_values, std::vector<int> &b_col_indices,
                                                    std::vector<int> &b_row_ptr, int &b_num_rows, int &b_num_cols) {
  int b_size = static_cast<int>(b.values.size());
  int b_row_ptr_size = static_cast<int>(b.row_ptr.size());
  int b_col_indices_size = static_cast<int>(b.col_indices.size());

  if (rank == 0) {
    b_values = b.values;
    b_col_indices = b.col_indices;
    b_row_ptr = b.row_ptr;
    b_num_rows = b.num_rows;
    b_num_cols = b.num_cols;
  } else {
    b_values.resize(b_size);
    b_col_indices.resize(b_col_indices_size);
    b_row_ptr.resize(b_row_ptr_size);
  }

  MPI_Bcast(&b_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&b_row_ptr_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&b_col_indices_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    b_values.resize(b_size);
    b_col_indices.resize(b_col_indices_size);
    b_row_ptr.resize(b_row_ptr_size);
  }
  MPI_Bcast(b_values.data(), b_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(b_col_indices.data(), b_col_indices_size, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(b_row_ptr.data(), b_row_ptr_size, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&b_num_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&b_num_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

void SalykinaASparseMatrixMultMPI::ComputeLocalRows(
    const SparseMatrixCRS &a, const std::vector<double> &b_values, const std::vector<int> &b_col_indices,
    const std::vector<int> &b_row_ptr, int b_num_cols, int local_start_row, int local_num_rows,
    std::vector<double> &local_values, std::vector<int> &local_col_indices, std::vector<int> &local_row_ptr) {
  std::vector<double> row_result(b_num_cols, 0.0);
  for (int local_row = 0; local_row < local_num_rows; local_row++) {
    int global_row = local_start_row + local_row;
    std::ranges::fill(row_result, 0.0);

    int row_start = a.row_ptr[global_row];
    int row_end = a.row_ptr[global_row + 1];

    for (int k = row_start; k < row_end; k++) {
      int col_a = a.col_indices[k];
      double val_a = a.values[k];

      int b_row_start = b_row_ptr[col_a];
      int b_row_end = b_row_ptr[col_a + 1];

      for (int j = b_row_start; j < b_row_end; j++) {
        int col_b = b_col_indices[j];
        double val_b = b_values[j];
        row_result[col_b] += val_a * val_b;
      }
    }

    for (int j = 0; j < b_num_cols; j++) {
      if (row_result[j] != 0.0) {
        local_values.push_back(row_result[j]);
        local_col_indices.push_back(j);
        local_row_ptr[local_row + 1]++;
      }
    }
  }

  for (int i = 0; i < local_num_rows; i++) {
    local_row_ptr[i + 1] += local_row_ptr[i];
  }
}

void SalykinaASparseMatrixMultMPI::AssembleResultsOnRank0(int rank, int size, const std::vector<double> &local_values,
                                                          const std::vector<int> &local_col_indices,
                                                          const std::vector<int> &local_row_ptr,
                                                          [[maybe_unused]] int local_num_rows,
                                                          const std::vector<int> &nnz_counts,
                                                          const std::vector<int> &row_counts) {
  auto &output = GetOutput();
  output.values.clear();
  output.col_indices.clear();
  output.row_ptr.assign(output.num_rows + 1, 0);

  for (int proc = 0; proc < size; proc++) {
    int proc_start_row = (proc * (output.num_rows / size)) + std::min(proc, output.num_rows % size);
    int proc_num_rows = row_counts[proc];
    std::vector<int> proc_row_ptr(proc_num_rows + 1);

    if (proc == rank) {
      proc_row_ptr = local_row_ptr;
    } else {
      MPI_Recv(proc_row_ptr.data(), proc_num_rows + 1, MPI_INT, proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    int proc_nnz = nnz_counts[proc];
    std::vector<double> proc_values(proc_nnz);
    std::vector<int> proc_col_indices(proc_nnz);

    if (proc == rank) {
      proc_values = local_values;
      proc_col_indices = local_col_indices;
    } else {
      MPI_Recv(proc_values.data(), proc_nnz, MPI_DOUBLE, proc, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(proc_col_indices.data(), proc_nnz, MPI_INT, proc, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    for (int i = 0; i < proc_num_rows; i++) {
      int global_row = proc_start_row + i;
      int local_start = proc_row_ptr[i];
      int local_end = proc_row_ptr[i + 1];

      output.row_ptr[global_row + 1] = local_end - local_start;

      for (int j = local_start; j < local_end; j++) {
        output.values.push_back(proc_values[j]);
        output.col_indices.push_back(proc_col_indices[j]);
      }
    }
  }

  for (int i = 0; i < output.num_rows; i++) {
    output.row_ptr[i + 1] += output.row_ptr[i];
  }
}

}  // namespace salykina_a_sparse_matrix_mult
