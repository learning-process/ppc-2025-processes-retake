#include "salena_s_sparse_matrix_mult/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <vector>

namespace salena_s_sparse_matrix_mult {

namespace {
void CalculateCountsAndDisplacements(int size, int rows, std::vector<int> &send_counts, std::vector<int> &displs) {
  int delta_rows = rows / size;
  int rem_rows = rows % size;
  int cur_displ = 0;
  for (int i = 0; i < size; ++i) {
    send_counts[static_cast<std::size_t>(i)] = delta_rows + (i < rem_rows ? 1 : 0);
    displs[static_cast<std::size_t>(i)] = cur_displ;
    cur_displ += send_counts[static_cast<std::size_t>(i)];
  }
}

void MultiplyLocalSparse(int local_rows, int b_cols, int my_start_row, int my_start_idx,
                         const std::vector<int> &a_row_ptr_full, const std::vector<int> &local_a_cols,
                         const std::vector<double> &local_a_vals, const SparseMatrixCRS &B_local,
                         std::vector<double> &local_c_vals, std::vector<int> &local_c_cols,
                         std::vector<int> &local_c_row_ptr) {
  std::vector<int> marker(static_cast<std::size_t>(b_cols), -1);
  std::vector<double> temp_values(static_cast<std::size_t>(b_cols), 0.0);

  for (int i = 0; i < local_rows; ++i) {
    int row_nz = 0;
    std::vector<int> current_row_cols;

    int local_row_start = a_row_ptr_full[static_cast<std::size_t>(my_start_row + i)] - my_start_idx;
    int local_row_end = a_row_ptr_full[static_cast<std::size_t>(my_start_row + i + 1)] - my_start_idx;

    for (int j = local_row_start; j < local_row_end; ++j) {
      int a_col = local_a_cols[static_cast<std::size_t>(j)];
      double a_val = local_a_vals[static_cast<std::size_t>(j)];

      for (int k = B_local.row_ptr[static_cast<std::size_t>(a_col)];
           k < B_local.row_ptr[static_cast<std::size_t>(a_col + 1)]; ++k) {
        int b_col = B_local.col_indices[static_cast<std::size_t>(k)];
        double b_val = B_local.values[static_cast<std::size_t>(k)];

        if (marker[static_cast<std::size_t>(b_col)] != i) {
          marker[static_cast<std::size_t>(b_col)] = i;
          current_row_cols.push_back(b_col);
          temp_values[static_cast<std::size_t>(b_col)] = a_val * b_val;
        } else {
          temp_values[static_cast<std::size_t>(b_col)] += a_val * b_val;
        }
      }
    }

    std::sort(current_row_cols.begin(), current_row_cols.end());
    for (int col : current_row_cols) {
      if (temp_values[static_cast<std::size_t>(col)] != 0.0) {
        local_c_vals.push_back(temp_values[static_cast<std::size_t>(col)]);
        local_c_cols.push_back(col);
        row_nz++;
      }
    }
    local_c_row_ptr[static_cast<std::size_t>(i + 1)] = local_c_row_ptr[static_cast<std::size_t>(i)] + row_nz;
  }
}
}  // namespace

SparseMatrixMultMPI::SparseMatrixMultMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool SparseMatrixMultMPI::ValidationImpl() {
  int is_mpi_init = 0;
  MPI_Initialized(&is_mpi_init);
  if (!is_mpi_init) {
    return false;
  }

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int is_valid = 1;
  if (rank == 0) {
    const auto &A = GetInput().A;
    const auto &B = GetInput().B;
    if (A.cols != B.rows || A.rows <= 0 || B.cols <= 0) {
      is_valid = 0;
    }
  }
  MPI_Bcast(&is_valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return is_valid == 1;
}

bool SparseMatrixMultMPI::PreProcessingImpl() {
  int is_mpi_init = 0;
  MPI_Initialized(&is_mpi_init);
  if (!is_mpi_init) {
    return false;
  }

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    GetOutput().rows = GetInput().A.rows;
    GetOutput().cols = GetInput().B.cols;
    GetOutput().row_ptr.assign(static_cast<std::size_t>(GetInput().A.rows + 1), 0);
  }
  return true;
}

bool SparseMatrixMultMPI::RunImpl() {
  int is_mpi_init = 0;
  MPI_Initialized(&is_mpi_init);
  if (!is_mpi_init) {
    return false;
  }

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int a_rows = 0, a_cols = 0, b_rows = 0, b_cols = 0, b_nnz = 0;

  if (rank == 0) {
    a_rows = GetInput().A.rows;
    a_cols = GetInput().A.cols;
    b_rows = GetInput().B.rows;
    b_cols = GetInput().B.cols;
    b_nnz = static_cast<int>(GetInput().B.values.size());
  }

  MPI_Bcast(&a_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&a_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&b_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&b_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&b_nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (a_rows == 0) {
    return false;
  }

  std::vector<double> dummy_d(1, 0.0);
  std::vector<int> dummy_i(1, 0);

  SparseMatrixCRS B_local;
  B_local.rows = b_rows;
  B_local.cols = b_cols;
  B_local.values.resize(static_cast<std::size_t>(std::max(1, b_nnz)));
  B_local.col_indices.resize(static_cast<std::size_t>(std::max(1, b_nnz)));
  B_local.row_ptr.resize(static_cast<std::size_t>(b_rows + 1));

  if (rank == 0) {
    if (b_nnz > 0) {
      std::copy(GetInput().B.values.begin(), GetInput().B.values.end(), B_local.values.begin());
      std::copy(GetInput().B.col_indices.begin(), GetInput().B.col_indices.end(), B_local.col_indices.begin());
    }
    B_local.row_ptr = GetInput().B.row_ptr;
  }

  if (b_nnz > 0) {
    MPI_Bcast(B_local.values.data(), b_nnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(B_local.col_indices.data(), b_nnz, MPI_INT, 0, MPI_COMM_WORLD);
  }
  MPI_Bcast(B_local.row_ptr.data(), b_rows + 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> send_counts(static_cast<std::size_t>(size), 0);
  std::vector<int> displs(static_cast<std::size_t>(size), 0);
  CalculateCountsAndDisplacements(size, a_rows, send_counts, displs);

  std::vector<int> a_row_ptr_full;
  if (rank == 0) {
    a_row_ptr_full = GetInput().A.row_ptr;
  } else {
    a_row_ptr_full.resize(static_cast<std::size_t>(a_rows + 1));
  }
  MPI_Bcast(a_row_ptr_full.data(), a_rows + 1, MPI_INT, 0, MPI_COMM_WORLD);

  int local_rows = send_counts[static_cast<std::size_t>(rank)];
  int my_start_row = displs[static_cast<std::size_t>(rank)];
  int my_start_idx = a_row_ptr_full[static_cast<std::size_t>(my_start_row)];
  int my_end_idx = a_row_ptr_full[static_cast<std::size_t>(my_start_row + local_rows)];
  int my_nnz = my_end_idx - my_start_idx;

  std::vector<int> a_send_counts(static_cast<std::size_t>(size), 0);
  std::vector<int> a_displs(static_cast<std::size_t>(size), 0);
  if (rank == 0) {
    for (int i = 0; i < size; ++i) {
      int s_row = displs[static_cast<std::size_t>(i)];
      int e_row = s_row + send_counts[static_cast<std::size_t>(i)];
      a_send_counts[static_cast<std::size_t>(i)] =
          a_row_ptr_full[static_cast<std::size_t>(e_row)] - a_row_ptr_full[static_cast<std::size_t>(s_row)];
      a_displs[static_cast<std::size_t>(i)] = a_row_ptr_full[static_cast<std::size_t>(s_row)];
    }
  }

  std::vector<double> local_a_vals(static_cast<std::size_t>(std::max(1, my_nnz)));
  std::vector<int> local_a_cols(static_cast<std::size_t>(std::max(1, my_nnz)));

  const double *a_vals_send = (rank == 0 && !GetInput().A.values.empty()) ? GetInput().A.values.data() : dummy_d.data();
  const int *a_cols_send =
      (rank == 0 && !GetInput().A.col_indices.empty()) ? GetInput().A.col_indices.data() : dummy_i.data();

  MPI_Scatterv(a_vals_send, a_send_counts.data(), a_displs.data(), MPI_DOUBLE, local_a_vals.data(), my_nnz, MPI_DOUBLE,
               0, MPI_COMM_WORLD);
  MPI_Scatterv(a_cols_send, a_send_counts.data(), a_displs.data(), MPI_INT, local_a_cols.data(), my_nnz, MPI_INT, 0,
               MPI_COMM_WORLD);

  std::vector<double> local_c_vals;
  std::vector<int> local_c_cols;
  std::vector<int> local_c_row_ptr(static_cast<std::size_t>(local_rows + 1), 0);

  MultiplyLocalSparse(local_rows, b_cols, my_start_row, my_start_idx, a_row_ptr_full, local_a_cols, local_a_vals,
                      B_local, local_c_vals, local_c_cols, local_c_row_ptr);

  std::vector<int> local_nnz_per_row(static_cast<std::size_t>(std::max(1, local_rows)));
  for (int i = 0; i < local_rows; i++) {
    local_nnz_per_row[static_cast<std::size_t>(i)] =
        local_c_row_ptr[static_cast<std::size_t>(i + 1)] - local_c_row_ptr[static_cast<std::size_t>(i)];
  }

  std::vector<int> global_nnz_per_row(static_cast<std::size_t>(rank == 0 ? std::max(1, a_rows) : 1));

  MPI_Gatherv(local_nnz_per_row.data(), local_rows, MPI_INT, global_nnz_per_row.data(), send_counts.data(),
              displs.data(), MPI_INT, 0, MPI_COMM_WORLD);

  int my_total_nnz = static_cast<int>(local_c_vals.size());
  std::vector<int> c_nnz_counts(static_cast<std::size_t>(size));
  MPI_Gather(&my_total_nnz, 1, MPI_INT, c_nnz_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> c_nnz_displs(static_cast<std::size_t>(size), 0);
  int total_c_nnz = 0;
  if (rank == 0) {
    for (int i = 0; i < size; i++) {
      c_nnz_displs[static_cast<std::size_t>(i)] = total_c_nnz;
      total_c_nnz += c_nnz_counts[static_cast<std::size_t>(i)];
    }
    GetOutput().values.resize(static_cast<std::size_t>(std::max(1, total_c_nnz)));
    GetOutput().col_indices.resize(static_cast<std::size_t>(std::max(1, total_c_nnz)));
  }

  const double *send_c_vals = my_total_nnz > 0 ? local_c_vals.data() : dummy_d.data();
  const int *send_c_cols = my_total_nnz > 0 ? local_c_cols.data() : dummy_i.data();
  double *recv_c_vals = rank == 0 ? GetOutput().values.data() : dummy_d.data();
  int *recv_c_cols = rank == 0 ? GetOutput().col_indices.data() : dummy_i.data();

  MPI_Gatherv(send_c_vals, my_total_nnz, MPI_DOUBLE, recv_c_vals, c_nnz_counts.data(), c_nnz_displs.data(), MPI_DOUBLE,
              0, MPI_COMM_WORLD);

  MPI_Gatherv(send_c_cols, my_total_nnz, MPI_INT, recv_c_cols, c_nnz_counts.data(), c_nnz_displs.data(), MPI_INT, 0,
              MPI_COMM_WORLD);

  if (rank == 0) {
    GetOutput().values.resize(static_cast<std::size_t>(total_c_nnz));
    GetOutput().col_indices.resize(static_cast<std::size_t>(total_c_nnz));
    GetOutput().row_ptr[0] = 0;
    for (int i = 0; i < a_rows; i++) {
      GetOutput().row_ptr[static_cast<std::size_t>(i + 1)] =
          GetOutput().row_ptr[static_cast<std::size_t>(i)] + global_nnz_per_row[static_cast<std::size_t>(i)];
    }
  }

  return true;
}

bool SparseMatrixMultMPI::PostProcessingImpl() {
  return true;
}

}  // namespace salena_s_sparse_matrix_mult
