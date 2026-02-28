#include "salena_s_sparse_matrix_mult/mpi/include/ops_mpi.hpp"
#include <mpi.h>
#include <vector>
#include <algorithm>

namespace salena_s_sparse_matrix_mult {

SparseMatrixMultMPI::SparseMatrixMultMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool SparseMatrixMultMPI::ValidationImpl() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    const auto& A = GetInput().A;
    const auto& B = GetInput().B;
    return (A.cols == B.rows) && (A.rows > 0) && (B.cols > 0);
  }
  return true;
}

bool SparseMatrixMultMPI::PreProcessingImpl() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    GetOutput().rows = GetInput().A.rows;
    GetOutput().cols = GetInput().B.cols;
    GetOutput().row_ptr.assign(GetInput().A.rows + 1, 0);
  }
  return true;
}

bool SparseMatrixMultMPI::RunImpl() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int a_rows = 0, a_cols = 0, b_rows = 0, b_cols = 0, b_nnz = 0;

  if (rank == 0) {
    a_rows = GetInput().A.rows;
    a_cols = GetInput().A.cols;
    b_rows = GetInput().B.rows;
    b_cols = GetInput().B.cols;
    b_nnz = GetInput().B.values.size();
  }

  MPI_Bcast(&a_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&a_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&b_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&b_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&b_nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (a_rows == 0) return false;

  SparseMatrixCRS B_local;
  B_local.rows = b_rows;
  B_local.cols = b_cols;
  B_local.values.resize(b_nnz);
  B_local.col_indices.resize(b_nnz);
  B_local.row_ptr.resize(b_rows + 1);

  if (rank == 0) {
    B_local.values = GetInput().B.values;
    B_local.col_indices = GetInput().B.col_indices;
    B_local.row_ptr = GetInput().B.row_ptr;
  }

  MPI_Bcast(B_local.values.data(), b_nnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(B_local.col_indices.data(), b_nnz, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(B_local.row_ptr.data(), b_rows + 1, MPI_INT, 0, MPI_COMM_WORLD);

  int delta_rows = a_rows / size;
  int rem_rows = a_rows % size;
  int local_rows = delta_rows + (rank < rem_rows ? 1 : 0);
  
  std::vector<int> send_counts(size, 0);
  std::vector<int> displs(size, 0);
  int cur_displ = 0;
  for (int i = 0; i < size; ++i) {
    send_counts[i] = delta_rows + (i < rem_rows ? 1 : 0);
    displs[i] = cur_displ;
    cur_displ += send_counts[i];
  }

  std::vector<int> a_row_ptr_full;
  if (rank == 0) a_row_ptr_full = GetInput().A.row_ptr;
  else a_row_ptr_full.resize(a_rows + 1);

  MPI_Bcast(a_row_ptr_full.data(), a_rows + 1, MPI_INT, 0, MPI_COMM_WORLD);

  int my_start_row = displs[rank];
  int my_end_row = my_start_row + local_rows;
  int my_start_idx = a_row_ptr_full[my_start_row];
  int my_end_idx = a_row_ptr_full[my_end_row];
  int my_nnz = my_end_idx - my_start_idx;

  std::vector<int> a_send_counts(size, 0);
  std::vector<int> a_displs(size, 0);
  if (rank == 0) {
    for(int i = 0; i < size; ++i) {
        int s_row = displs[i];
        int e_row = s_row + send_counts[i];
        a_send_counts[i] = a_row_ptr_full[e_row] - a_row_ptr_full[s_row];
        a_displs[i] = a_row_ptr_full[s_row];
    }
  }

  std::vector<double> local_a_vals(my_nnz);
  std::vector<int> local_a_cols(my_nnz);

  MPI_Scatterv(rank == 0 ? GetInput().A.values.data() : nullptr, a_send_counts.data(), a_displs.data(), MPI_DOUBLE,
               local_a_vals.data(), my_nnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Scatterv(rank == 0 ? GetInput().A.col_indices.data() : nullptr, a_send_counts.data(), a_displs.data(), MPI_INT,
               local_a_cols.data(), my_nnz, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<double> local_c_vals;
  std::vector<int> local_c_cols;
  std::vector<int> local_c_row_ptr(local_rows + 1, 0);
  std::vector<int> marker(b_cols, -1);
  std::vector<double> temp_values(b_cols, 0.0);

  for (int i = 0; i < local_rows; ++i) {
    int row_nz = 0;
    std::vector<int> current_row_cols;
    
    int local_row_start = a_row_ptr_full[my_start_row + i] - my_start_idx;
    int local_row_end = a_row_ptr_full[my_start_row + i + 1] - my_start_idx;

    for (int j = local_row_start; j < local_row_end; ++j) {
      int a_col = local_a_cols[j];
      double a_val = local_a_vals[j];

      for (int k = B_local.row_ptr[a_col]; k < B_local.row_ptr[a_col + 1]; ++k) {
        int b_col = B_local.col_indices[k];
        double b_val = B_local.values[k];

        if (marker[b_col] != i) {
          marker[b_col] = i;
          current_row_cols.push_back(b_col);
          temp_values[b_col] = a_val * b_val;
        } else {
          temp_values[b_col] += a_val * b_val;
        }
      }
    }

    std::sort(current_row_cols.begin(), current_row_cols.end());
    for (int col : current_row_cols) {
      if (temp_values[col] != 0.0) {
        local_c_vals.push_back(temp_values[col]);
        local_c_cols.push_back(col);
        row_nz++;
      }
    }
    local_c_row_ptr[i + 1] = local_c_row_ptr[i] + row_nz;
  }

  std::vector<int> local_nnz_per_row(local_rows);
  for(int i = 0; i < local_rows; i++) local_nnz_per_row[i] = local_c_row_ptr[i+1] - local_c_row_ptr[i];

  std::vector<int> global_nnz_per_row;
  if (rank == 0) global_nnz_per_row.resize(a_rows);

  MPI_Gatherv(local_nnz_per_row.data(), local_rows, MPI_INT, 
              rank == 0 ? global_nnz_per_row.data() : nullptr, send_counts.data(), displs.data(), MPI_INT, 
              0, MPI_COMM_WORLD);

  int my_total_nnz = local_c_vals.size();
  std::vector<int> c_nnz_counts(size);
  MPI_Gather(&my_total_nnz, 1, MPI_INT, c_nnz_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> c_nnz_displs(size, 0);
  int total_c_nnz = 0;
  if (rank == 0) {
    for(int i = 0; i < size; i++) {
        c_nnz_displs[i] = total_c_nnz;
        total_c_nnz += c_nnz_counts[i];
    }
    GetOutput().values.resize(total_c_nnz);
    GetOutput().col_indices.resize(total_c_nnz);
  }

  MPI_Gatherv(local_c_vals.data(), my_total_nnz, MPI_DOUBLE,
              rank == 0 ? GetOutput().values.data() : nullptr, c_nnz_counts.data(), c_nnz_displs.data(), MPI_DOUBLE,
              0, MPI_COMM_WORLD);
              
  MPI_Gatherv(local_c_cols.data(), my_total_nnz, MPI_INT,
              rank == 0 ? GetOutput().col_indices.data() : nullptr, c_nnz_counts.data(), c_nnz_displs.data(), MPI_INT,
              0, MPI_COMM_WORLD);

  if (rank == 0) {
    GetOutput().row_ptr[0] = 0;
    for(int i = 0; i < a_rows; i++) {
        GetOutput().row_ptr[i+1] = GetOutput().row_ptr[i] + global_nnz_per_row[i];
    }
  }

  return true;
}

bool SparseMatrixMultMPI::PostProcessingImpl() {
  return true;
}

}  // namespace salena_s_sparse_matrix_mult