#include "salena_s_matrix_vector_mult/mpi/include/ops_mpi.hpp"
#include <mpi.h>
#include <algorithm>

namespace salena_s_matrix_vector_mult {

TestTaskMPI::TestTaskMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  if (GetInput().rows > 0) {
    GetOutput().resize(GetInput().rows, 0.0);
  }
}

bool TestTaskMPI::ValidationImpl() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    if (GetInput().rows <= 0 || GetInput().cols <= 0) return false;
    if (GetInput().matrix.size() != static_cast<size_t>(GetInput().rows * GetInput().cols)) return false;
    if (GetInput().vec.size() != static_cast<size_t>(GetInput().cols)) return false;
  }
  return true;
}

bool TestTaskMPI::PreProcessingImpl() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    GetOutput().assign(GetInput().rows, 0.0);
  }
  return true;
}

bool TestTaskMPI::RunImpl() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int rows = 0;
  int cols = 0;

  if (rank == 0) {
    rows = GetInput().rows;
    cols = GetInput().cols;
  }

  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rows == 0 || cols == 0) return false;

  int delta_cols = cols / size;
  int rem_cols = cols % size;
  
  std::vector<int> send_counts(size);
  std::vector<int> displs(size);
  
  int current_displ = 0;
  for (int i = 0; i < size; ++i) {
    int c = delta_cols + (i < rem_cols ? 1 : 0);
    send_counts[i] = c * rows; 
    displs[i] = current_displ;
    current_displ += send_counts[i];
  }

  std::vector<int> vec_counts(size);
  std::vector<int> vec_displs(size);
  int cur_vec = 0;
  for (int i = 0; i < size; ++i) {
    int c = delta_cols + (i < rem_cols ? 1 : 0);
    vec_counts[i] = c;
    vec_displs[i] = cur_vec;
    cur_vec += c;
  }

  std::vector<double> local_matrix(send_counts[rank]);
  std::vector<double> local_vec(vec_counts[rank]);

  std::vector<double> matrix_transposed;
  if (rank == 0) {
    matrix_transposed.resize(rows * cols);
    const auto& matrix = GetInput().matrix;
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        matrix_transposed[j * rows + i] = matrix[i * cols + j];
      }
    }
  }

  MPI_Scatterv(rank == 0 ? matrix_transposed.data() : nullptr, send_counts.data(), displs.data(), MPI_DOUBLE,
               local_matrix.data(), send_counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

  MPI_Scatterv(rank == 0 ? GetInput().vec.data() : nullptr, vec_counts.data(), vec_displs.data(), MPI_DOUBLE,
               local_vec.data(), vec_counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

  std::vector<double> local_res(rows, 0.0);
  int my_cols_count = vec_counts[rank];

  for (int j = 0; j < my_cols_count; ++j) {
    double vec_val = local_vec[j];
    for (int i = 0; i < rows; ++i) {
      local_res[i] += local_matrix[j * rows + i] * vec_val;
    }
  }

  if (rank == 0) {
    MPI_Reduce(local_res.data(), GetOutput().data(), rows, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  } else {
    MPI_Reduce(local_res.data(), nullptr, rows, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  }

  return true;
}

bool TestTaskMPI::PostProcessingImpl() {
  return true;
}

}  // namespace salena_s_matrix_vector_mult