#include "salena_s_matrix_vector_mult/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <vector>

#include "salena_s_matrix_vector_mult/common/include/common.hpp"

namespace salena_s_matrix_vector_mult {

TestTaskMPI::TestTaskMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  if (GetInput().rows > 0) {
    GetOutput().resize(static_cast<std::size_t>(GetInput().rows), 0.0);
  }
}

bool TestTaskMPI::ValidationImpl() {
  int is_mpi_init = 0;
  MPI_Initialized(&is_mpi_init);
  if (!is_mpi_init) {
    return false;
  }

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int is_valid = 1;
  if (rank == 0) {
    if (GetInput().rows <= 0 || GetInput().cols <= 0) {
      is_valid = 0;
    } else if (GetInput().matrix.size() !=
               static_cast<std::size_t>(GetInput().rows) * static_cast<std::size_t>(GetInput().cols)) {
      is_valid = 0;
    } else if (GetInput().vec.size() != static_cast<std::size_t>(GetInput().cols)) {
      is_valid = 0;
    }
  }
  MPI_Bcast(&is_valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return is_valid == 1;
}

bool TestTaskMPI::PreProcessingImpl() {
  int is_mpi_init = 0;
  MPI_Initialized(&is_mpi_init);
  if (!is_mpi_init) {
    return false;
  }

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    GetOutput().assign(static_cast<std::size_t>(GetInput().rows), 0.0);
  }
  return true;
}

std::vector<double> TestTaskMPI::Transpose(const std::vector<double> &matrix, int rows, int cols) {
  std::vector<double> transposed(static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols));
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      transposed[static_cast<std::size_t>((j * rows) + i)] = matrix[static_cast<std::size_t>((i * cols) + j)];
    }
  }
  return transposed;
}

bool TestTaskMPI::RunImpl() {
  int is_mpi_init = 0;
  MPI_Initialized(&is_mpi_init);
  if (!is_mpi_init) {
    return false;
  }

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int rows = (rank == 0) ? GetInput().rows : 0;
  int cols = (rank == 0) ? GetInput().cols : 0;

  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rows == 0 || cols == 0) {
    return false;
  }

  std::vector<int> send_counts(static_cast<std::size_t>(size));
  std::vector<int> displs(static_cast<std::size_t>(size));
  std::vector<int> vec_counts(static_cast<std::size_t>(size));
  std::vector<int> vec_displs(static_cast<std::size_t>(size));

  int delta_cols = cols / size;
  int rem_cols = cols % size;
  int current_displ = 0;
  int cur_vec = 0;

  for (int i = 0; i < size; ++i) {
    int c = delta_cols + (i < rem_cols ? 1 : 0);
    send_counts[static_cast<std::size_t>(i)] = c * rows;
    displs[static_cast<std::size_t>(i)] = current_displ;
    current_displ += c * rows;

    vec_counts[static_cast<std::size_t>(i)] = c;
    vec_displs[static_cast<std::size_t>(i)] = cur_vec;
    cur_vec += c;
  }

  std::vector<double> dummy_d(1, 0.0);

  std::vector<double> local_matrix(static_cast<std::size_t>(std::max(1, send_counts[static_cast<std::size_t>(rank)])));
  std::vector<double> local_vec(static_cast<std::size_t>(std::max(1, vec_counts[static_cast<std::size_t>(rank)])));
  std::vector<double> matrix_transposed;

  if (rank == 0) {
    matrix_transposed = Transpose(GetInput().matrix, rows, cols);
  }

  const double *matrix_send = (rank == 0 && !matrix_transposed.empty()) ? matrix_transposed.data() : dummy_d.data();
  MPI_Scatterv(matrix_send, send_counts.data(), displs.data(), MPI_DOUBLE, local_matrix.data(),
               send_counts[static_cast<std::size_t>(rank)], MPI_DOUBLE, 0, MPI_COMM_WORLD);

  const double *vec_send = (rank == 0 && !GetInput().vec.empty()) ? GetInput().vec.data() : dummy_d.data();
  MPI_Scatterv(vec_send, vec_counts.data(), vec_displs.data(), MPI_DOUBLE, local_vec.data(),
               vec_counts[static_cast<std::size_t>(rank)], MPI_DOUBLE, 0, MPI_COMM_WORLD);

  std::vector<double> local_res(static_cast<std::size_t>(std::max(1, rows)), 0.0);
  int my_cols_count = vec_counts[static_cast<std::size_t>(rank)];

  for (int j = 0; j < my_cols_count; ++j) {
    double vec_val = local_vec[static_cast<std::size_t>(j)];
    for (int i = 0; i < rows; ++i) {
      local_res[static_cast<std::size_t>(i)] += local_matrix[static_cast<std::size_t>((j * rows) + i)] * vec_val;
    }
  }

  double *res_recv = rank == 0 ? GetOutput().data() : dummy_d.data();
  MPI_Reduce(local_res.data(), res_recv, rows, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  return true;
}

bool TestTaskMPI::PostProcessingImpl() {
  return true;
}

}  // namespace salena_s_matrix_vector_mult
