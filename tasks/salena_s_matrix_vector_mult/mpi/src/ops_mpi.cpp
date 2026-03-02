#include "salena_s_matrix_vector_mult/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <vector>

#include "salena_s_matrix_vector_mult/common/include/common.hpp"

namespace salena_s_matrix_vector_mult {
namespace {

void CalculateCountsAndDisplacements(int size, int cols, int rows, std::vector<int> &send_counts,
                                     std::vector<int> &displs, std::vector<int> &vec_counts,
                                     std::vector<int> &vec_displs) {
  int delta_cols = cols / size;
  int rem_cols = cols % size;
  int current_displ = 0;
  int cur_vec = 0;
  for (int i = 0; i < size; ++i) {
    int c = delta_cols + (i < rem_cols ? 1 : 0);
    send_counts[i] = c * rows;
    displs[i] = current_displ;
    current_displ += send_counts[i];

    vec_counts[i] = c;
    vec_displs[i] = cur_vec;
    cur_vec += c;
  }
}

void MultiplyLocal(int rows, int my_cols_count, const std::vector<double> &local_matrix,
                   const std::vector<double> &local_vec, std::vector<double> &local_res) {
  for (int j = 0; j < my_cols_count; ++j) {
    double vec_val = local_vec[j];
    for (int i = 0; i < rows; ++i) {
      local_res[static_cast<std::size_t>(i)] += local_matrix[static_cast<std::size_t>((j * rows) + i)] * vec_val;
    }
  }
}

}  // namespace

TestTaskMPI::TestTaskMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  if (GetInput().rows > 0) {
    GetOutput().resize(GetInput().rows, 0.0);
  }
}

bool TestTaskMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    if (GetInput().rows <= 0 || GetInput().cols <= 0) {
      return false;
    }
    if (GetInput().matrix.size() !=
        static_cast<std::size_t>(GetInput().rows) * static_cast<std::size_t>(GetInput().cols)) {
      return false;
    }
    if (GetInput().vec.size() != static_cast<std::size_t>(GetInput().cols)) {
      return false;
    }
  }
  return true;
}

bool TestTaskMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    GetOutput().assign(GetInput().rows, 0.0);
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

  std::vector<int> send_counts(size);
  std::vector<int> displs(size);
  std::vector<int> vec_counts(size);
  std::vector<int> vec_displs(size);

  CalculateCountsAndDisplacements(size, cols, rows, send_counts, displs, vec_counts, vec_displs);

  std::vector<double> local_matrix(send_counts[rank]);
  std::vector<double> local_vec(vec_counts[rank]);
  std::vector<double> matrix_transposed;

  if (rank == 0) {
    matrix_transposed = Transpose(GetInput().matrix, rows, cols);
  }

  MPI_Scatterv(rank == 0 ? matrix_transposed.data() : nullptr, send_counts.data(), displs.data(), MPI_DOUBLE,
               local_matrix.data(), send_counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

  MPI_Scatterv(rank == 0 ? GetInput().vec.data() : nullptr, vec_counts.data(), vec_displs.data(), MPI_DOUBLE,
               local_vec.data(), vec_counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

  std::vector<double> local_res(rows, 0.0);
  int my_cols_count = vec_counts[rank];

  MultiplyLocal(rows, my_cols_count, local_matrix, local_vec, local_res);

  MPI_Reduce(local_res.data(), rank == 0 ? GetOutput().data() : nullptr, rows, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  return true;
}

bool TestTaskMPI::PostProcessingImpl() {
  return true;
}

}  // namespace salena_s_matrix_vector_mult
