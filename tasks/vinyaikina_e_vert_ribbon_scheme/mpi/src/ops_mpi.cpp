#include "vinyaikina_e_vert_ribbon_scheme/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <vector>

#include "vinyaikina_e_vert_ribbon_scheme/common/include/common.hpp"

namespace vinyaikina_e_vert_ribbon_scheme {

namespace {

void ComputeColumnDistribution(int cols, int num_proc, std::vector<int> &counts, std::vector<int> &starts) {
  int base = cols / num_proc;
  int rem = cols % num_proc;
  for (int i = 0; i < num_proc; i++) {
    counts[i] = base + (i < rem ? 1 : 0);
  }
  starts[0] = 0;
  for (int i = 1; i < num_proc; i++) {
    starts[i] = starts[i - 1] + counts[i - 1];
  }
}

std::vector<int> PackMatrixBuffer(int rows, int cols, int num_proc, const std::vector<int> &matrix,
                                  const std::vector<int> &col_counts, const std::vector<int> &col_starts,
                                  std::vector<int> &send_counts, std::vector<int> &send_offsets) {
  int total_send = 0;
  for (int pr = 0; pr < num_proc; pr++) {
    send_counts[pr] = rows * col_counts[pr];
    send_offsets[pr] = total_send;
    total_send += send_counts[pr];
  }
  std::vector<int> buf(total_send);
  for (int pr = 0; pr < num_proc; pr++) {
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < col_counts[pr]; j++) {
        buf[send_offsets[pr] + (i * col_counts[pr]) + j] = matrix[(i * cols) + col_starts[pr] + j];
      }
    }
  }
  return buf;
}

}  // namespace

VinyaikinaEVertRibbonSchemeMPI::VinyaikinaEVertRibbonSchemeMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool VinyaikinaEVertRibbonSchemeMPI::ValidationImpl() {
  return GetInput() > 0 && GetOutput() == 0;
}

bool VinyaikinaEVertRibbonSchemeMPI::PreProcessingImpl() {
  rows_ = GetInput();
  cols_ = GetInput();
  if (rows_ <= 0 || cols_ <= 0) {
    return false;
  }
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    matrix_.assign(static_cast<std::size_t>(rows_) * cols_, 1);
    vector_.assign(cols_, 1);
  }
  result_.assign(rows_, 0);
  return true;
}

bool VinyaikinaEVertRibbonSchemeMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::vector<int> col_counts(size);
  std::vector<int> col_starts(size);
  ComputeColumnDistribution(cols_, size, col_counts, col_starts);

  int my_cols = col_counts[rank];

  std::vector<int> matrix_send_buf;
  std::vector<int> matrix_send_counts(size, 0);
  std::vector<int> matrix_send_offsets(size, 0);

  if (rank == 0) {
    matrix_send_buf =
        PackMatrixBuffer(rows_, cols_, size, matrix_, col_counts, col_starts, matrix_send_counts, matrix_send_offsets);
  }

  int matrix_recv_count = rows_ * my_cols;
  std::vector<int> local_matrix(matrix_recv_count);
  MPI_Scatterv(matrix_send_buf.data(), matrix_send_counts.data(), matrix_send_offsets.data(), MPI_INT,
               local_matrix.data(), matrix_recv_count, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> vector_send_counts(size, 0);
  std::vector<int> vector_send_offsets(size, 0);
  for (int pr = 0; pr < size; pr++) {
    vector_send_counts[pr] = col_counts[pr];
    vector_send_offsets[pr] = col_starts[pr];
  }

  std::vector<int> local_vector(my_cols);
  MPI_Scatterv(vector_.data(), vector_send_counts.data(), vector_send_offsets.data(), MPI_INT, local_vector.data(),
               my_cols, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> local_result(rows_, 0);
  for (int i = 0; i < rows_; i++) {
    for (int j = 0; j < my_cols; j++) {
      local_result[i] += local_matrix[(i * my_cols) + j] * local_vector[j];
    }
  }

  result_.assign(rows_, 0);
  MPI_Reduce(local_result.data(), result_.data(), rows_, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  int total = 0;
  if (rank == 0) {
    for (int i = 0; i < rows_; i++) {
      total += result_[i];
    }
    for (int dest = 1; dest < size; dest++) {
      MPI_Send(&total, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
    }
  } else {
    MPI_Recv(&total, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  GetOutput() = total;
  return true;
}

bool VinyaikinaEVertRibbonSchemeMPI::PostProcessingImpl() {
  if (GetInput() == 0) {
    return false;
  }
  GetOutput() /= GetInput();
  return true;
}

}  // namespace vinyaikina_e_vert_ribbon_scheme
