#include "vinyaikina_e_matrix_sum/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <vector>

#include "vinyaikina_e_matrix_sum/common/include/common.hpp"

namespace vinyaikina_e_matrix_sum {

VinyaikinaEMatrixSumMPI::VinyaikinaEMatrixSumMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool VinyaikinaEMatrixSumMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    auto &input = GetInput();
    if (input.size() < 2) {
      return false;
    }
    int r = input[0];
    int c = input[1];
    if (r <= 0 || c <= 0) {
      return false;
    }
    return input.size() == (static_cast<size_t>(r) * static_cast<size_t>(c)) + 2;
  }
  return true;
}

bool VinyaikinaEMatrixSumMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    auto &input = GetInput();
    rows_ = input[0];
    cols_ = input[1];
    matrix_.assign(input.begin() + 2, input.end());
  }
  return true;
}

bool VinyaikinaEMatrixSumMPI::RunImpl() {
  int rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  MPI_Bcast(&rows_, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols_, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int base_rows = rows_ / world_size;
  int extra = rows_ % world_size;

  std::vector<int> sendcounts(world_size);
  std::vector<int> displs(world_size);
  int offset = 0;
  for (int i = 0; i < world_size; i++) {
    int nrows = base_rows + (i < extra ? 1 : 0);
    sendcounts[i] = nrows * cols_;
    displs[i] = offset;
    offset += sendcounts[i];
  }

  int local_rows = base_rows + (rank < extra ? 1 : 0);
  std::vector<int> local_data(static_cast<size_t>(local_rows) * cols_);

  MPI_Scatterv(matrix_.data(), sendcounts.data(), displs.data(), MPI_INT, local_data.data(), local_rows * cols_,
               MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> local_sums(local_rows);
  for (int i = 0; i < local_rows; i++) {
    int sum = 0;
    for (int j = 0; j < cols_; j++) {
      sum += local_data[(i * cols_) + j];
    }
    local_sums[i] = sum;
  }

  std::vector<int> recvcounts(world_size);
  std::vector<int> rdispls(world_size);
  offset = 0;
  for (int i = 0; i < world_size; i++) {
    recvcounts[i] = base_rows + (i < extra ? 1 : 0);
    rdispls[i] = offset;
    offset += recvcounts[i];
  }

  row_sums_.resize(rows_);
  MPI_Gatherv(local_sums.data(), local_rows, MPI_INT, row_sums_.data(), recvcounts.data(), rdispls.data(), MPI_INT, 0,
              MPI_COMM_WORLD);

  return true;
}

bool VinyaikinaEMatrixSumMPI::PostProcessingImpl() {
  row_sums_.resize(rows_);
  MPI_Bcast(row_sums_.data(), rows_, MPI_INT, 0, MPI_COMM_WORLD);
  GetOutput() = row_sums_;
  return true;
}

}  // namespace vinyaikina_e_matrix_sum
