#include "cheremkhin_a_matr_max_colum/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <vector>

#include "cheremkhin_a_matr_max_colum/common/include/common.hpp"

namespace cheremkhin_a_matr_max_colum {

namespace {

struct ColRange {
  int start = 0;
  int end = 0;  // exclusive
};

ColRange GetColRange(int rank, int size, int num_cols) {
  ColRange range;

  const int cols_per_process = num_cols / size;
  const int remainder = num_cols % size;

  range.start = (rank * cols_per_process) + std::min(rank, remainder);
  const int cols_for_rank = cols_per_process + (rank < remainder ? 1 : 0);
  range.end = range.start + cols_for_rank;

  return range;
}

std::vector<int> CalcLocalMax(const InType &matrix, const ColRange &range, int num_rows) {
  const int cols_count = range.end - range.start;
  std::vector<int> local_max_values;
  local_max_values.reserve(static_cast<std::size_t>(cols_count));

  for (int col = range.start; col < range.end; ++col) {
    int max_in_col = matrix[0][col];
    for (int row = 1; row < num_rows; ++row) {
      max_in_col = std::max(max_in_col, matrix[row][col]);
    }
    local_max_values.push_back(max_in_col);
  }

  return local_max_values;
}

void FillResultPart(std::vector<int> &result, const ColRange &range, const std::vector<int> &local_max_values) {
  int local_index = 0;
  for (int col = range.start; col < range.end; ++col) {
    result[col] = local_max_values[local_index];
    local_index = local_index + 1;
  }
}

}  // namespace

CheremkhinAMatrMaxColumMPI::CheremkhinAMatrMaxColumMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput().reserve(in.size());
  GetInput() = in;
  GetOutput() = {};
}

bool CheremkhinAMatrMaxColumMPI::ValidationImpl() {
  return (!GetInput().empty());
}

bool CheremkhinAMatrMaxColumMPI::PreProcessingImpl() {
  return true;
}

bool CheremkhinAMatrMaxColumMPI::RunImpl() {
  const std::vector<std::vector<int>> &matrix = GetInput();

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const int num_rows = static_cast<int>(matrix.size());
  const int num_cols = static_cast<int>(matrix[0].size());

  const ColRange local_range = GetColRange(rank, size, num_cols);
  const std::vector<int> local_max_values = CalcLocalMax(matrix, local_range, num_rows);

  std::vector<int> result(num_cols);
  if (rank == 0) {
    FillResultPart(result, local_range, local_max_values);

    for (int i = 1; i < size; ++i) {
      const ColRange proc_range = GetColRange(i, size, num_cols);
      const int proc_cols = proc_range.end - proc_range.start;

      std::vector<int> recv_buffer(proc_cols);

      MPI_Recv(recv_buffer.data(), proc_cols, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      for (int idx = 0; idx < proc_cols; ++idx) {
        result[proc_range.start + idx] = recv_buffer[idx];
      }
    }

  } else {
    MPI_Send(local_max_values.data(), static_cast<int>(local_max_values.size()), MPI_INT, 0, 0, MPI_COMM_WORLD);
  }

  MPI_Bcast(result.data(), num_cols, MPI_INT, 0, MPI_COMM_WORLD);
  GetOutput() = result;

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool CheremkhinAMatrMaxColumMPI::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace cheremkhin_a_matr_max_colum
