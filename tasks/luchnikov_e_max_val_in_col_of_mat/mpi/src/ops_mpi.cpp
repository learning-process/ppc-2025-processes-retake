#include "luchnikov_e_max_val_in_col_of_mat/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <limits>
#include <vector>

#include "luchnikov_e_max_val_in_col_of_mat/common/include/common.hpp"

namespace luchnikov_e_max_val_in_col_of_mat {

LuchnilkovEMaxValInColOfMatMPI::LuchnilkovEMaxValInColOfMatMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType();
}

bool LuchnilkovEMaxValInColOfMatMPI::ValidationImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &size_);
  return !GetInput().empty() && !GetInput()[0].empty();
}

bool LuchnilkovEMaxValInColOfMatMPI::PreProcessingImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &size_);
  return true;
}

bool LuchnilkovEMaxValInColOfMatMPI::RunImpl() {
  const auto &input = GetInput();
  size_t rows = input.size();
  size_t cols = input[0].size();

  std::vector<int> local_max(cols, std::numeric_limits<int>::min());

  for (size_t i = rank_; i < rows; i += size_) {
    for (size_t j = 0; j < cols; ++j) {
      local_max[j] = std::max(local_max[j], input[i][j]);
    }
  }

  if (rank_ == 0) {
    GetOutput().resize(cols, std::numeric_limits<int>::min());
  }

  MPI_Reduce(local_max.data(), GetOutput().data(), cols, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

  return true;
}

bool LuchnilkovEMaxValInColOfMatMPI::PostProcessingImpl() {
  return true;
}

}  // namespace luchnikov_e_max_val_in_col_of_mat
