#include "zyuzin_n_sum_elements_of_matrix/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <numeric>
#include <vector>

#include "zyuzin_n_sum_elements_of_matrix/common/include/common.hpp"

namespace zyuzin_n_sum_elements_of_matrix {

ZyuzinNSumElementsOfMatrixMPI::ZyuzinNSumElementsOfMatrixMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool ZyuzinNSumElementsOfMatrixMPI::ValidationImpl() {
  const auto &matrix = GetInput();
  return std::get<0>(matrix) > 0 && std::get<1>(matrix) > 0 &&
         std::get<2>(matrix).size() ==
             static_cast<std::size_t>(std::get<0>(matrix) * static_cast<std::size_t>(std::get<1>(matrix)));
}

bool ZyuzinNSumElementsOfMatrixMPI::PreProcessingImpl() {
  return true;
}

bool ZyuzinNSumElementsOfMatrixMPI::RunImpl() {
  const auto &matrix = GetInput();
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int rows_per_proc = std::get<0>(matrix) / size;
  int ost = std::get<0>(matrix) % size;

  std::vector<int> send_counts(size, 0);
  std::vector<int> start_indexs(size, 0);
  int offset = 0;
  for (int i = 0; i < size; ++i) {
    int rows = rows_per_proc + (ost > 0 ? 1 : 0);
    if (ost > 0) {
      --ost;
    }
    send_counts[i] = rows * std::get<1>(matrix);
    start_indexs[i] = offset;
    offset += send_counts[i];
  }

  std::vector<double> local_data(send_counts[rank]);
  MPI_Scatterv(std::get<2>(matrix).data(), send_counts.data(), start_indexs.data(), MPI_DOUBLE, local_data.data(),
               send_counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

  double local_sum = std::accumulate(local_data.begin(), local_data.end(), 0.0);
  double global_sum = 0.0;
  MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  GetOutput() = global_sum;
  return true;
}

bool ZyuzinNSumElementsOfMatrixMPI::PostProcessingImpl() {
  return true;
}

}  // namespace zyuzin_n_sum_elements_of_matrix
