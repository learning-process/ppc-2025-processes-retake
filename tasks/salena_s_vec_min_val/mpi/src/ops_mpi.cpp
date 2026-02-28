#include "salena_s_vec_min_val/mpi/include/ops_mpi.hpp"
#include <mpi.h>
#include <algorithm>
#include <limits>
#include <vector>

namespace salena_s_vec_min_val {

TestTaskMPI::TestTaskMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool TestTaskMPI::ValidationImpl() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    return !GetInput().empty();
  }
  return true;
}

bool TestTaskMPI::PreProcessingImpl() {
  GetOutput() = std::numeric_limits<int>::max();
  return true;
}

bool TestTaskMPI::RunImpl() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int total_elements = 0;
  if (rank == 0) {
    total_elements = GetInput().size();
  }
  MPI_Bcast(&total_elements, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (total_elements == 0) return false;

  std::vector<int> sendcounts(size);
  std::vector<int> displs(size);
  int chunk = total_elements / size;
  int rem = total_elements % size;

  int current_displ = 0;
  for (int i = 0; i < size; ++i) {
    sendcounts[i] = chunk + (i < rem ? 1 : 0);
    displs[i] = current_displ;
    current_displ += sendcounts[i];
  }

  std::vector<int> local_data(sendcounts[rank]);
  MPI_Scatterv(rank == 0 ? GetInput().data() : nullptr, sendcounts.data(), displs.data(), MPI_INT,
               local_data.data(), sendcounts[rank], MPI_INT, 0, MPI_COMM_WORLD);

  int local_min = std::numeric_limits<int>::max();
  for (int val : local_data) {
    local_min = std::min(local_min, val);
  }

  int global_min = 0;
  MPI_Reduce(&local_min, &global_min, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    GetOutput() = global_min;
  }

  return true;
}

bool TestTaskMPI::PostProcessingImpl() {
  return true;
}

}  // namespace salena_s_vec_min_valcc