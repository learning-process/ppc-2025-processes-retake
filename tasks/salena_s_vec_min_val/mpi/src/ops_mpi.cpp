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
  int is_mpi_init = 0;
  MPI_Initialized(&is_mpi_init);
  if (!is_mpi_init) {
    return false;
  }

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int is_valid = 1;
  if (rank == 0) {
    is_valid = !GetInput().empty() ? 1 : 0;
  }
  MPI_Bcast(&is_valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return is_valid == 1;
}

bool TestTaskMPI::PreProcessingImpl() {
  GetOutput() = std::numeric_limits<int>::max();
  return true;
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

  int total_elements = 0;
  if (rank == 0) {
    total_elements = static_cast<int>(GetInput().size());
  }

  MPI_Bcast(&total_elements, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (total_elements == 0) {
    return false;
  }

  std::vector<int> sendcounts(size);
  std::vector<int> displs(size);

  int remainder = total_elements % size;
  int sum = 0;
  for (int i = 0; i < size; i++) {
    sendcounts[static_cast<std::size_t>(i)] = total_elements / size;
    if (remainder > 0) {
      sendcounts[static_cast<std::size_t>(i)]++;
      remainder--;
    }
    displs[static_cast<std::size_t>(i)] = sum;
    sum += sendcounts[static_cast<std::size_t>(i)];
  }

  std::vector<int> dummy_i(1, 0);
  std::vector<int> local_data(static_cast<std::size_t>(std::max(1, sendcounts[static_cast<std::size_t>(rank)])));

  const int *input_send = (rank == 0 && !GetInput().empty()) ? GetInput().data() : dummy_i.data();
  MPI_Scatterv(input_send, sendcounts.data(), displs.data(), MPI_INT, local_data.data(),
               sendcounts[static_cast<std::size_t>(rank)], MPI_INT, 0, MPI_COMM_WORLD);

  int local_min = std::numeric_limits<int>::max();
  if (sendcounts[static_cast<std::size_t>(rank)] > 0) {
    for (int i = 0; i < sendcounts[static_cast<std::size_t>(rank)]; ++i) {
      local_min = std::min(local_min, local_data[static_cast<std::size_t>(i)]);
    }
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

}  // namespace salena_s_vec_min_val
