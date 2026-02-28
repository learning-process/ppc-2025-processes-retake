#include "nazyrov_a_min_val_vec/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <vector>

#include "nazyrov_a_min_val_vec/common/include/common.hpp"

namespace nazyrov_a_min_val_vec {

MinValVecMPI::MinValVecMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    GetInput() = in;
  } else {
    GetInput() = InType();
  }
  GetOutput() = 0;
}

bool MinValVecMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    return !GetInput().empty();
  }
  return true;
}

bool MinValVecMPI::PreProcessingImpl() {
  return true;
}

bool MinValVecMPI::RunImpl() {
  int rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int total_size = 0;
  if (rank == 0) {
    total_size = static_cast<int>(GetInput().size());
  }
  MPI_Bcast(&total_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int base = total_size / world_size;
  int remainder = total_size % world_size;

  std::vector<int> sendcounts(static_cast<std::size_t>(world_size));
  std::vector<int> displs(static_cast<std::size_t>(world_size));
  int offset = 0;
  for (int i = 0; i < world_size; ++i) {
    sendcounts[static_cast<std::size_t>(i)] = base + (i < remainder ? 1 : 0);
    displs[static_cast<std::size_t>(i)] = offset;
    offset += sendcounts[static_cast<std::size_t>(i)];
  }

  int local_size = sendcounts[static_cast<std::size_t>(rank)];
  std::vector<int> local_data(static_cast<std::size_t>(local_size));

  MPI_Scatterv(rank == 0 ? GetInput().data() : nullptr, sendcounts.data(), displs.data(), MPI_INT, local_data.data(),
               local_size, MPI_INT, 0, MPI_COMM_WORLD);

  int local_min = INT_MAX;
  for (int i = 0; i < local_size; ++i) {
    local_min = std::min(local_data[static_cast<std::size_t>(i)], local_min);
  }

  int global_min = INT_MAX;
  MPI_Allreduce(&local_min, &global_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  GetOutput() = global_min;
  return true;
}

bool MinValVecMPI::PostProcessingImpl() {
  return true;
}

}  // namespace nazyrov_a_min_val_vec
