#include "yushkova_p_hypercube/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <vector>

#include "yushkova_p_hypercube/common/include/common.hpp"

namespace yushkova_p_hypercube {

namespace {

bool IsPowerOfTwo(int value) {
  return value > 0 && (value & (value - 1)) == 0;
}

int HypercubeDimension(int size) {
  int result = 0;
  while ((1 << result) < size) {
    ++result;
  }
  return result;
}

}  // namespace

YushkovaPHypercubeMPI::YushkovaPHypercubeMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool YushkovaPHypercubeMPI::ValidationImpl() {
  int world_size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const int source = std::get<0>(GetInput());
  const int destination = std::get<1>(GetInput());

  const bool correct_source = source >= 0 && source < world_size;
  const bool correct_destination = destination >= 0 && destination < world_size;
  return IsPowerOfTwo(world_size) && correct_source && correct_destination;
}

bool YushkovaPHypercubeMPI::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool YushkovaPHypercubeMPI::RunImpl() {
  int rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const int source = std::get<0>(GetInput());
  const int destination = std::get<1>(GetInput());
  const int payload = std::get<2>(GetInput());

  if (!IsPowerOfTwo(world_size)) {
    return false;
  }

  const int dimension = HypercubeDimension(world_size);
  std::vector<MPI_Comm> dimension_comms(static_cast<size_t>(dimension), MPI_COMM_NULL);

  for (int bit = 0; bit < dimension; ++bit) {
    const int color = rank & ~(1 << bit);
    MPI_Comm_split(MPI_COMM_WORLD, color, rank, &dimension_comms[static_cast<size_t>(bit)]);
  }

  int current_owner = source;
  int value = (rank == source) ? payload : 0;
  const int route_mask = source ^ destination;

  for (int bit = 0; bit < dimension; ++bit) {
    const bool bit_differs = (route_mask & (1 << bit)) != 0;
    if (!bit_differs) {
      continue;
    }

    const int next_owner = current_owner ^ (1 << bit);
    const int tag = 1000 + bit;

    if (rank == current_owner) {
      MPI_Send(&value, 1, MPI_INT, next_owner, tag, MPI_COMM_WORLD);
    } else if (rank == next_owner) {
      MPI_Recv(&value, 1, MPI_INT, current_owner, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    current_owner = next_owner;
  }

  for (MPI_Comm &comm : dimension_comms) {
    if (comm != MPI_COMM_NULL) {
      MPI_Comm_free(&comm);
    }
  }

  MPI_Bcast(&value, 1, MPI_INT, destination, MPI_COMM_WORLD);
  GetOutput() = value;
  return true;
}

bool YushkovaPHypercubeMPI::PostProcessingImpl() {
  return true;
}

}  // namespace yushkova_p_hypercube
