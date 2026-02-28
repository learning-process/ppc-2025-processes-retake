#include "nazyrov_a_broadcast/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <utility>
#include <vector>

#include "nazyrov_a_broadcast/common/include/common.hpp"

namespace nazyrov_a_broadcast {

namespace {

void TreeBcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm) {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  if (size <= 1) {
    return;
  }

  int relative = (rank - root + size) % size;

  int mask = 1;
  while (mask < size) {
    if ((relative & mask) != 0) {
      int parent = relative - mask;
      int src = (parent + root) % size;
      MPI_Recv(buffer, count, datatype, src, 0, comm, MPI_STATUS_IGNORE);
      break;
    }
    mask <<= 1;
  }

  mask >>= 1;
  while (mask > 0) {
    int child = relative + mask;
    if (child < size) {
      int dest = (child + root) % size;
      MPI_Send(buffer, count, datatype, dest, 0, comm);
    }
    mask >>= 1;
  }
}

}  // namespace

BroadcastMPI::BroadcastMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  InType copy = in;
  GetInput() = std::move(copy);
  GetOutput() = {};
}

bool BroadcastMPI::ValidationImpl() {
  int size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  const auto &input = GetInput();
  return input.root >= 0 && input.root < size;
}

bool BroadcastMPI::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

bool BroadcastMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &input = GetInput();
  int root = input.root;

  int data_size = 0;
  if (rank == root) {
    data_size = static_cast<int>(input.data.size());
  }
  TreeBcast(&data_size, 1, MPI_INT, root, MPI_COMM_WORLD);

  std::vector<int> buffer(static_cast<std::size_t>(data_size));
  if (rank == root) {
    for (std::size_t i = 0; i < input.data.size(); ++i) {
      buffer[i] = input.data[i];
    }
  }
  TreeBcast(buffer.data(), data_size, MPI_INT, root, MPI_COMM_WORLD);

  GetOutput() = buffer;
  return true;
}

bool BroadcastMPI::PostProcessingImpl() {
  return true;
}

}  // namespace nazyrov_a_broadcast
