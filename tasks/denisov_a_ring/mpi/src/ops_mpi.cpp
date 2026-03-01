#include "denisov_a_ring/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <vector>

#include "denisov_a_ring/common/include/common.hpp"

namespace denisov_a_ring {

RingTopologyMPI::RingTopologyMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool RingTopologyMPI::ValidationImpl() {
  int world_size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const auto &in = GetInput();
  if (in.source < 0 || in.source >= world_size) {
    return false;
  }

  if (in.destination < 0 || in.destination >= world_size) {
    return false;
  }

  return true;
}

bool RingTopologyMPI::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

bool RingTopologyMPI::RunImpl() {
  int rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const auto &in = GetInput();
  int start_node = in.source;
  int target_node = in.destination;

  if (start_node == target_node) {
    if (rank == start_node) {
      GetOutput() = in.data;
    }
    BroadcastResult(GetOutput(), rank, start_node);
    return true;
  }

  int next = (rank + 1) % world_size;
  int prev = (rank - 1 + world_size) % world_size;

  int total_steps = (target_node - start_node + world_size) % world_size;
  int local_step = (rank - start_node + world_size) % world_size;

  std::vector<int> local_buf;

  if (local_step == 0) {
    local_buf = in.data;
    SendVector(local_buf, next);
  } else if (local_step < total_steps) {
    ReceiveVector(local_buf, prev);
    SendVector(local_buf, next);
  } else if (local_step == total_steps) {
    ReceiveVector(local_buf, prev);
  }

  if (rank == target_node) {
    GetOutput() = local_buf;
  }

  BroadcastResult(GetOutput(), rank, target_node);
  return true;
}

bool RingTopologyMPI::PostProcessingImpl() {
  return true;
}

void RingTopologyMPI::SendVector(const std::vector<int> &data, int to_rank) {
  int count = static_cast<int>(data.size());
  MPI_Send(&count, 1, MPI_INT, to_rank, 0, MPI_COMM_WORLD);

  if (count > 0) {
    MPI_Send(data.data(), count, MPI_INT, to_rank, 1, MPI_COMM_WORLD);
  }
}

void RingTopologyMPI::ReceiveVector(std::vector<int> &data, int from_rank) {
  int count = 0;
  MPI_Recv(&count, 1, MPI_INT, from_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  data.resize(count);
  if (count > 0) {
    MPI_Recv(data.data(), count, MPI_INT, from_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
}

void RingTopologyMPI::BroadcastResult(std::vector<int> &out, int rank, int root) {
  int size = (rank == root) ? static_cast<int>(out.size()) : 0;
  MPI_Bcast(&size, 1, MPI_INT, root, MPI_COMM_WORLD);

  if (rank != root) {
    out.resize(size);
  }

  if (size > 0) {
    MPI_Bcast(out.data(), size, MPI_INT, root, MPI_COMM_WORLD);
  }
}

}  // namespace denisov_a_ring
