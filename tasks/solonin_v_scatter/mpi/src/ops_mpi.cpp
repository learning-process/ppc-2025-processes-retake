#include "solonin_v_scatter/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <tuple>
#include <vector>

#include "solonin_v_scatter/common/include/common.hpp"

namespace solonin_v_scatter {

SoloninVScatterMPI::SoloninVScatterMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<int>();
}

bool SoloninVScatterMPI::ValidationImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

  // Validate only on root — other ranks may have empty send_buf
  if (rank_ == std::get<2>(GetInput())) {
    const auto &buf = std::get<0>(GetInput());
    int count = std::get<1>(GetInput());
    int root = std::get<2>(GetInput());
    if (buf.empty() || count <= 0 || root < 0 || root >= world_size_) return false;
    if (static_cast<int>(buf.size()) < count * world_size_) return false;
  }
  return true;
}

bool SoloninVScatterMPI::PreProcessingImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
  GetOutput().clear();
  return true;
}

bool SoloninVScatterMPI::RunImpl() {
  int count = 0;
  int root = 0;

  if (rank_ == std::get<2>(GetInput())) {
    count = std::get<1>(GetInput());
    root = std::get<2>(GetInput());
  }

  // Broadcast count and root to all processes
  MPI_Bcast(&count, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&root, 1, MPI_INT, 0, MPI_COMM_WORLD);

  GetOutput().resize(count);

  if (world_size_ == 1) {
    // Single process: just copy own chunk
    const auto &buf = std::get<0>(GetInput());
    GetOutput().assign(buf.begin(), buf.begin() + count);
    return true;
  }

  // Custom scatter: root sends chunks to each rank via MPI_Send/MPI_Recv
  // This is the core of the task — implementing scatter WITHOUT MPI_Scatter
  if (rank_ == root) {
    const auto &send_buf = std::get<0>(GetInput());

    // Copy own chunk first
    std::copy(send_buf.begin() + root * count,
              send_buf.begin() + root * count + count,
              GetOutput().begin());

    // Send to all other ranks
    for (int dest = 0; dest < world_size_; dest++) {
      if (dest == root) continue;
      MPI_Send(send_buf.data() + dest * count, count, MPI_INT, dest, 0, MPI_COMM_WORLD);
    }
  } else {
    MPI_Recv(GetOutput().data(), count, MPI_INT, root, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  return true;
}

bool SoloninVScatterMPI::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace solonin_v_scatter
