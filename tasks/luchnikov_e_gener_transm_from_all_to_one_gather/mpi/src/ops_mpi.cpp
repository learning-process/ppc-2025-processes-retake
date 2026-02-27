#include "luchnikov_e_gener_transm_from_all_to_one_gather/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <vector>

namespace luchnikov_e_gener_transm_from_all_to_one_gather {

LuchnikovEGenerTransmFromAllToOneGatherMPI::LuchnikovEGenerTransmFromAllToOneGatherMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType();
}

bool LuchnikovEGenerTransmFromAllToOneGatherMPI::ValidationImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &size_);
  return !GetInput().empty();
}

bool LuchnikovEGenerTransmFromAllToOneGatherMPI::PreProcessingImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &size_);

  size_t total_size = GetInput().size();
  size_t base_size = total_size / size_;
  size_t remainder = total_size % size_;

  size_t local_start = rank_ * base_size + std::min(static_cast<size_t>(rank_), remainder);
  size_t local_end = local_start + base_size + (static_cast<size_t>(rank_) < remainder ? 1 : 0);

  local_data_.clear();
  for (size_t i = local_start; i < local_end && i < total_size; ++i) {
    local_data_.push_back(GetInput()[i]);
  }

  return true;
}

bool LuchnikovEGenerTransmFromAllToOneGatherMPI::RunImpl() {
  std::vector<int> recv_counts(size_);
  std::vector<int> displacements(size_);

  int local_size = static_cast<int>(local_data_.size());

  MPI_Gather(&local_size, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank_ == 0) {
    displacements[0] = 0;
    for (int i = 1; i < size_; ++i) {
      displacements[i] = displacements[i - 1] + recv_counts[i - 1];
    }

    int total_size = displacements[size_ - 1] + recv_counts[size_ - 1];
    GetOutput().resize(total_size);
  }

  MPI_Gatherv(local_data_.data(), local_size, MPI_INT, GetOutput().data(), recv_counts.data(), displacements.data(),
              MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool LuchnikovEGenerTransmFromAllToOneGatherMPI::PostProcessingImpl() {
  if (rank_ == 0) {
    std::sort(GetOutput().begin(), GetOutput().end());
  }
  return true;
}

}  // namespace luchnikov_e_gener_transm_from_all_to_one_gather
