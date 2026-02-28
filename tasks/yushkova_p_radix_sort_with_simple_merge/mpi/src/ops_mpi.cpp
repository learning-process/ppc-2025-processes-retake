#include "yushkova_p_radix_sort_with_simple_merge/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "yushkova_p_radix_sort_with_simple_merge/common/include/common.hpp"

namespace yushkova_p_radix_sort_with_simple_merge {
namespace {

std::vector<std::uint64_t> BuildKeyBuffer(const std::vector<double> &data) {
  std::vector<std::uint64_t> keys(data.size());
  for (std::size_t i = 0; i < data.size(); ++i) {
    keys[i] = EncodeDoubleKey(data[i]);
  }
  return keys;
}

void ByteCountingPass(std::vector<std::uint64_t> &keys, std::vector<std::uint64_t> &temp, int shift) {
  constexpr std::size_t kBuckets = 256;
  std::vector<std::size_t> freq(kBuckets, 0);

  for (const std::uint64_t key : keys) {
    const auto bucket = static_cast<std::size_t>((key >> shift) & 0xFFULL);
    ++freq[bucket];
  }

  std::vector<std::size_t> position(kBuckets, 0);
  position[0] = 0;
  for (std::size_t i = 1; i < position.size(); ++i) {
    position[i] = position[i - 1] + freq[i - 1];
  }

  for (const std::uint64_t key : keys) {
    const auto bucket = static_cast<std::size_t>((key >> shift) & 0xFFULL);
    temp[position[bucket]] = key;
    ++position[bucket];
  }

  keys.swap(temp);
}

void RadixSortDoubleVector(std::vector<double> &data) {
  if (data.size() < 2) {
    return;
  }

  std::vector<std::uint64_t> keys = BuildKeyBuffer(data);
  std::vector<std::uint64_t> temp(keys.size());

  for (int shift = 0; shift < 64; shift += 8) {
    ByteCountingPass(keys, temp, shift);
  }

  for (std::size_t i = 0; i < data.size(); ++i) {
    data[i] = DecodeDoubleKey(keys[i]);
  }
}

std::vector<double> MergeTwoSortedVectors(const std::vector<double> &left, const std::vector<double> &right) {
  std::vector<double> merged;
  merged.reserve(left.size() + right.size());

  std::size_t i = 0;
  std::size_t j = 0;
  while (i < left.size() && j < right.size()) {
    if (left[i] <= right[j]) {
      merged.push_back(left[i]);
      ++i;
    } else {
      merged.push_back(right[j]);
      ++j;
    }
  }

  while (i < left.size()) {
    merged.push_back(left[i]);
    ++i;
  }
  while (j < right.size()) {
    merged.push_back(right[j]);
    ++j;
  }

  return merged;
}

void BuildScatterPlan(int total_size, int world_size, std::vector<int> &counts, std::vector<int> &displs) {
  counts.assign(world_size, 0);
  displs.assign(world_size, 0);

  const int base = total_size / world_size;
  const int extra = total_size % world_size;

  int offset = 0;
  for (int rank = 0; rank < world_size; ++rank) {
    counts[rank] = base + ((rank < extra) ? 1 : 0);
    displs[rank] = offset;
    offset += counts[rank];
  }
}

bool TryReceiveAndMergeAtStep(int rank, int world_size, int step, std::vector<double> &local_data) {
  if ((rank % (2 * step)) != 0) {
    return false;
  }

  const int sender = rank + step;
  if (sender >= world_size) {
    return false;
  }

  int received_count = 0;
  MPI_Recv(&received_count, 1, MPI_INT, sender, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  std::vector<double> received_data(received_count);
  if (received_count > 0) {
    MPI_Recv(received_data.data(), received_count, MPI_DOUBLE, sender, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  local_data = MergeTwoSortedVectors(local_data, received_data);
  return true;
}

bool TrySendAndFinishAtStep(int rank, int step, const std::vector<double> &local_data) {
  if ((rank % (2 * step)) != step) {
    return false;
  }

  const int receiver = rank - step;
  const int send_count = static_cast<int>(local_data.size());
  MPI_Send(&send_count, 1, MPI_INT, receiver, 0, MPI_COMM_WORLD);
  if (send_count > 0) {
    MPI_Send(local_data.data(), send_count, MPI_DOUBLE, receiver, 1, MPI_COMM_WORLD);
  }
  return true;
}

void MergeByTree(int rank, int world_size, std::vector<double> &local_data) {
  for (int step = 1; step < world_size; step *= 2) {
    if (TrySendAndFinishAtStep(rank, step, local_data)) {
      break;
    }
    TryReceiveAndMergeAtStep(rank, world_size, step, local_data);
  }
}

}  // namespace

YushkovaPRadixSortWithSimpleMergeMPI::YushkovaPRadixSortWithSimpleMergeMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  std::get<0>(GetOutput()).clear();
  std::get<1>(GetOutput()) = -1;
}

bool YushkovaPRadixSortWithSimpleMergeMPI::ValidationImpl() {
  int initialized = 0;
  MPI_Initialized(&initialized);
  return (initialized != 0) && (GetDynamicTypeOfTask() == GetStaticTypeOfTask());
}

bool YushkovaPRadixSortWithSimpleMergeMPI::PreProcessingImpl() {
  merged_result_.clear();
  return true;
}

bool YushkovaPRadixSortWithSimpleMergeMPI::RunImpl() {
  int rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int total_size = 0;
  if (rank == 0) {
    total_size = static_cast<int>(GetInput().size());
  }
  MPI_Bcast(&total_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> counts;
  std::vector<int> displs;
  BuildScatterPlan(total_size, world_size, counts, displs);

  std::vector<double> local_data(counts[rank]);

  MPI_Scatterv(rank == 0 ? GetInput().data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE, local_data.data(),
               counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

  RadixSortDoubleVector(local_data);

  MergeByTree(rank, world_size, local_data);

  if (rank == 0) {
    merged_result_ = std::move(local_data);
  }

  std::get<1>(GetOutput()) = rank;
  return true;
}

bool YushkovaPRadixSortWithSimpleMergeMPI::PostProcessingImpl() {
  if (!merged_result_.empty()) {
    std::get<0>(GetOutput()) = merged_result_;
  } else {
    std::get<0>(GetOutput()).clear();
  }
  return true;
}

}  // namespace yushkova_p_radix_sort_with_simple_merge
