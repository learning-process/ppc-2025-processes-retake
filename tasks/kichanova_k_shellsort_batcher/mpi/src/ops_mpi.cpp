#include "kichanova_k_shellsort_batcher/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

#include "kichanova_k_shellsort_batcher/common/include/common.hpp"
#include "util/include/util.hpp"

namespace kichanova_k_shellsort_batcher {

KichanovaKShellsortBatcherMPI::KichanovaKShellsortBatcherMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool KichanovaKShellsortBatcherMPI::ValidationImpl() {
  return (GetInput() > 0) && (GetOutput() == 0);
}

bool KichanovaKShellsortBatcherMPI::PreProcessingImpl() {
  int initialized = 0;
  MPI_Initialized(&initialized);
  return initialized != 0 && GetInput() > 0;
}

bool KichanovaKShellsortBatcherMPI::RunImpl() {
  const InType n = GetInput();
  if (n <= 0) {
    return false;
  }

  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const InType base = n / size;
  const InType rem = n % size;
  const InType local_n = base + (rank < rem ? 1 : 0);

  std::vector<int> local_data(local_n);

  std::mt19937 rng(static_cast<unsigned int>(n));
  std::uniform_int_distribution<int> dist(0, 1000000);

  InType offset = 0;
  for (int i = 0; i < rank; ++i) {
    InType proc_n = base + (i < rem ? 1 : 0);
    offset += proc_n;
  }

  for (InType i = 0; i < offset; ++i) {
    (void)dist(rng);
  }

  for (InType i = 0; i < local_n; ++i) {
    local_data[i] = dist(rng);
  }

  ShellSort(local_data);

  for (int phase = 0; phase < size; ++phase) {
    int partner = -1;

    if (phase % 2 == 0) {
      if (rank % 2 == 0) {
        partner = rank + 1;
      } else {
        partner = rank - 1;
      }
    } else {
      if (rank % 2 == 1) {
        partner = rank + 1;
      } else {
        partner = rank - 1;
      }
    }

    if (partner >= 0 && partner < size) {
      ExchangeAndMerge(local_data, partner, rank, 1000 + phase);
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }

  std::int64_t local_checksum = 0;
  for (const auto &val : local_data) {
    local_checksum += val;
  }

  std::int64_t global_checksum = 0;
  MPI_Allreduce(&local_checksum, &global_checksum, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);

  GetOutput() = static_cast<OutType>(global_checksum & 0x7FFFFFFF);

  return true;
}

void KichanovaKShellsortBatcherMPI::ShellSort(std::vector<int> &arr) {
  const std::size_t n = arr.size();
  if (n < 2) {
    return;
  }

  std::size_t gap = 1;
  while (gap < n / 3) {
    gap = (gap * 3) + 1;
  }

  while (gap > 0) {
    for (std::size_t i = gap; i < n; ++i) {
      const int tmp = arr[i];
      std::size_t j = i;
      while (j >= gap && arr[j - gap] > tmp) {
        arr[j] = arr[j - gap];
        j -= gap;
      }
      arr[j] = tmp;
    }
    gap = (gap - 1) / 3;
  }
}

void KichanovaKShellsortBatcherMPI::ExchangeAndMerge(std::vector<int> &local_data, int partner, int rank,
                                                                 int tag) {
  if (partner == MPI_PROC_NULL || partner == rank) {
    return;
  }

  int send_size = static_cast<int>(local_data.size());
  int recv_size = 0;

  MPI_Sendrecv(&send_size, 1, MPI_INT, partner, tag * 2, &recv_size, 1, MPI_INT, partner, tag * 2, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);

  std::vector<int> recv_buffer(recv_size);

  MPI_Sendrecv(local_data.data(), send_size, MPI_INT, partner, tag * 2 + 1, recv_buffer.data(), recv_size, MPI_INT,
               partner, tag * 2 + 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  std::vector<int> merged(send_size + recv_size);
  std::merge(local_data.begin(), local_data.end(), recv_buffer.begin(), recv_buffer.end(), merged.begin());

  if (rank < partner) {
    local_data.assign(merged.begin(), merged.begin() + send_size);
  } else {
    local_data.assign(merged.end() - send_size, merged.end());
  }
}

bool KichanovaKShellsortBatcherMPI::PostProcessingImpl() {
  return GetOutput() > 0;
}

}  // namespace kichanova_k_shellsort_batcher