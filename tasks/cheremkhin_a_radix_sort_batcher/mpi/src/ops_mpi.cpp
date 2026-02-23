#include "cheremkhin_a_radix_sort_batcher/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "cheremkhin_a_radix_sort_batcher/common/include/common.hpp"

namespace cheremkhin_a_radix_sort_batcher {

namespace {

constexpr std::uint32_t kSignMask = 0x80000000U;
constexpr std::size_t kRadix = 256;

inline bool IsPowerOfTwo(int x) {
  return x > 0 && (x & (x - 1)) == 0;
}

inline std::uint8_t GetByteForRadixSort(int v, int byte_idx) {
  const std::uint32_t key = static_cast<std::uint32_t>(v) ^ kSignMask;
  return static_cast<std::uint8_t>((key >> (static_cast<std::uint32_t>(byte_idx) * 8U)) & 0xFFU);
}

std::vector<int> RadixSortSigned32(const std::vector<int> &in) {
  if (in.size() <= 1) {
    return in;
  }

  std::vector<int> a;
  a.reserve(in.size());
  a.assign(in.begin(), in.end());
  std::vector<int> tmp(in.size());

  for (int byte_idx = 0; byte_idx < 4; ++byte_idx) {
    std::vector<std::size_t> cnt(kRadix, 0);
    for (int v : a) {
      ++cnt[GetByteForRadixSort(v, byte_idx)];
    }

    std::vector<std::size_t> pos(kRadix);
    std::size_t sum = 0;
    for (std::size_t i = 0; i < kRadix; ++i) {
      pos[i] = sum;
      sum += cnt[i];
    }

    for (int v : a) {
      const std::uint8_t b = GetByteForRadixSort(v, byte_idx);
      tmp[pos[b]++] = v;
    }

    a.swap(tmp);
  }

  return a;
}

std::vector<int> MergeSorted(const std::vector<int> &a, const std::vector<int> &b) {
  std::vector<int> out;
  out.resize(a.size() + b.size());
  std::ranges::merge(a, b, out.begin());
  return out;
}

void CompareSplit(int rank, int partner, int keep_cnt, std::vector<int> *local) {
  std::vector<int> recv(static_cast<std::size_t>(keep_cnt));

  MPI_Sendrecv(local->data(), keep_cnt, MPI_INT, partner, 0, recv.data(), keep_cnt, MPI_INT, partner, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);

  std::vector<int> merged = MergeSorted(*local, recv);

  std::vector<int> new_local(static_cast<std::size_t>(keep_cnt));
  if (rank < partner) {
    std::copy_n(merged.begin(), static_cast<std::size_t>(keep_cnt), new_local.begin());
  } else {
    std::copy_n(merged.end() - static_cast<std::ptrdiff_t>(keep_cnt), static_cast<std::size_t>(keep_cnt),
                new_local.begin());
  }
  local->swap(new_local);
}

void OddEvenMerge(int lo, int n, int r, std::vector<std::pair<int, int>> *comps) {
  const int m = r * 2;
  if (m < n) {
    OddEvenMerge(lo, n, m, comps);
    OddEvenMerge(lo + r, n, m, comps);
    for (int i = lo + r; i + r < lo + n; i += m) {
      comps->emplace_back(i, i + r);
    }
  } else {
    comps->emplace_back(lo, lo + r);
  }
}

void OddEvenMergeSort(int lo, int n, std::vector<std::pair<int, int>> *comps) {
  if (n > 1) {
    const int m = n / 2;
    OddEvenMergeSort(lo, m, comps);
    OddEvenMergeSort(lo + m, m, comps);
    OddEvenMerge(lo, n, 1, comps);
  }
}

}  // namespace

CheremkhinARadixSortBatcherMPI::CheremkhinARadixSortBatcherMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool CheremkhinARadixSortBatcherMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    return !GetInput().empty();
  }
  return true;
}

bool CheremkhinARadixSortBatcherMPI::PreProcessingImpl() {
  return true;
}

bool CheremkhinARadixSortBatcherMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int n = 0;
  if (rank == 0) {
    n = static_cast<int>(GetInput().size());
  }
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  const int block = (n + size - 1) / size;
  const int padded_n = block * size;

  std::vector<int> padded;
  if (rank == 0) {
    padded.assign(static_cast<std::size_t>(padded_n), std::numeric_limits<int>::max());
    std::copy(GetInput().begin(), GetInput().end(), padded.begin());
  }

  std::vector<int> local(static_cast<std::size_t>(block));
  MPI_Scatter(padded.data(), block, MPI_INT, local.data(), block, MPI_INT, 0, MPI_COMM_WORLD);

  local = RadixSortSigned32(local);

  if (size > 1) {
    if (IsPowerOfTwo(size)) {
      std::vector<std::pair<int, int>> comps;
      OddEvenMergeSort(0, size, &comps);

      for (const auto &[a, b] : comps) {
        if (rank == a || rank == b) {
          const int partner = (rank == a) ? b : a;
          CompareSplit(rank, partner, block, &local);
        }
        MPI_Barrier(MPI_COMM_WORLD);
      }
    } else {
      for (int phase = 0; phase < size; ++phase) {
        int partner = -1;
        if ((phase % 2) == 0) {
          partner = ((rank % 2) == 0) ? rank + 1 : rank - 1;
        } else {
          partner = ((rank % 2) == 0) ? rank - 1 : rank + 1;
        }
        if (partner >= 0 && partner < size) {
          CompareSplit(rank, partner, block, &local);
        }
        MPI_Barrier(MPI_COMM_WORLD);
      }
    }
  }

  std::vector<int> gathered;
  if (rank == 0) {
    gathered.resize(static_cast<std::size_t>(padded_n));
  }
  MPI_Gather(local.data(), block, MPI_INT, gathered.data(), block, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> out(static_cast<std::size_t>(n));
  if (rank == 0) {
    std::copy_n(gathered.begin(), static_cast<std::size_t>(n), out.begin());
  }
  MPI_Bcast(out.data(), n, MPI_INT, 0, MPI_COMM_WORLD);

  GetOutput() = out;
  return true;
}

bool CheremkhinARadixSortBatcherMPI::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace cheremkhin_a_radix_sort_batcher
