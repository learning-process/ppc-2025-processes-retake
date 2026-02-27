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

struct MpiCtx {
  int rank = 0;
  int size = 0;
  int n = 0;
  int block = 0;
  int padded_n = 0;
};

enum class NetKind : std::uint8_t {
  kMergeSort,
  kMerge,
};

struct NetFrame {
  NetKind kind{};
  int lo = 0;
  int n = 0;
  int r = 0;      // only for merge
  int stage = 0;  // 0=descend, 1=emit/post
};

inline void PushMergeSort(std::vector<NetFrame> &st, int lo, int n, int stage = 0) {
  st.push_back(NetFrame{.kind = NetKind::kMergeSort, .lo = lo, .n = n, .r = 0, .stage = stage});
}

inline void PushMerge(std::vector<NetFrame> &st, int lo, int n, int r, int stage = 0) {
  st.push_back(NetFrame{.kind = NetKind::kMerge, .lo = lo, .n = n, .r = r, .stage = stage});
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

inline void HandleMergeSortFrame(const NetFrame &f, std::vector<NetFrame> &st) {
  if (f.n <= 1) {
    return;
  }

  const int m = f.n / 2;
  if (f.stage == 0) {
    PushMergeSort(st, f.lo, f.n, 1);
    PushMergeSort(st, f.lo + m, m);
    PushMergeSort(st, f.lo, m);
  } else {
    PushMerge(st, f.lo, f.n, 1);
  }
}

inline void EmitMergeComparators(int lo, int n, int r, std::vector<std::pair<int, int>> &comps) {
  const int m = r * 2;
  for (int i = lo + r; i + r < lo + n; i += m) {
    comps.emplace_back(i, i + r);
  }
}

inline void HandleMergeFrame(const NetFrame &f, std::vector<NetFrame> &st, std::vector<std::pair<int, int>> &comps) {
  const int m = f.r * 2;
  if (m >= f.n) {
    comps.emplace_back(f.lo, f.lo + f.r);
    return;
  }

  if (f.stage == 0) {
    PushMerge(st, f.lo, f.n, f.r, 1);
    PushMerge(st, f.lo + f.r, f.n, m);
    PushMerge(st, f.lo, f.n, m);
    return;
  }

  EmitMergeComparators(f.lo, f.n, f.r, comps);
}

std::vector<std::pair<int, int>> BuildOddEvenMergeNetwork(int n) {
  std::vector<std::pair<int, int>> comps;
  if (n <= 1) {
    return comps;
  }

  std::vector<NetFrame> st;
  PushMergeSort(st, 0, n);

  while (!st.empty()) {
    NetFrame f = st.back();
    st.pop_back();

    if (f.kind == NetKind::kMergeSort) {
      HandleMergeSortFrame(f, st);
    } else {
      HandleMergeFrame(f, st, comps);
    }
  }

  return comps;
}

MpiCtx MakeCtx(const InType &input) {
  MpiCtx ctx;
  MPI_Comm_rank(MPI_COMM_WORLD, &ctx.rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ctx.size);

  if (ctx.rank == 0) {
    ctx.n = static_cast<int>(input.size());
  }
  MPI_Bcast(&ctx.n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  ctx.block = (ctx.n + ctx.size - 1) / ctx.size;
  ctx.padded_n = ctx.block * ctx.size;
  return ctx;
}

std::vector<int> MakePaddedInputOnRoot(const InType &input, const MpiCtx &ctx) {
  if (ctx.rank != 0) {
    return {};
  }
  std::vector<int> padded(static_cast<std::size_t>(ctx.padded_n), std::numeric_limits<int>::max());
  std::ranges::copy(input, padded.begin());
  return padded;
}

std::vector<int> ScatterToLocal(const std::vector<int> &padded_on_root, const MpiCtx &ctx) {
  std::vector<int> local(static_cast<std::size_t>(ctx.block));
  MPI_Scatter(padded_on_root.data(), ctx.block, MPI_INT, local.data(), ctx.block, MPI_INT, 0, MPI_COMM_WORLD);
  return local;
}

void ApplyBatcherNetworkIfPowerOfTwo(const MpiCtx &ctx, std::vector<int> *local) {
  const std::vector<std::pair<int, int>> comps = BuildOddEvenMergeNetwork(ctx.size);
  for (const auto &[a, b] : comps) {
    if (ctx.rank == a || ctx.rank == b) {
      const int partner = (ctx.rank == a) ? b : a;
      CompareSplit(ctx.rank, partner, ctx.block, local);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
}

void ApplyOddEvenTransposition(const MpiCtx &ctx, std::vector<int> *local) {
  for (int phase = 0; phase < ctx.size; ++phase) {
    int partner = -1;
    if ((phase % 2) == 0) {
      partner = ((ctx.rank % 2) == 0) ? ctx.rank + 1 : ctx.rank - 1;
    } else {
      partner = ((ctx.rank % 2) == 0) ? ctx.rank - 1 : ctx.rank + 1;
    }

    if (partner >= 0 && partner < ctx.size) {
      CompareSplit(ctx.rank, partner, ctx.block, local);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
}

void GlobalMergeAcrossRanks(const MpiCtx &ctx, std::vector<int> *local) {
  if (ctx.size <= 1) {
    return;
  }
  if (IsPowerOfTwo(ctx.size)) {
    ApplyBatcherNetworkIfPowerOfTwo(ctx, local);
  } else {
    ApplyOddEvenTransposition(ctx, local);
  }
}

std::vector<int> GatherOnRoot(const MpiCtx &ctx, const std::vector<int> &local) {
  std::vector<int> gathered;
  if (ctx.rank == 0) {
    gathered.resize(static_cast<std::size_t>(ctx.padded_n));
  }
  MPI_Gather(local.data(), ctx.block, MPI_INT, gathered.data(), ctx.block, MPI_INT, 0, MPI_COMM_WORLD);
  return gathered;
}

std::vector<int> FinalizeAndBroadcastOutput(const MpiCtx &ctx, const std::vector<int> &gathered_on_root) {
  std::vector<int> out(static_cast<std::size_t>(ctx.n));
  if (ctx.rank == 0) {
    std::copy_n(gathered_on_root.begin(), static_cast<std::size_t>(ctx.n), out.begin());
  }
  MPI_Bcast(out.data(), ctx.n, MPI_INT, 0, MPI_COMM_WORLD);
  return out;
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
  const MpiCtx ctx = MakeCtx(GetInput());

  const std::vector<int> padded = MakePaddedInputOnRoot(GetInput(), ctx);
  std::vector<int> local = ScatterToLocal(padded, ctx);

  local = RadixSortSigned32(local);
  GlobalMergeAcrossRanks(ctx, &local);

  const std::vector<int> gathered = GatherOnRoot(ctx, local);
  GetOutput() = FinalizeAndBroadcastOutput(ctx, gathered);
  return true;
}

bool CheremkhinARadixSortBatcherMPI::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace cheremkhin_a_radix_sort_batcher
