#include "cheremkhin_a_radix_sort_batcher/seq/include/ops_seq.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "cheremkhin_a_radix_sort_batcher/common/include/common.hpp"

namespace cheremkhin_a_radix_sort_batcher {

namespace {

constexpr std::uint32_t kSignMask = 0x80000000U;
constexpr std::size_t kRadix = 256;

inline std::uint8_t GetByteForRadixSort(int v, int byte_idx) {
  const std::uint32_t key = static_cast<std::uint32_t>(v) ^ kSignMask;
  return static_cast<std::uint8_t>((key >> (static_cast<std::uint32_t>(byte_idx) * 8U)) & 0xFFU);
}

std::vector<int> RadixSortSigned32(const std::vector<int> &in) {
  if (in.empty()) {
    return {};
  }
  if (in.size() == 1) {
    return std::vector<int>{in[0]};
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

}  // namespace

CheremkhinARadixSortBatcherSEQ::CheremkhinARadixSortBatcherSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool CheremkhinARadixSortBatcherSEQ::ValidationImpl() {
  return !GetInput().empty();
}

bool CheremkhinARadixSortBatcherSEQ::PreProcessingImpl() {
  return true;
}

bool CheremkhinARadixSortBatcherSEQ::RunImpl() {
  GetOutput() = RadixSortSigned32(GetInput());
  return true;
}

bool CheremkhinARadixSortBatcherSEQ::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace cheremkhin_a_radix_sort_batcher
