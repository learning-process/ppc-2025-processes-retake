#include "yushkova_p_radix_sort_with_simple_merge/seq/include/ops_seq.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
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
  std::array<std::size_t, 256> freq{};
  freq.fill(0);

  for (const std::uint64_t key : keys) {
    const std::size_t bucket = static_cast<std::size_t>((key >> shift) & 0xFFULL);
    ++freq[bucket];
  }

  std::array<std::size_t, 256> position{};
  position[0] = 0;
  for (std::size_t i = 1; i < position.size(); ++i) {
    position[i] = position[i - 1] + freq[i - 1];
  }

  for (const std::uint64_t key : keys) {
    const std::size_t bucket = static_cast<std::size_t>((key >> shift) & 0xFFULL);
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

}  // namespace

YushkovaPRadixSortWithSimpleMergeSEQ::YushkovaPRadixSortWithSimpleMergeSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  std::get<0>(GetOutput()).clear();
  std::get<1>(GetOutput()) = -1;
}

bool YushkovaPRadixSortWithSimpleMergeSEQ::ValidationImpl() {
  return true;
}

bool YushkovaPRadixSortWithSimpleMergeSEQ::PreProcessingImpl() {
  sorted_data_.clear();
  return true;
}

bool YushkovaPRadixSortWithSimpleMergeSEQ::RunImpl() {
  sorted_data_ = GetInput();
  RadixSortDoubleVector(sorted_data_);
  return true;
}

bool YushkovaPRadixSortWithSimpleMergeSEQ::PostProcessingImpl() {
  std::get<0>(GetOutput()) = sorted_data_;
  std::get<1>(GetOutput()) = 0;
  return true;
}

}  // namespace yushkova_p_radix_sort_with_simple_merge
