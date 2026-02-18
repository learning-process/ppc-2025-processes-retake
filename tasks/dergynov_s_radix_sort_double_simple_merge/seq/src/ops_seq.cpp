#include "dergynov_s_radix_sort_double_simple_merge/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstring>
#include <vector>

namespace dergynov_s_radix_sort_double_simple_merge {
namespace {

void RadixSortDoubles(std::vector<double> &data) {
  if (data.size() <= 1) return;

  std::vector<uint64_t> keys(data.size());
  for (size_t i = 0; i < data.size(); ++i) {
    keys[i] = DoubleToSortableUint64(data[i]);
  }

  const int kRadix = 256;
  std::vector<uint64_t> temp(data.size());

  for (int shift = 0; shift < 64; shift += 8) {
    std::vector<size_t> count(kRadix + 1, 0);

    for (size_t i = 0; i < keys.size(); ++i) {
      uint8_t digit = (keys[i] >> shift) & 0xFF;
      ++count[digit + 1];
    }

    for (int i = 0; i < kRadix; ++i) {
      count[i + 1] += count[i];
    }

    for (size_t i = 0; i < keys.size(); ++i) {
      uint8_t digit = (keys[i] >> shift) & 0xFF;
      size_t pos = count[digit];
      temp[pos] = keys[i];
      count[digit] = pos + 1;
    }

    keys.swap(temp);
  }

  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = SortableUint64ToDouble(keys[i]);
  }
}

}  // namespace

DergynovSRadixSortDoubleSimpleMergeSEQ::DergynovSRadixSortDoubleSimpleMergeSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  sorted_.clear();
  std::get<0>(GetOutput()).clear();
  std::get<1>(GetOutput()) = -1;
}

bool DergynovSRadixSortDoubleSimpleMergeSEQ::ValidationImpl() {
  return true;
}

bool DergynovSRadixSortDoubleSimpleMergeSEQ::PreProcessingImpl() {
  sorted_.clear();
  return true;
}

bool DergynovSRadixSortDoubleSimpleMergeSEQ::RunImpl() {
  sorted_ = GetInput();
  RadixSortDoubles(sorted_);
  return true;
}

bool DergynovSRadixSortDoubleSimpleMergeSEQ::PostProcessingImpl() {
  std::get<0>(GetOutput()) = sorted_;
  std::get<1>(GetOutput()) = 0;
  return true;
}

}  // namespace dergynov_s_radix_sort_double_simple_merge
