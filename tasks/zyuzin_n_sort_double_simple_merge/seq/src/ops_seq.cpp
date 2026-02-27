#include "zyuzin_n_sort_double_simple_merge/seq/include/ops_seq.hpp"

#include <cstdint>
#include <cstring>
#include <vector>

#include "zyuzin_n_sort_double_simple_merge/common/include/common.hpp"

namespace zyuzin_n_sort_double_simple_merge {

ZyuzinNSortDoubleWithSimpleMergeSEQ::ZyuzinNSortDoubleWithSimpleMergeSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<double>();
}

bool ZyuzinNSortDoubleWithSimpleMergeSEQ::ValidationImpl() {
  return true;
}

bool ZyuzinNSortDoubleWithSimpleMergeSEQ::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

std::vector<std::uint64_t> ZyuzinNSortDoubleWithSimpleMergeSEQ::ConvertDoublesToBits(const std::vector<double> &data) {
  std::vector<std::uint64_t> bits(data.size(), 0);
  for (size_t i = 0; i < data.size(); ++i) {
    std::uint64_t x = 0;
    std::memcpy(&x, &data[i], sizeof(double));
    if ((x >> 63) == 0) {
      x ^= 0x8000000000000000;
    } else {
      x = ~x;
    }
    bits[i] = x;
  }
  return bits;
}

std::vector<std::uint64_t> ZyuzinNSortDoubleWithSimpleMergeSEQ::SortBits(const std::vector<std::uint64_t> &bits) {
  const int radix = 256;
  const std::size_t n = bits.size();
  if (n == 0) {
    return {};
  }

  std::vector<std::uint64_t> source = bits;
  std::vector<std::uint64_t> destination(n);

  for (int byte_idx = 0; byte_idx < 8; ++byte_idx) {
    int shift = byte_idx * 8;

    std::vector<std::size_t> count(radix, 0);
    for (std::uint64_t value : source) {
      std::size_t digit = (value >> shift) & 0xFF;
      ++count[digit];
    }

    for (int i = 1; i < radix; ++i) {
      count[i] += count[i - 1];
    }

    for (std::size_t i = n; i > 0; --i) {
      std::uint64_t value = source[i - 1];
      std::size_t digit = (value >> shift) & 0xFF;
      destination[--count[digit]] = value;
    }

    source.swap(destination);
  }

  return source;
}

std::vector<double> ZyuzinNSortDoubleWithSimpleMergeSEQ::ConvertBitsToDoubles(const std::vector<std::uint64_t> &data) {
  std::vector<std::uint64_t> bits = data;
  std::vector<double> doubles(data.size(), 0.0);
  for (size_t i = 0; i < data.size(); ++i) {
    if ((bits[i] >> 63) == 0) {
      bits[i] = ~bits[i];
    } else {
      bits[i] ^= 0x8000000000000000;
    }
    std::memcpy(&doubles[i], &bits[i], sizeof(double));
  }
  return doubles;
}

bool ZyuzinNSortDoubleWithSimpleMergeSEQ::RunImpl() {
  const auto &input = GetInput();
  std::vector<double> data = input;
  auto bits = ConvertDoublesToBits(data);
  auto sorted_bits = SortBits(bits);
  auto sorted_data = ConvertBitsToDoubles(sorted_bits);
  GetOutput() = sorted_data;
  return true;
}

bool ZyuzinNSortDoubleWithSimpleMergeSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace zyuzin_n_sort_double_simple_merge
