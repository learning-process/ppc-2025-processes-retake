#include "marov_radix_sort_double/seq/include/ops_seq.hpp"

#include <cstdint>
#include <cstring>
#include <vector>

#include "marov_radix_sort_double/common/include/common.hpp"

namespace marov_radix_sort_double {

namespace {

// Преобразование double в uint64_t для сортировки
uint64_t DoubleToSortableUint64(double val) {
  uint64_t bits = 0;
  std::memcpy(&bits, &val, sizeof(double));
  // Инвертируем биты для отрицательных чисел
  if ((bits >> 63) != 0) {
    bits = ~bits;
  } else {
    bits |= (1ULL << 63);
  }
  return bits;
}

// Обратное преобразование
double SortableUint64ToDouble(uint64_t bits) {
  // Восстанавливаем исходное представление
  if ((bits >> 63) != 0) {
    bits &= ~(1ULL << 63);
  } else {
    bits = ~bits;
  }
  double val = 0;
  std::memcpy(&val, &bits, sizeof(double));
  return val;
}

// Поразрядная сортировка для массива double
void RadixSortDoubles(std::vector<double>& data) {
  if (data.size() <= 1) {
    return;
  }

  std::vector<uint64_t> keys(data.size());
  for (size_t i = 0; i < data.size(); ++i) {
    keys[i] = DoubleToSortableUint64(data[i]);
  }

  const int k_radix = 256;
  std::vector<uint64_t> temp(data.size());

  for (int shift = 0; shift < 64; shift += 8) {
    std::vector<size_t> count(k_radix + 1, 0);

    for (uint64_t key : keys) {
      uint8_t digit = (key >> shift) & 0xFF;
      ++count[digit + 1];
    }

    for (int i = 0; i < k_radix; ++i) {
      count[i + 1] += count[i];
    }

    for (uint64_t key : keys) {
      uint8_t digit = (key >> shift) & 0xFF;
      size_t pos = count[digit];
      temp[pos] = key;
      ++count[digit];
    }

    keys.swap(temp);
  }

  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = SortableUint64ToDouble(keys[i]);
  }
}

}  // namespace

MarovRadixSortDoubleSEQ::MarovRadixSortDoubleSEQ(const InType& in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool MarovRadixSortDoubleSEQ::ValidationImpl() {
  return true;
}

bool MarovRadixSortDoubleSEQ::PreProcessingImpl() {
  return true;
}

bool MarovRadixSortDoubleSEQ::RunImpl() {
  auto& input = GetInput();
  RadixSortDoubles(input);
  GetOutput() = input;
  return true;
}

bool MarovRadixSortDoubleSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace marov_radix_sort_double
