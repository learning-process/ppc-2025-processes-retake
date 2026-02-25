#include "ops_seq.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

void radix_sort_double(std::vector<double> &arr) {
  if (arr.empty())
    return;

  // Преобразование double в uint64_t для сортировки
  std::vector<uint64_t> bits(arr.size());
  for (size_t i = 0; i < arr.size(); i++) {
    uint64_t val;
    std::memcpy(&val, &arr[i], sizeof(double));
    // Инвертируем бит знака для правильной сортировки
    if (val >> 63) {
      val = ~val;
    } else {
      val |= 1ULL << 63;
    }
    bits[i] = val;
  }

  // Поразрядная сортировка по байтам
  const int BITS_PER_PASS = 8;
  const int RADIX = 1 << BITS_PER_PASS;
  const int PASSES = sizeof(uint64_t) * 8 / BITS_PER_PASS;

  std::vector<uint64_t> temp(arr.size());

  for (int pass = 0; pass < PASSES; pass++) {
    std::vector<int> count(RADIX, 0);

    // Подсчёт
    for (size_t i = 0; i < bits.size(); i++) {
      int digit = (bits[i] >> (pass * BITS_PER_PASS)) & (RADIX - 1);
      count[digit]++;
    }

    // Преобразование в позиции
    for (int i = 1; i < RADIX; i++) {
      count[i] += count[i - 1];
    }

    // Распределение
    for (int i = bits.size() - 1; i >= 0; i--) {
      int digit = (bits[i] >> (pass * BITS_PER_PASS)) & (RADIX - 1);
      temp[--count[digit]] = bits[i];
    }

    bits.swap(temp);
  }

  // Обратное преобразование
  for (size_t i = 0; i < arr.size(); i++) {
    uint64_t val = bits[i];
    if (val >> 63) {
      val &= ~(1ULL << 63);
    } else {
      val = ~val;
    }
    std::memcpy(&arr[i], &val, sizeof(double));
  }
}
