#include "denisov_a_quick_sort_simple_merging/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "denisov_a_quick_sort_simple_merging/common/include/common.hpp"

namespace denisov_a_quick_sort_simple_merging {

DenisovAQuickSortMergeSEQ::DenisovAQuickSortMergeSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool DenisovAQuickSortMergeSEQ::ValidationImpl() {
  return GetOutput().empty();
}

bool DenisovAQuickSortMergeSEQ::PreProcessingImpl() {
  const auto &src = GetInput();
  auto &dst = GetOutput();

  dst = src;

  if (dst.size() != src.size()) {
    return false;
  }

  for (size_t idx = 0; idx < dst.size(); idx++) {
    if (dst[idx] != src[idx]) {
      return false;
    }
  }

  return true;
}

bool DenisovAQuickSortMergeSEQ::RunImpl() {
  auto &arr = GetOutput();

  if (arr.empty() || arr.size() == 1) {
    return true;
  }

  size_t half = arr.size() / 2;

  std::vector<int> left_part(arr.begin(), arr.begin() + static_cast<ptrdiff_t>(half));
  std::vector<int> right_part(arr.begin() + static_cast<ptrdiff_t>(half), arr.end());

  QuickSort(left_part, 0, static_cast<int>(left_part.size()) - 1);
  QuickSort(right_part, 0, static_cast<int>(right_part.size()) - 1);

  arr = Merge(left_part, right_part);
  return true;
}

bool DenisovAQuickSortMergeSEQ::PostProcessingImpl() {
  const auto &out = GetOutput();
  const auto &in = GetInput();

  if (out.empty()) {
    return in.empty();
  }

  if (!std::ranges::is_sorted(out)) {
    return false;
  }

  if (out.size() != in.size()) {
    return false;
  }

  int64_t sum_in = 0;
  int64_t sum_out = 0;

  for (int v : in) {
    sum_in += v;
  }
  for (int v : out) {
    sum_out += v;
  }

  return sum_in == sum_out;
}

// NOLINTNEXTLINE(misc-no-recursion)
void DenisovAQuickSortMergeSEQ::QuickSort(std::vector<int> &data, int begin, int end) {
  if (begin >= end) {
    return;
  }

  int pivot = data[(begin + end) / 2];
  int i = begin;
  int j = end;

  while (i <= j) {
    while (data[i] < pivot) {
      i++;
    }
    while (data[j] > pivot) {
      j--;
    }

    if (i <= j) {
      std::swap(data[i], data[j]);
      i++;
      j--;
    }
  }

  QuickSort(data, begin, j);
  QuickSort(data, i, end);
}

std::vector<int> DenisovAQuickSortMergeSEQ::Merge(const std::vector<int> &left_block,
                                                  const std::vector<int> &right_block) {
  std::vector<int> res;
  res.reserve(left_block.size() + right_block.size());

  size_t i = 0;
  size_t j = 0;

  while (i < left_block.size() && j < right_block.size()) {
    if (left_block[i] <= right_block[j]) {
      res.push_back(left_block[i]);
      i++;
    } else {
      res.push_back(right_block[j]);
      j++;
    }
  }

  while (i < left_block.size()) {
    res.push_back(left_block[i]);
    i++;
  }

  while (j < right_block.size()) {
    res.push_back(right_block[j]);
    j++;
  }

  return res;
}

}  // namespace denisov_a_quick_sort_simple_merging
