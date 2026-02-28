#include "vinyaikina_e_quicksort_simple/seq/include/ops_seq.hpp"

#include <algorithm>
#include <utility>
#include <vector>

#include "vinyaikina_e_quicksort_simple/common/include/common.hpp"

namespace vinyaikina_e_quicksort_simple {

namespace {

std::pair<int, int> Partition(std::vector<int> &arr, int lo, int hi) {
  int pivot = arr[lo + ((hi - lo) / 2)];
  int i = lo;
  int j = hi;
  while (i <= j) {
    while (arr[i] < pivot) {
      i++;
    }
    while (arr[j] > pivot) {
      j--;
    }
    if (i <= j) {
      std::swap(arr[i], arr[j]);
      i++;
      j--;
    }
  }
  return {i, j};
}

void QuickSort(std::vector<int> &arr, int left, int right) {
  std::vector<std::pair<int, int>> stack;
  stack.emplace_back(left, right);
  while (!stack.empty()) {
    auto [lo, hi] = stack.back();
    stack.pop_back();
    if (lo >= hi) {
      continue;
    }
    auto [i, j] = Partition(arr, lo, hi);
    if (lo < j) {
      stack.emplace_back(lo, j);
    }
    if (i < hi) {
      stack.emplace_back(i, hi);
    }
  }
}

}  // namespace

VinyaikinaEQuicksortSimpleSEQ::VinyaikinaEQuicksortSimpleSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool VinyaikinaEQuicksortSimpleSEQ::ValidationImpl() {
  return true;
}

bool VinyaikinaEQuicksortSimpleSEQ::PreProcessingImpl() {
  data_ = GetInput();
  return true;
}

bool VinyaikinaEQuicksortSimpleSEQ::RunImpl() {
  if (data_.size() > 1) {
    QuickSort(data_, 0, static_cast<int>(data_.size()) - 1);
  }
  return true;
}

bool VinyaikinaEQuicksortSimpleSEQ::PostProcessingImpl() {
  GetOutput() = data_;
  return true;
}

}  // namespace vinyaikina_e_quicksort_simple
