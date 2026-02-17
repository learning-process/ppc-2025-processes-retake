#include "kichanova_k_shellsort_batcher/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

#include "kichanova_k_shellsort_batcher/common/include/common.hpp"
#include "util/include/util.hpp"

namespace kichanova_k_shellsort_batcher {

KichanovaKShellsortBatcherSEQ::KichanovaKShellsortBatcherSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool KichanovaKShellsortBatcherSEQ::ValidationImpl() {
  return GetInput() > 0;
}

bool KichanovaKShellsortBatcherSEQ::PreProcessingImpl() {
  return true;
}

bool KichanovaKShellsortBatcherSEQ::RunImpl() {
  const InType n = GetInput();

  if (n <= 0) {
    return false;
  }

  std::vector<int> data(static_cast<std::size_t>(n));
  std::mt19937 gen(static_cast<unsigned int>(n));
  std::uniform_int_distribution<int> dist(0, 1000000);

  for (int &v : data) {
    v = dist(gen);
  }

  std::vector<int> expected = data;
  std::sort(expected.begin(), expected.end());

  ShellSort(data);

  const auto mid = data.size() / 2;
  std::vector<int> left(data.begin(), data.begin() + static_cast<std::vector<int>::difference_type>(mid));
  std::vector<int> right(data.begin() + static_cast<std::vector<int>::difference_type>(mid), data.end());
  std::vector<int> merged;
  OddEvenBatcherMerge(left, right, merged);
  data.swap(merged);

  if (!std::is_sorted(data.begin(), data.end())) {
    return false;
  }
  if (data != expected) {
    return false;
  }

  std::int64_t checksum = std::accumulate(data.begin(), data.end(), static_cast<std::int64_t>(0));
  GetOutput() = static_cast<OutType>(checksum & 0x7FFFFFFF);

  return true;
}

void KichanovaKShellsortBatcherSEQ::ShellSort(std::vector<int> &arr) {
  const std::size_t n = arr.size();
  if (n < 2) {
    return;
  }

  std::size_t gap = 1;
  while (gap < n / 3) {
    gap = (gap * 3) + 1;
  }

  while (gap > 0) {
    for (std::size_t i = gap; i < n; ++i) {
      const int tmp = arr[i];
      std::size_t j = i;
      while (j >= gap && arr[j - gap] > tmp) {
        arr[j] = arr[j - gap];
        j -= gap;
      }
      arr[j] = tmp;
    }
    gap = (gap - 1) / 3;
  }
}

void KichanovaKShellsortBatcherSEQ::OddEvenBatcherMerge(const std::vector<int> &left,
                                                                    const std::vector<int> &right,
                                                                    std::vector<int> &merged) {
  merged.resize(left.size() + right.size());
  std::merge(left.begin(), left.end(), right.begin(), right.end(), merged.begin());

  for (int j = 0; j < 2; ++j) {
    auto start = static_cast<std::size_t>(j);
    for (std::size_t i = start; i + 1 < merged.size(); i += 2) {
      if (merged[i] > merged[i + 1]) {
        std::swap(merged[i], merged[i + 1]);
      }
    }
  }
}

bool KichanovaKShellsortBatcherSEQ::PostProcessingImpl() {
  return GetOutput() > 0;
}

}  // namespace kichanova_k_shellsort_batcher