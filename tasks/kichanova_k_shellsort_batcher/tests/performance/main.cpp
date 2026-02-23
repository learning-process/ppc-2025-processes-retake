#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

#include "kichanova_k_shellsort_batcher/common/include/common.hpp"
#include "kichanova_k_shellsort_batcher/mpi/include/ops_mpi.hpp"
#include "kichanova_k_shellsort_batcher/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace kichanova_k_shellsort_batcher {

class KichanovaKShellsortBatcherPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 10000;
  InType input_size_ = 0;

  void SetUp() override {
    input_size_ = kCount_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    std::vector<int> data(static_cast<std::size_t>(input_size_));

    std::mt19937 rng(static_cast<unsigned int>(input_size_));
    std::uniform_int_distribution<int> dist(0, 1000000);

    for (int &v : data) {
      v = dist(rng);
    }

    const std::size_t n = data.size();
    if (n >= 2) {
      std::size_t gap = 1;
      while (gap < n / 3) {
        gap = (gap * 3) + 1;
      }

      while (gap > 0) {
        for (std::size_t i = gap; i < n; ++i) {
          const int tmp = data[i];
          std::size_t j = i;
          while (j >= gap && data[j - gap] > tmp) {
            data[j] = data[j - gap];
            j -= gap;
          }
          data[j] = tmp;
        }
        gap = (gap - 1) / 3;
      }
    }

    const auto mid = data.size() / 2;
    std::vector<int> left(data.begin(), data.begin() + static_cast<std::vector<int>::difference_type>(mid));
    std::vector<int> right(data.begin() + static_cast<std::vector<int>::difference_type>(mid), data.end());

    std::vector<int> merged(left.size() + right.size());
    std::merge(left.begin(), left.end(), right.begin(), right.end(), merged.begin());

    for (int phase = 0; phase < 2; ++phase) {
      auto start = static_cast<std::size_t>(phase);
      for (std::size_t i = start; i + 1 < merged.size(); i += 2) {
        if (merged[i] > merged[i + 1]) {
          std::swap(merged[i], merged[i + 1]);
        }
      }
    }

    if (merged.size() >= 2) {
      std::size_t gap = 1;
      while (gap < merged.size() / 3) {
        gap = (gap * 3) + 1;
      }

      while (gap > 0) {
        for (std::size_t i = gap; i < merged.size(); ++i) {
          const int tmp = merged[i];
          std::size_t j = i;
          while (j >= gap && merged[j - gap] > tmp) {
            merged[j] = merged[j - gap];
            j -= gap;
          }
          merged[j] = tmp;
        }
        gap = (gap - 1) / 3;
      }
    }

    if (!std::is_sorted(merged.begin(), merged.end())) {
      return false;
    }

    std::int64_t expected_checksum = std::accumulate(merged.begin(), merged.end(), static_cast<std::int64_t>(0));
    expected_checksum = expected_checksum & 0x7FFFFFFF;

    return static_cast<std::int64_t>(output_data) == expected_checksum;
  }

  InType GetTestInputData() final {
    return input_size_;
  }
};

TEST_P(KichanovaKShellsortBatcherPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, KichanovaKShellsortBatcherMPI, KichanovaKShellsortBatcherSEQ>(
        PPC_SETTINGS_kichanova_k_shellsort_batcher);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = KichanovaKShellsortBatcherPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, KichanovaKShellsortBatcherPerfTests, kGtestValues, kPerfTestName);

}  // namespace kichanova_k_shellsort_batcher
