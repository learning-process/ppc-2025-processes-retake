#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "kichanova_k_shellsort_batcher/common/include/common.hpp"
#include "kichanova_k_shellsort_batcher/mpi/include/ops_mpi.hpp"
#include "kichanova_k_shellsort_batcher/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace kichanova_k_shellsort_batcher {

class KichanovaKShellsortBatcherFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_size_ = static_cast<InType>(std::get<0>(params));
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

 private:
  InType input_size_ = 0;
};

namespace {

TEST_P(KichanovaKShellsortBatcherFuncTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 4> kTestParam = {std::make_tuple(100, "small"), std::make_tuple(1000, "medium"),
                                            std::make_tuple(5000, "large"), std::make_tuple(10000, "xlarge")};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<KichanovaKShellsortBatcherMPI, InType>(
                                               kTestParam, PPC_SETTINGS_kichanova_k_shellsort_batcher),
                                           ppc::util::AddFuncTask<KichanovaKShellsortBatcherSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_kichanova_k_shellsort_batcher));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = KichanovaKShellsortBatcherFuncTests::PrintFuncTestName<KichanovaKShellsortBatcherFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, KichanovaKShellsortBatcherFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace kichanova_k_shellsort_batcher
