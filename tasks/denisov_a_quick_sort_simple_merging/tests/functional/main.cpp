#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "denisov_a_quick_sort_simple_merging/common/include/common.hpp"
#include "denisov_a_quick_sort_simple_merging/mpi/include/ops_mpi.hpp"
#include "denisov_a_quick_sort_simple_merging/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace denisov_a_quick_sort_simple_merging {

class DenisovAQuickSortMergeFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &param) {
    return std::to_string(std::get<0>(param)) + "_" + std::get<1>(param);
  }

 protected:
  void SetUp() override {
    const auto &test_info = std::get<static_cast<size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    const int test_number = std::get<0>(test_info);

    if (test_number == 1) {
      input_data_.clear();
    } else if (test_number == 2) {
      input_data_ = {42};
    } else if (test_number == 3) {
      input_data_ = {2, 5, 22, 23, 25, 26, 37, 41, 43};
    } else if (test_number == 4) {
      input_data_ = {97, 89, 83, 79, 73, 71, 67, 61, 59, 53, 47};
    } else if (test_number == 5) {
      input_data_ = {17, 29, 17, 11, 29, 7, 17, 19, 29, 11, 23};
    } else if (test_number == 6) {
      input_data_ = {-17, 31, -13, 0, -19, 11, 23, -7, 31, 5};
    } else if (test_number == 7) {
      input_data_.resize(127);
      for (size_t i = 0; i < 127; i++) {
        input_data_[i] = static_cast<int>(126 - i);
      }
    } else if (test_number == 8) {
      input_data_ = {91, 17, 83, 29, 5, 67, 41, 13, 97, 3, 59, 31, 7, 73, 19};
    } else if (test_number == 9) {
      input_data_ = {37, 11};
    } else if (test_number == 10) {
      input_data_ = {47, 23, 71, 3, 59, 13, 89, 31, 7, 61, 19, 43, 79, 11, 53};
    } else if (test_number == 11) {
      input_data_ = {64, 32, 16, 8, 4, 2, 1, 128, 256, 512};
    } else {
      input_data_ = {83, 17, 59, 7, 41, 97, 23, 71, 13, 89};
    }
  }

  bool CheckTestOutputData(OutType &out) final {
    if (input_data_.empty()) {
      return out.empty();
    }

    if (out.size() != input_data_.size()) {
      return false;
    }

    std::vector<int> sorted_ref = input_data_;
    std::ranges::sort(sorted_ref);

    return out == sorted_ref && std::ranges::is_sorted(out);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

TEST_P(DenisovAQuickSortMergeFuncTests, QuickSortMergeTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 12> kTestParam = {
    std::make_tuple(1, "empty"),    std::make_tuple(2, "single"),     std::make_tuple(3, "sorted"),
    std::make_tuple(4, "reversed"), std::make_tuple(5, "duplicates"), std::make_tuple(6, "negative"),
    std::make_tuple(7, "large"),    std::make_tuple(8, "mixed"),      std::make_tuple(9, "small"),
    std::make_tuple(10, "medium"),  std::make_tuple(11, "powers"),    std::make_tuple(12, "basic")};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<DenisovAQuickSortMergeMPI, InType>(
                                               kTestParam, PPC_SETTINGS_denisov_a_quick_sort_simple_merging),
                                           ppc::util::AddFuncTask<DenisovAQuickSortMergeSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_denisov_a_quick_sort_simple_merging));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = DenisovAQuickSortMergeFuncTests::PrintFuncTestName<DenisovAQuickSortMergeFuncTests>;

INSTANTIATE_TEST_SUITE_P(QuickSortMergeTests, DenisovAQuickSortMergeFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace denisov_a_quick_sort_simple_merging
