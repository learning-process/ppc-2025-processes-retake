#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <string>
#include <tuple>
#include <vector>

#include "cheremkhin_a_radix_sort_batcher/common/include/common.hpp"
#include "cheremkhin_a_radix_sort_batcher/mpi/include/ops_mpi.hpp"
#include "cheremkhin_a_radix_sort_batcher/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace cheremkhin_a_radix_sort_batcher {

namespace {

std::vector<int> SortedCopy(std::vector<int> v) {
  std::ranges::sort(v);
  return v;
}

}  // namespace

class CheremkhinARadixSortBatcherRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    const auto &input = std::get<0>(test_param);
    const int first = input.empty() ? 0 : input[0];
    std::string name = std::to_string(input.size()) + "_" + std::to_string(first);
    std::ranges::replace_if(name, [](char c) {
      const auto uc = static_cast<unsigned char>(c);
      return !std::isalnum(uc) && c != '_';
    }, '_');
    return name;
  }

 protected:
  void SetUp() override {
    const TestType params = std::get<2>(GetParam());
    input_data_ = std::get<0>(params);
    correct_answer_ = std::get<1>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == correct_answer_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  OutType correct_answer_;
  InType input_data_;
};

namespace {

TEST_P(CheremkhinARadixSortBatcherRunFuncTestsProcesses, SortIntVector) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 5> kTestParam = {
    std::make_tuple(InType{5, 4, 3, 2, 1}, SortedCopy(InType{5, 4, 3, 2, 1})),
    std::make_tuple(InType{1, 2, 3, 4, 5}, SortedCopy(InType{1, 2, 3, 4, 5})),
    std::make_tuple(InType{0, -1, -5, 3, 3, 2, -1}, SortedCopy(InType{0, -1, -5, 3, 3, 2, -1})),
    std::make_tuple(InType{42}, SortedCopy(InType{42})),
    std::make_tuple(InType{1000000, -1000000, 7, 0, -7, 7}, SortedCopy(InType{1000000, -1000000, 7, 0, -7, 7}))};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<CheremkhinARadixSortBatcherMPI, InType>(
                                               kTestParam, PPC_SETTINGS_cheremkhin_a_radix_sort_batcher),
                                           ppc::util::AddFuncTask<CheremkhinARadixSortBatcherSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_cheremkhin_a_radix_sort_batcher));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = CheremkhinARadixSortBatcherRunFuncTestsProcesses::PrintFuncTestName<
    CheremkhinARadixSortBatcherRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(RadixSortBatcherTest, CheremkhinARadixSortBatcherRunFuncTestsProcesses, kGtestValues,
                         kPerfTestName);

}  // namespace

}  // namespace cheremkhin_a_radix_sort_batcher
