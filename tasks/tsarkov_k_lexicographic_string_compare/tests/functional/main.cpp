#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <utility>

#include "tsarkov_k_lexicographic_string_compare/common/include/common.hpp"
#include "tsarkov_k_lexicographic_string_compare/mpi/include/ops_mpi.hpp"
#include "tsarkov_k_lexicographic_string_compare/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace tsarkov_k_lexicographic_string_compare {

class TsarkovKRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<0>(test_param) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    const TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = {std::get<0>(params), std::get<1>(params)};
    expected_ = (input_data_.first <= input_data_.second) ? 1 : 0;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_{0};
};

namespace {

TEST_P(TsarkovKRunFuncTestsProcesses, LexicographicOrderCheck) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 6> kTestParam = {
    std::make_tuple("", ""),       std::make_tuple("", "a"),      std::make_tuple("a", ""),
    std::make_tuple("abc", "abc"), std::make_tuple("abc", "abd"), std::make_tuple("b", "aa"),
};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<TsarkovKLexicographicStringCompareMPI, InType>(
                                               kTestParam, PPC_SETTINGS_tsarkov_k_lexicographic_string_compare),
                                           ppc::util::AddFuncTask<TsarkovKLexicographicStringCompareSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_tsarkov_k_lexicographic_string_compare));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kFuncTestName = TsarkovKRunFuncTestsProcesses::PrintFuncTestName<TsarkovKRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(StringCompareTests, TsarkovKRunFuncTestsProcesses, kGtestValues, kFuncTestName);

}  // namespace

}  // namespace tsarkov_k_lexicographic_string_compare
