#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>

#include "salykina_a_count_letters_in_string/common/include/common.hpp"
#include "salykina_a_count_letters_in_string/mpi/include/ops_mpi.hpp"
#include "salykina_a_count_letters_in_string/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace salykina_a_count_letters_in_string {

class SalykinaARunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return "lenght_" + std::to_string(std::get<1>(test_param));
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    std::string base_string = std::get<0>(params);
    expected_count_ = std::get<1>(params);
    input_data_ = base_string;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (expected_count_ == output_data);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_count_ = 0;
};

namespace {

TEST_P(SalykinaARunFuncTestsProcesses, CountLetters) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {std::make_tuple("Hello", 5), std::make_tuple("Hello 1 2 3 4 5 World", 10),
                                            std::make_tuple("The sun rises every morning in the east.", 32)};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<SalykinaACountLettersMPI, InType>(
                                               kTestParam, PPC_SETTINGS_salykina_a_count_letters_in_string),
                                           ppc::util::AddFuncTask<SalykinaACountLettersSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_salykina_a_count_letters_in_string));
const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);
const auto kPerfTestName = SalykinaARunFuncTestsProcesses::PrintFuncTestName<SalykinaARunFuncTestsProcesses>;
INSTANTIATE_TEST_SUITE_P(StringCountTests, SalykinaARunFuncTestsProcesses, kGtestValues, kPerfTestName);
}  // namespace

}  // namespace salykina_a_count_letters_in_string
