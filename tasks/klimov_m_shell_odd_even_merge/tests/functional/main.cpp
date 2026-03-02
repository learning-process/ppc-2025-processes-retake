#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <fstream>
#include <string>
#include <tuple>

#include "klimov_m_shell_odd_even_merge/common/include/common.hpp"
#include "klimov_m_shell_odd_even_merge/mpi/include/ops_mpi.hpp"
#include "klimov_m_shell_odd_even_merge/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace klimov_m_shell_odd_even_merge {

class ShellBatcherFuncTest : public ppc::util::BaseRunFuncTests<InputType, OutputType, TestParam> {
 public:
  static std::string PrintTestParam(const TestParam &test_param) {
    size_t dot_pos = test_param.find('.');
    return test_param.substr(0, dot_pos);
  }

 protected:
  void SetUp() override {
    TestParam file_name = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    std::string full_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_klimov_m_shell_odd_even_merge, file_name);
    std::ifstream fin(full_path);

    int val = 0;
    while (fin >> val) {
      input_data_.push_back(val);
    }
    fin.close();
  }

  bool CheckTestOutputData(OutputType &out_data) final {
    return std::ranges::is_sorted(out_data);
  }

  InputType GetTestInputData() final {
    return input_data_;
  }

 private:
  InputType input_data_;
};

namespace {

TEST_P(ShellBatcherFuncTest, TestFromFiles) {
  ExecuteTest(GetParam());
}

const std::array<TestParam, 3> kTestFiles = {"test1.txt", "test2.txt", "test3.txt"};

const auto kTaskList = std::tuple_cat(
    ppc::util::AddFuncTask<ShellBatcherMPI, InputType>(kTestFiles, PPC_SETTINGS_klimov_m_shell_odd_even_merge),
    ppc::util::AddFuncTask<ShellBatcherSEQ, InputType>(kTestFiles, PPC_SETTINGS_klimov_m_shell_odd_even_merge));

const auto kTestValues = ppc::util::ExpandToValues(kTaskList);
const auto kNamePrinter = ShellBatcherFuncTest::PrintFuncTestName<ShellBatcherFuncTest>;

INSTANTIATE_TEST_SUITE_P(ShellBatcherFunctionalTests, ShellBatcherFuncTest, kTestValues, kNamePrinter);

}  // namespace

}  // namespace klimov_m_shell_odd_even_merge
