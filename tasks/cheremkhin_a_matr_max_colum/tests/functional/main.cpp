#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <string>
#include <tuple>
#include <vector>

#include "cheremkhin_a_matr_max_colum/common/include/common.hpp"
#include "cheremkhin_a_matr_max_colum/mpi/include/ops_mpi.hpp"
#include "cheremkhin_a_matr_max_colum/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace cheremkhin_a_matr_max_colum {

class CheremkhinARunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    std::string test_name =
        std::to_string(std::get<0>(test_param)[0][0]) + "_" + std::to_string(std::get<1>(test_param)[0]);
    std::ranges::replace_if(test_name, [](char c) {
      const auto uc = static_cast<unsigned char>(c);
      return !std::isalnum(uc) && c != '_';
    }, '_');
    return test_name;
  }

 protected:
  void SetUp() override {
    TestType params = std::get<2>(GetParam());
    input_data_ = std::get<0>(params);
    correct_answer_ = std::get<1>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return correct_answer_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  std::vector<int> correct_answer_;
  InType input_data_;
};

namespace {

TEST_P(CheremkhinARunFuncTestsProcesses, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {
    std::make_tuple(std::vector<std::vector<int>>{{1, 2, 3}, {1, 1, 0}, {0, -1, 5}}, std::vector<int>{1, 2, 5}),
    std::make_tuple(std::vector<std::vector<int>>{{-1, -5}, {-3, 1}}, std::vector<int>{-1, 1}),
    std::make_tuple(std::vector<std::vector<int>>{{5}}, std::vector<int>{5})};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<CheremkhinAMatrMaxColumMPI, InType>(kTestParam, PPC_SETTINGS_cheremkhin_a_matr_max_colum),
    ppc::util::AddFuncTask<CheremkhinAMatrMaxColumSEQ, InType>(kTestParam, PPC_SETTINGS_cheremkhin_a_matr_max_colum));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = CheremkhinARunFuncTestsProcesses::PrintFuncTestName<CheremkhinARunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(MaxColumTest, CheremkhinARunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace cheremkhin_a_matr_max_colum
