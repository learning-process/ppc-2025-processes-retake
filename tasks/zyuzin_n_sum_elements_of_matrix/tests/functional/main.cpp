#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"
#include "zyuzin_n_sum_elements_of_matrix/common/include/common.hpp"
#include "zyuzin_n_sum_elements_of_matrix/mpi/include/ops_mpi.hpp"
#include "zyuzin_n_sum_elements_of_matrix/seq/include/ops_seq.hpp"

namespace zyuzin_n_sum_elements_of_matrix {

class ZyuzinNSumElementsOfMatrixFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return test_param;
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    const std::string input_path =
        ppc::util::GetAbsoluteTaskPath(PPC_ID_zyuzin_n_sum_elements_of_matrix, "inputs/" + params + ".txt");
    const std::string output_path =
        ppc::util::GetAbsoluteTaskPath(PPC_ID_zyuzin_n_sum_elements_of_matrix, "outputs/" + params + ".txt");

    std::ifstream ifs(input_path);
    int rows = 0;
    int cols = 0;
    ifs >> rows >> cols;
    std::vector<double> data(static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols), 0.0);
    double val = 0.0;
    int i = 0;
    while (ifs >> val && i < rows * cols) {
      data[i++] = val;
    }
    matrix_ = std::make_tuple(rows, cols, std::move(data));

    std::ifstream ofs(output_path);
    double outv = 0.0;
    ofs >> outv;
    expected_sum_ = outv;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const double k_eps = 1e-9;
    return std::fabs(expected_sum_ - output_data) <= k_eps;
  }

  InType GetTestInputData() final {
    return matrix_;
  }

 private:
  InType matrix_;
  OutType expected_sum_{0.0};
};

namespace {

TEST_P(ZyuzinNSumElementsOfMatrixFuncTests, MatrixSumTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 10> kTestParam = {"test1", "test2", "test3", "test4", "test5",
                                             "test6", "test7", "test8", "test9", "test10"};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<ZyuzinNSumElementsOfMatrixMPI, InType>(
                                               kTestParam, PPC_SETTINGS_zyuzin_n_sum_elements_of_matrix),
                                           ppc::util::AddFuncTask<ZyuzinNSumElementsOfMatrixSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_zyuzin_n_sum_elements_of_matrix));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = ZyuzinNSumElementsOfMatrixFuncTests::PrintFuncTestName<ZyuzinNSumElementsOfMatrixFuncTests>;

INSTANTIATE_TEST_SUITE_P(ZyuzinNMatrixSumTests, ZyuzinNSumElementsOfMatrixFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace zyuzin_n_sum_elements_of_matrix
