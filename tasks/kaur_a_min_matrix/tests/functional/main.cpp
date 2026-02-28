#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdlib>
#include <ctime>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "kaur_a_min_matrix/common/include/common.hpp"
#include "kaur_a_min_matrix/mpi/include/ops_mpi.hpp"
#include "kaur_a_min_matrix/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace kaur_a_min_matrix {

class KaurAMinMatrixFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    int min_val = std::get<2>(test_param);
    std::string str_min = (min_val < 0) ? "minus" + std::to_string(-min_val) : std::to_string(min_val);
    return std::to_string(std::get<0>(test_param)) + "x" + std::to_string(std::get<1>(test_param)) + "_min_" + str_min +
           "_" + std::get<3>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    int rows = std::get<0>(params);
    int columns = std::get<1>(params);
    int min_val = std::get<2>(params);

    std::vector<int> matrix(static_cast<size_t>(rows) * static_cast<size_t>(columns));

    std::uniform_int_distribution<int> dist(min_val, 250);

    for (int &elem : matrix) {
      elem = dist(gen_);
    }

    matrix[(rows * columns) / 2] = min_val;

    input_data_ = std::make_tuple(static_cast<size_t>(rows), static_cast<size_t>(columns), matrix);
    expected_min_ = min_val;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (expected_min_ == output_data);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_min_ = 0;
  std::mt19937 gen_{std::random_device{}()};
};

namespace {

TEST_P(KaurAMinMatrixFuncTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 6> kTestParam = {std::make_tuple(3, 3, -15, "Small_matrix"),
                                            std::make_tuple(5, 5, 32, "Medium_matrix"),
                                            std::make_tuple(10, 10, -41, "Large_matrix"),
                                            std::make_tuple(10, 5, 4, "Different_size_matrix_rows"),
                                            std::make_tuple(6, 11, -1, "Different_size_matrix_columns"),
                                            std::make_tuple(1, 1, -10, "Only_1_elem")};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<KaurAMinMatrixMPI, InType>(kTestParam, PPC_SETTINGS_kaur_a_min_matrix),
                   ppc::util::AddFuncTask<KaurAMinMatrixSEQ, InType>(kTestParam, PPC_SETTINGS_kaur_a_min_matrix));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = KaurAMinMatrixFuncTests::PrintFuncTestName<KaurAMinMatrixFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, KaurAMinMatrixFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace kaur_a_min_matrix
