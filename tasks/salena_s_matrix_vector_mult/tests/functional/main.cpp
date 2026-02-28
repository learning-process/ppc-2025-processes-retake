#include <gtest/gtest.h>
#include <algorithm>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "salena_s_matrix_vector_mult/common/include/common.hpp"
#include "salena_s_matrix_vector_mult/mpi/include/ops_mpi.hpp"
#include "salena_s_matrix_vector_mult/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace salena_s_matrix_vector_mult {

using TestType = std::tuple<int, int, std::string>;

class MatVecMultFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<2>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int rows = std::get<0>(params);
    int cols = std::get<1>(params);

    input_data_.rows = rows;
    input_data_.cols = cols;
    input_data_.matrix.resize(rows * cols);
    input_data_.vec.resize(cols);

    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(-10.0, 10.0);
    for (int i = 0; i < rows * cols; ++i) {
      input_data_.matrix[i] = dist(gen);
    }
    for (int i = 0; i < cols; ++i) {
      input_data_.vec[i] = dist(gen);
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    std::vector<double> expected_res(input_data_.rows, 0.0);
    for (int i = 0; i < input_data_.rows; ++i) {
      for (int j = 0; j < input_data_.cols; ++j) {
        expected_res[i] += input_data_.matrix[i * input_data_.cols + j] * input_data_.vec[j];
      }
    }

    if (output_data.size() != expected_res.size()) return false;
    for (size_t i = 0; i < expected_res.size(); ++i) {
      // Сравниваем с точностью до 1e-4, так как это double
      if (std::abs(expected_res[i] - output_data[i]) > 1e-4) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

TEST_P(MatVecMultFuncTests, MultMatVec) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {
    std::make_tuple(10, 10, "10x10"),
    std::make_tuple(50, 50, "50x50"),
    std::make_tuple(100, 55, "100x55")
};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<TestTaskMPI, InType>(kTestParam, PPC_SETTINGS_salena_s_matrix_vector_mult),
                   ppc::util::AddFuncTask<TestTaskSEQ, InType>(kTestParam, PPC_SETTINGS_salena_s_matrix_vector_mult));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);
const auto kPerfTestName = MatVecMultFuncTests::PrintFuncTestName<MatVecMultFuncTests>;
INSTANTIATE_TEST_SUITE_P(MatVecMultTests, MatVecMultFuncTests, kGtestValues, kPerfTestName);

}  // namespace salena_s_matrix_vector_mult