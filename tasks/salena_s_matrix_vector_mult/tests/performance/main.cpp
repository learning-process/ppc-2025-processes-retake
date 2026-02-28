#include <gtest/gtest.h>
#include <algorithm>
#include <random>
#include <vector>

#include "salena_s_matrix_vector_mult/common/include/common.hpp"
#include "salena_s_matrix_vector_mult/mpi/include/ops_mpi.hpp"
#include "salena_s_matrix_vector_mult/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace salena_s_matrix_vector_mult {

class MatVecMultPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_{};

  void SetUp() override {
    int rows = 500;
    int cols = 500;
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
    return output_data.size() == static_cast<size_t>(input_data_.rows);
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(MatVecMultPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, TestTaskMPI, TestTaskSEQ>(PPC_SETTINGS_salena_s_matrix_vector_mult);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);
const auto kPerfTestName = MatVecMultPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, MatVecMultPerfTests, kGtestValues, kPerfTestName);

}  // namespace salena_s_matrix_vector_mult