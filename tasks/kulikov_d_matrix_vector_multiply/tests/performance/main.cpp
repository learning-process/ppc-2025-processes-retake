#include <gtest/gtest.h>
#include <mpi.h>

#include <cstddef>
#include <random>
#include <string>
#include <vector>

#include "kulikov_d_matrix_vector_multiply/common/include/common.hpp"
#include "kulikov_d_matrix_vector_multiply/mpi/include/ops_mpi.hpp"
#include "kulikov_d_matrix_vector_multiply/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace kulikov_d_matrix_vector_multiply {

class KulikovMatrixMultiplyRunPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  InType input_data;
  std::vector<int> expected_data;

  void SetUp() override {
    const auto &param = GetParam();
    const std::string &test_name = std::get<1>(param);

    const bool is_task_run = test_name.find("task_run") != std::string::npos;

    int rows = 0;
    int cols = 0;

    if (is_task_run) {
      rows = 300;
      cols = 300;
    } else {
      rows = 1000;
      cols = 1000;
    }

    input_data.rows = rows;
    input_data.cols = cols;
    input_data.matrix.resize(static_cast<size_t>(rows) * static_cast<size_t>(cols));
    input_data.vector.resize(static_cast<size_t>(cols));
    expected_data.resize(static_cast<size_t>(rows));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(-5, 5);

    for (int j = 0; j < cols; ++j) {
      input_data.vector[j] = dist(gen);
    }

    for (int i = 0; i < rows; ++i) {
      int sum = 0;
      for (int j = 0; j < cols; ++j) {
        const int val = dist(gen);
        input_data.matrix[(i * cols) + j] = val;
        sum += val * input_data.vector[j];
      }
      expected_data[i] = sum;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank != 0) {
      return true;
    }

    if (output_data.size() != expected_data.size()) {
      return false;
    }

    for (size_t i = 0; i < output_data.size(); ++i) {
      if (output_data[i] != expected_data[i]) {
        return false;
      }
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_data;
  }
};

TEST_P(KulikovMatrixMultiplyRunPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, KulikovDMatrixMultiplyMPI, KulikovDMatrixMultiplySEQ>(
    PPC_SETTINGS_kulikov_d_matrix_vector_multiply);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

INSTANTIATE_TEST_SUITE_P(RunModeTests, KulikovMatrixMultiplyRunPerfTests, kGtestValues,
                         KulikovMatrixMultiplyRunPerfTests::CustomPerfTestName);

}  // namespace kulikov_d_matrix_vector_multiply
