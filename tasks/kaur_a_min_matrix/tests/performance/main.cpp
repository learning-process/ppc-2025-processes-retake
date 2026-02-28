#include <gtest/gtest.h>

#include <cstddef>
#include <tuple>
#include <vector>

#include "kaur_a_min_matrix/common/include/common.hpp"
#include "kaur_a_min_matrix/mpi/include/ops_mpi.hpp"
#include "kaur_a_min_matrix/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace kaur_a_min_matrix {

class KaurAMinMatrixPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  void SetUp() override {
    size_t rows = 5000;
    size_t columns = 5000;
    std::vector<int> matrix(rows * columns);
    for (size_t i = 0; i < rows; i++) {
      for (size_t j = 0; j < columns; j++) {
        int index = static_cast<int>((i * columns) + j);
        matrix[index] = static_cast<int>((i * i) + j);
      }
    }
    matrix[(rows * columns) / 2] = expected_min_;
    input_data_ = std::make_tuple(rows, columns, matrix);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_min_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  int expected_min_ = -1000000;
};

TEST_P(KaurAMinMatrixPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

namespace {
const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, KaurAMinMatrixMPI, KaurAMinMatrixSEQ>(PPC_SETTINGS_kaur_a_min_matrix);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = KaurAMinMatrixPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, KaurAMinMatrixPerfTests, kGtestValues, kPerfTestName);
}  // namespace

}  // namespace kaur_a_min_matrix
