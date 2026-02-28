#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "solonin_v_col_min_matrix/common/include/common.hpp"
#include "solonin_v_col_min_matrix/mpi/include/ops_mpi.hpp"
#include "solonin_v_col_min_matrix/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace solonin_v_col_min_matrix {

namespace {

inline InType Generate(int64_t i, int64_t j) {
  uint64_t seed = (i * 100000007ULL + j * 1000000009ULL) ^ 42ULL;
  seed ^= seed >> 12;
  seed ^= seed << 25;
  seed ^= seed >> 27;
  uint64_t value = seed * 0x2545F4914F6CDD1DULL;
  return static_cast<InType>((value % 2000001ULL) - 1000000);
}

inline std::vector<InType> CalculateExpectedColumnMins(InType n) {
  std::vector<InType> expected_mins(static_cast<size_t>(n), std::numeric_limits<InType>::max());
  for (InType i = 0; i < n; i++) {
    for (InType j = 0; j < n; j++) {
      InType value = Generate(static_cast<int64_t>(i), static_cast<int64_t>(j));
      expected_mins[static_cast<size_t>(j)] = std::min(value, expected_mins[static_cast<size_t>(j)]);
    }
  }
  return expected_mins;
}

}  // namespace

class SoloninVMinMatrixPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const InType kTestSize_ = 10000;
  InType input_data_{};
  std::vector<InType> expected_mins_;

  void SetUp() override {
    input_data_ = kTestSize_;
    expected_mins_ = CalculateExpectedColumnMins(input_data_);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != static_cast<size_t>(input_data_)) {
      return false;
    }
    for (std::size_t j = 0; j < output_data.size(); j++) {
      if (output_data[j] != expected_mins_[j]) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final { return input_data_; }
};

TEST_P(SoloninVMinMatrixPerfTests, RunPerfModes) { ExecuteTest(GetParam()); }

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, SoloninVMinMatrixMPI, SoloninVMinMatrixSEQ>(
    PPC_SETTINGS_solonin_v_col_min_matrix);
const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);
const auto kPerfTestName = SoloninVMinMatrixPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, SoloninVMinMatrixPerfTests, kGtestValues, kPerfTestName);

}  // namespace solonin_v_col_min_matrix
