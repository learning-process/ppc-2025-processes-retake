#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

#include "solonin_v_col_min_matrix/common/include/common.hpp"
#include "solonin_v_col_min_matrix/mpi/include/ops_mpi.hpp"
#include "solonin_v_col_min_matrix/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

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

class SoloninVMinMatrixFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);
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

 private:
  InType input_data_ = 0;
  std::vector<InType> expected_mins_;
};

namespace {

TEST_P(SoloninVMinMatrixFuncTests, ComputesColumnMinimumsForDiverseSizes) { ExecuteTest(GetParam()); }

const std::array<TestType, 11> kFunctionalParams = {
    std::make_tuple(1, "tuple_unit"),   std::make_tuple(2, "tuple_even"),   std::make_tuple(3, "tuple_odd"),
    std::make_tuple(5, "tuple_5"),      std::make_tuple(17, "tuple_prime"), std::make_tuple(64, "tuple_64"),
    std::make_tuple(99, "tuple_99"),    std::make_tuple(100, "tuple_100"),  std::make_tuple(128, "tuple_128"),
    std::make_tuple(256, "tuple_256"), std::make_tuple(512, "tuple_512")};

const auto kTaskMatrix =
    std::tuple_cat(ppc::util::AddFuncTask<SoloninVMinMatrixMPI, InType>(kFunctionalParams,
                                                                        PPC_SETTINGS_solonin_v_col_min_matrix),
                   ppc::util::AddFuncTask<SoloninVMinMatrixSEQ, InType>(kFunctionalParams,
                                                                        PPC_SETTINGS_solonin_v_col_min_matrix));

const auto kParameterizedValues = ppc::util::ExpandToValues(kTaskMatrix);
const auto kFunctionalTestName = SoloninVMinMatrixFuncTests::PrintFuncTestName<SoloninVMinMatrixFuncTests>;

INSTANTIATE_TEST_SUITE_P(MinimumColumnSearchSuite, SoloninVMinMatrixFuncTests, kParameterizedValues,
                         kFunctionalTestName);

TEST(SoloninVMinMatrixValidation, RejectsZeroInputSeq) {
  SoloninVMinMatrixSEQ task(0);
  EXPECT_FALSE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
}

TEST(SoloninVMinMatrixValidation, RejectsZeroInputMpi) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }
  SoloninVMinMatrixMPI task(0);
  EXPECT_FALSE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
}

TEST(SoloninVMinMatrixValidation, AcceptsPositiveInputSeq) {
  SoloninVMinMatrixSEQ task(10);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());
}

TEST(SoloninVMinMatrixValidation, AcceptsPositiveInputMpi) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }
  SoloninVMinMatrixMPI task(10);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());
}

}  // namespace

}  // namespace solonin_v_col_min_matrix
