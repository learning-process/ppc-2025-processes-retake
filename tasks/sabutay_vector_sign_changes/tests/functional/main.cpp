#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>

#include "sabutay_vector_sign_changes/common/include/common.hpp"
#include "sabutay_vector_sign_changes/mpi/include/ops_mpi.hpp"
#include "sabutay_vector_sign_changes/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace sabutay_vector_sign_changes {

namespace {

inline int Sign(double x) {
  if (x > 0.0) {
    return 1;
  }
  if (x < 0.0) {
    return -1;
  }
  return 0;
}

inline double GetElement(std::size_t i) {
  if (i % 5 == 0) {
    return 0.0;
  }
  if (i % 2 == 0) {
    return 1.0;
  }
  return -1.0;
}

inline int CountAlternations(std::size_t n) {
  int prev = 0;
  int cnt = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const int s = Sign(GetElement(i));
    if (s == 0) {
      continue;
    }
    if (prev != 0 && s != prev) {
      ++cnt;
    }
    prev = s;
  }
  return cnt;
}

}  // namespace

class SabutayKRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  InType input_data{};
  OutType expected_output{};

  void SetUp() override {
    const TestType params = std::get<TestType>(GetParam());
    const int n_int = std::get<0>(params);
    input_data = static_cast<InType>(n_int);
    expected_output = CountAlternations(static_cast<std::size_t>(n_int));
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int rank = 0;
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (initialized != 0) {
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }

    if (rank != 0) {
      return true;
    }
    return expected_output == output_data;
  }

  InType GetTestInputData() final {
    return input_data;
  }
};

TEST_P(SabutayKRunFuncTestsProcesses, AlternationsCorrect) {
  ExecuteTest(GetParam());
}

namespace {

const std::array<TestType, 5> kTestParam = {
    std::make_tuple(0, "n0"),   std::make_tuple(1, "n1"),   std::make_tuple(5, "n5"),
    std::make_tuple(10, "n10"), std::make_tuple(25, "n25"),
};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<SabutayVectorSignChangesMPI, InType>(kTestParam, PPC_SETTINGS_example_processes),
    ppc::util::AddFuncTask<SabutayVectorSignChangesSEQ, InType>(kTestParam, PPC_SETTINGS_example_processes));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kTestName = SabutayKRunFuncTestsProcesses::PrintFuncTestName<SabutayKRunFuncTestsProcesses>;

using ParamType = SabutayKRunFuncTestsProcesses::ParamType;

::testing::internal::ParamGenerator<ParamType> SabutayFuncEvalGenerate() {
  return kGtestValues;
}

std::string SabutayFuncEvalGenerateName(const ::testing::TestParamInfo<ParamType> &info) {
  return kTestName(info);
}

const int kRastvorovFuncDummy =
    ::testing::UnitTest::GetInstance()
        ->parameterized_test_registry()
        .GetTestSuitePatternHolder<SabutayKRunFuncTestsProcesses>("SabutayKRunFuncTestsProcesses",
                                                                  ::testing::internal::CodeLocation(__FILE__, __LINE__))
        ->AddTestSuiteInstantiation("AlternationsFuncTests", &SabutayFuncEvalGenerate, &SabutayFuncEvalGenerateName,
                                    __FILE__, __LINE__);

}  // namespace

}  // namespace sabutay_vector_sign_changes
