#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "fedoseev_multi_step_scheme_parallelization_by_characteristics/common/include/common.hpp"
#include "fedoseev_multi_step_scheme_parallelization_by_characteristics/mpi/include/ops_mpi.hpp"
#include "fedoseev_multi_step_scheme_parallelization_by_characteristics/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace fedoseev_multi_step_scheme_parallelization_by_characteristics {

class FedoseevRunFuncTestsProcesses3 : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (output_data >= 0.0) && (output_data <= 1.0);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_ = 0;
};

namespace {

TEST_P(FedoseevRunFuncTestsProcesses3, MultiStepSchemeTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {std::make_tuple(3, "3"), std::make_tuple(5, "5"), std::make_tuple(7, "7")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<FedoseevMultiStepSchemeMPI, InType>(kTestParam, PPC_SETTINGS_example_processes_3),
    ppc::util::AddFuncTask<FedoseevMultiStepSchemeSEQ, InType>(kTestParam, PPC_SETTINGS_example_processes_3));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = FedoseevRunFuncTestsProcesses3::PrintFuncTestName<FedoseevRunFuncTestsProcesses3>;

INSTANTIATE_TEST_SUITE_P(MultiStepSchemeTests, FedoseevRunFuncTestsProcesses3, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace fedoseev_multi_step_scheme_parallelization_by_characteristics
