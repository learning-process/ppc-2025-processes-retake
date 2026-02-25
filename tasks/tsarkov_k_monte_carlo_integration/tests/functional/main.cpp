#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "tsarkov_k_monte_carlo_integration/common/include/common.hpp"
#include "tsarkov_k_monte_carlo_integration/mpi/include/ops_mpi.hpp"
#include "tsarkov_k_monte_carlo_integration/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace tsarkov_k_monte_carlo_integration {

class TsarkovKMonteCarloFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    const TestType test_tuple = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(test_tuple);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // Для Monte-Carlo мы проверяем корректность по диапазону значений.
    // f(x)=exp(-||x||^2) на [0,1]^d => значение интеграла строго (0,1].
    return std::isfinite(output_data) && (output_data > 0.0) && (output_data <= 1.0);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

TEST_P(TsarkovKMonteCarloFuncTests, MonteCarloIntegral) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {
    std::make_tuple(InType{2, 1000, 42}, "d2_n1000"),
    std::make_tuple(InType{3, 2000, 7}, "d3_n2000"),
    std::make_tuple(InType{5, 5000, 123}, "d5_n5000"),
};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<TsarkovKMonteCarloIntegrationMPI, InType>(
                                               kTestParam, PPC_SETTINGS_tsarkov_k_monte_carlo_integration),
                                           ppc::util::AddFuncTask<TsarkovKMonteCarloIntegrationSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_tsarkov_k_monte_carlo_integration));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kTestName = TsarkovKMonteCarloFuncTests::PrintFuncTestName<TsarkovKMonteCarloFuncTests>;

INSTANTIATE_TEST_SUITE_P(MonteCarloTests, TsarkovKMonteCarloFuncTests, kGtestValues, kTestName);

}  // namespace

}  // namespace tsarkov_k_monte_carlo_integration
