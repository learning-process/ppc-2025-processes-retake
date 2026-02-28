#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numbers>
#include <string>
#include <tuple>
#include <vector>

#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"
#include "yusupkina_m_mnog_integ_monte_carlo/common/include/common.hpp"
#include "yusupkina_m_mnog_integ_monte_carlo/mpi/include/ops_mpi.hpp"
#include "yusupkina_m_mnog_integ_monte_carlo/seq/include/ops_seq.hpp"

namespace yusupkina_m_mnog_integ_monte_carlo {

struct TestCase {
  std::string name;
  double x_min;
  double x_max;
  double y_min;
  double y_max;
  std::function<double(double, double)> f;
  double expected;
  int64_t num_points;
};

class YusupkinaMMnogIntegMonteCarloFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    int case_idx = std::get<0>(params);
    static const std::vector<TestCase> kTestCases = {
        {"const", 0.0, 1.0, 0.0, 1.0, [](double, double) { return 1.0; }, 1.0, 100000},

        {"linear", 0.0, 1.0, 0.0, 1.0, [](double x, double y) { return x + y; }, 1.0, 100000},

        {"product", 0.0, 1.0, 0.0, 1.0, [](double x, double y) { return x * y; }, 0.25, 100000},

        {"trig", 0.0, std::numbers::pi / 2.0, 0.0, std::numbers::pi / 2.0,
         [](double x, double y) { return std::sin(x) * std::cos(y); }, 1.0, 100000},

        {"sin_sum", 0.0, std::numbers::pi / 2.0, 0.0, std::numbers::pi / 2.0,
         [](double x, double y) { return std::sin(x + y); }, 2.0, 100000},

        {"smallN", 0.0, 1.0, 0.0, 1.0, [](double x, double y) { return x + y; }, 1.0, 1000},

        {"largeN", 0.0, 1.0, 0.0, 1.0, [](double x, double y) { return x + y; }, 1.0, 1000000},

        {"square", 0.0, 1.0, 0.0, 1.0, [](double x, double y) { return (x * x) + (y * y); }, 2.0 / 3.0, 100000},

        {"shifted", 2.0, 5.0, -1.0, 3.0, [](double, double) { return 1.0; }, 12.0, 100000},

        {"negative", -2.0, 2.0, -2.0, 2.0, [](double x, double y) { return x + y; }, 0.0, 500000},

        {"zeroArea", 0.0, 0.0, 0.0, 1.0, [](double, double) { return 1.0; }, 0.0, 100000}};

    const auto &tc = kTestCases[case_idx];

    input_data_ = InputData(tc.x_min, tc.x_max, tc.y_min, tc.y_max, tc.f, tc.num_points);
    exp_output_ = tc.expected;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (std::abs(exp_output_) < 1e-10) {
      return std::abs(output_data) < 0.15;
    }
    double rel_tolerance = 5.0 / std::pow(input_data_.num_points, 0.4);
    double rel_error = std::abs(output_data - exp_output_) / std::abs(exp_output_);
    return rel_error <= rel_tolerance;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType exp_output_ = 0.0;
};

namespace {

TEST_P(YusupkinaMMnogIntegMonteCarloFuncTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 11> kTestParam = {
    std::make_tuple(0, "const"),    std::make_tuple(1, "linear"),   std::make_tuple(2, "product"),
    std::make_tuple(3, "trig"),     std::make_tuple(4, "sin_sum"),  std::make_tuple(5, "smallN"),
    std::make_tuple(6, "largeN"),   std::make_tuple(7, "square"),   std::make_tuple(8, "shifted"),
    std::make_tuple(9, "negative"), std::make_tuple(10, "zeroArea")};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<YusupkinaMMnogIntegMonteCarloMPI, InType>(
                                               kTestParam, PPC_SETTINGS_yusupkina_m_mnog_integ_monte_carlo),
                                           ppc::util::AddFuncTask<YusupkinaMMnogIntegMonteCarloSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_yusupkina_m_mnog_integ_monte_carlo));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    YusupkinaMMnogIntegMonteCarloFuncTests::PrintFuncTestName<YusupkinaMMnogIntegMonteCarloFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, YusupkinaMMnogIntegMonteCarloFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace yusupkina_m_mnog_integ_monte_carlo
