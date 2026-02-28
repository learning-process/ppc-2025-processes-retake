#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <string>
#include <tuple>

#include "nazyrov_a_global_opt_2d/common/include/common.hpp"
#include "nazyrov_a_global_opt_2d/mpi/include/ops_mpi.hpp"
#include "nazyrov_a_global_opt_2d/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace nazyrov_a_global_opt_2d {

class NazyrovAGlobalOpt2dFuncTest : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int test_case = std::get<0>(params);

    input_data_ = OptInput();
    input_data_.epsilon = 0.01;
    input_data_.r_param = 2.5;
    input_data_.max_iterations = 500;

    switch (test_case) {
      case 0:
        // f(x,y) = x^2 + y^2, min at (0,0)
        input_data_.func = [](double x, double y) { return (x * x) + (y * y); };
        input_data_.x_min = -2.0;
        input_data_.x_max = 2.0;
        input_data_.y_min = -2.0;
        input_data_.y_max = 2.0;
        expected_.x_opt = 0.0;
        expected_.y_opt = 0.0;
        expected_.func_min = 0.0;
        break;
      case 1:
        // f(x,y) = (x-1)^2 + (y-1)^2, min at (1,1)
        input_data_.func = [](double x, double y) { return ((x - 1.0) * (x - 1.0)) + ((y - 1.0) * (y - 1.0)); };
        input_data_.x_min = -1.0;
        input_data_.x_max = 3.0;
        input_data_.y_min = -1.0;
        input_data_.y_max = 3.0;
        expected_.x_opt = 1.0;
        expected_.y_opt = 1.0;
        expected_.func_min = 0.0;
        break;
      case 2:
        // f(x,y) = (x+0.5)^2 + (y-0.5)^2
        input_data_.func = [](double x, double y) { return ((x + 0.5) * (x + 0.5)) + ((y - 0.5) * (y - 0.5)); };
        input_data_.x_min = -2.0;
        input_data_.x_max = 2.0;
        input_data_.y_min = -2.0;
        input_data_.y_max = 2.0;
        expected_.x_opt = -0.5;
        expected_.y_opt = 0.5;
        expected_.func_min = 0.0;
        break;
      case 3:
        // f(x,y) = x^2 + 4*y^2, min at (0,0)
        input_data_.func = [](double x, double y) { return (x * x) + (4.0 * y * y); };
        input_data_.x_min = -3.0;
        input_data_.x_max = 3.0;
        input_data_.y_min = -3.0;
        input_data_.y_max = 3.0;
        expected_.x_opt = 0.0;
        expected_.y_opt = 0.0;
        expected_.func_min = 0.0;
        break;
      case 4:
        // f(x,y) = 2*x^2 + y^2 + x*y - 7*x - 5*y
        // min at (9/7, 13/7), f_min = -448/49 ≈ -9.1429
        input_data_.func = [](double x, double y) { return (2.0 * x * x) + (y * y) + (x * y) - (7.0 * x) - (5.0 * y); };
        input_data_.x_min = -1.0;
        input_data_.x_max = 4.0;
        input_data_.y_min = -1.0;
        input_data_.y_max = 4.0;
        input_data_.max_iterations = 2000;
        expected_.x_opt = 9.0 / 7.0;
        expected_.y_opt = 13.0 / 7.0;
        expected_.func_min = -448.0 / 49.0;
        break;
      case 5:
        // f(x,y) = sin(x) + cos(y), min at (−pi/2, pi)
        input_data_.func = [](double x, double y) { return std::sin(x) + std::cos(y); };
        input_data_.x_min = -4.0;
        input_data_.x_max = 4.0;
        input_data_.y_min = 0.0;
        input_data_.y_max = 4.0;
        input_data_.max_iterations = 1000;
        expected_.func_min = -2.0;
        expected_.x_opt = -std::numbers::pi / 2.0;
        expected_.y_opt = std::numbers::pi;
        break;
      default:
        input_data_.func = [](double x, double y) { return (x * x) + (y * y); };
        input_data_.x_min = -1.0;
        input_data_.x_max = 1.0;
        input_data_.y_min = -1.0;
        input_data_.y_max = 1.0;
        expected_.x_opt = 0.0;
        expected_.y_opt = 0.0;
        expected_.func_min = 0.0;
        break;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (ppc::util::IsUnderMpirun()) {
      int rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      if (rank != 0) {
        return true;
      }
    }
    constexpr double kTol = 0.1;
    return std::abs(output_data.func_min - expected_.func_min) < kTol;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_;
};

namespace {

TEST_P(NazyrovAGlobalOpt2dFuncTest, GlobalOpt2d) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 6> kTestParam = {
    std::make_tuple(0, "sphere"),   std::make_tuple(1, "shifted_sphere"), std::make_tuple(2, "offset_sphere"),
    std::make_tuple(3, "elliptic"), std::make_tuple(4, "quadratic"),      std::make_tuple(5, "sincos"),
};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<GlobalOpt2dMPI, InType>(kTestParam, PPC_SETTINGS_nazyrov_a_global_opt_2d),
                   ppc::util::AddFuncTask<GlobalOpt2dSEQ, InType>(kTestParam, PPC_SETTINGS_nazyrov_a_global_opt_2d));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kFuncTestName = NazyrovAGlobalOpt2dFuncTest::PrintFuncTestName<NazyrovAGlobalOpt2dFuncTest>;

INSTANTIATE_TEST_SUITE_P(GlobalOpt2dTests, NazyrovAGlobalOpt2dFuncTest, kGtestValues, kFuncTestName);

}  // namespace

}  // namespace nazyrov_a_global_opt_2d
