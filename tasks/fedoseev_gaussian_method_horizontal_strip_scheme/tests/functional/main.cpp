#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "fedoseev_gaussian_method_horizontal_strip_scheme/common/include/common.hpp"
#include "fedoseev_gaussian_method_horizontal_strip_scheme/mpi/include/ops_mpi.hpp"
#include "fedoseev_gaussian_method_horizontal_strip_scheme/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace fedoseev_gaussian_method_horizontal_strip_scheme {

InType GenerateTestSystem(int n, int seed);

class FedoseevRunFuncTestsProcesses2
    : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" +
           std::to_string(std::get<1>(test_param));
  }

 protected:
  void SetUp() override {
    TestType params =
        std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(
            GetParam());
    int matrix_size = std::get<0>(params);
    int seed = std::get<1>(params);

    input_data_ = GenerateTestSystem(matrix_size, seed);

    reference_solution_.resize(matrix_size);
    for (int i = 0; i < matrix_size; ++i) {
      reference_solution_[i] = static_cast<double>(i + 1);
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != reference_solution_.size()) return false;

    double tolerance = 1e-6;
    for (size_t i = 0; i < output_data.size(); ++i) {
      if (std::abs(output_data[i] - reference_solution_[i]) > tolerance)
        return false;
    }

    const auto &a = input_data_;
    int n = static_cast<int>(a.size());
    for (int i = 0; i < n; ++i) {
      double sum = 0.0;
      for (int j = 0; j < n; ++j) {
        sum += a[i][j] * output_data[j];
      }
      if (std::abs(sum - a[i][n]) > tolerance * n) return false;
    }
    return true;
  }

  InType GetTestInputData() final { return input_data_; }

 private:
  InType input_data_;
  OutType reference_solution_;
};

InType GenerateTestSystem(int n, int seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<> dis(0.1, 1.0);

  InType a(n, std::vector<double>(n + 1));

  for (int i = 0; i < n; ++i) {
    double sum = 0.0;
    for (int j = 0; j < n; ++j) {
      if (i == j) {
        a[i][j] = 10.0 * n + (i + 1);
      } else {
        a[i][j] = dis(gen);
      }
      if (i != j) sum += std::abs(a[i][j]);
    }
    if (std::abs(a[i][i]) <= sum) {
      a[i][i] = sum + 1.0;
    }

    double b = 0.0;
    for (int j = 0; j < n; ++j) {
      b += a[i][j] * (j + 1);
    }
    a[i][n] = b;
  }
  return a;
}

namespace {

TEST_P(FedoseevRunFuncTestsProcesses2, GaussianEliminationTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 4> kTestParam = {std::make_tuple(10, 1),
                                             std::make_tuple(20, 2),
                                             std::make_tuple(30, 3),
                                             std::make_tuple(40, 4)};

using MPITask = FedoseevGaussianMethodHorizontalStripSchemeMPI;
using SEQTask = FedoseevGaussianMethodHorizontalStripSchemeSEQ;

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<MPITask, InType>(
                       kTestParam, PPC_SETTINGS_example_processes_2),
                   ppc::util::AddFuncTask<SEQTask, InType>(
                       kTestParam, PPC_SETTINGS_example_processes_2));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);
const auto kPerfTestName =
    FedoseevRunFuncTestsProcesses2::PrintFuncTestName<FedoseevRunFuncTestsProcesses2>;

INSTANTIATE_TEST_SUITE_P(GaussianTests, FedoseevRunFuncTestsProcesses2,
                         kGtestValues, kPerfTestName);

}  // namespace

}  // namespace fedoseev_gaussian_method_horizontal_strip_scheme