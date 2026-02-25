#include <gtest/gtest.h>

#include <random>
#include <string>
#include <vector>

#include "fedoseev_gaussian_method_horizontal_strip_scheme/common/include/common.hpp"
#include "fedoseev_gaussian_method_horizontal_strip_scheme/mpi/include/ops_mpi.hpp"
#include "fedoseev_gaussian_method_horizontal_strip_scheme/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace fedoseev_gaussian_method_horizontal_strip_scheme {

InType generatePerformanceTestSystem(int n);

class FedoseevRunPerfTestProcesses2 : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kSmallSize = 100;
  const int kMediumSize = 500;
  const int kLargeSize = 1000;
  const int kExtraLargeSize = 2000;

  InType input_data_{};

  void SetUp() override {
    auto test_info = ::testing::UnitTest::GetInstance()->current_test_info();
    std::string test_name = test_info->name();

    int size = kMediumSize;
    if (test_name.find("Small") != std::string::npos)
      size = kSmallSize;
    else if (test_name.find("Medium") != std::string::npos)
      size = kMediumSize;
    else if (test_name.find("Large") != std::string::npos)
      size = kLargeSize;
    else if (test_name.find("ExtraLarge") != std::string::npos)
      size = kExtraLargeSize;

    input_data_ = generatePerformanceTestSystem(size);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const auto &A = input_data_;
    return !output_data.empty() && output_data.size() == A.size();
  }

  InType GetTestInputData() final { return input_data_; }
};

InType generatePerformanceTestSystem(int n) {
  static std::mt19937 gen(42);
  std::uniform_real_distribution<> dis(1.0, 100.0);

  InType A(n, std::vector<double>(n + 1));

  for (int i = 0; i < n; ++i) {
    double sum = 0.0;
    for (int j = 0; j < n; ++j) {
      A[i][j] = dis(gen);
      sum += std::abs(A[i][j]);
    }
    A[i][i] = sum + 1.0;
    A[i][n] = dis(gen);
  }
  return A;
}

TEST_P(FedoseevRunPerfTestProcesses2, SmallSystem) { ExecuteTest(GetParam()); }
TEST_P(FedoseevRunPerfTestProcesses2, MediumSystem) { ExecuteTest(GetParam()); }
TEST_P(FedoseevRunPerfTestProcesses2, LargeSystem) { ExecuteTest(GetParam()); }
TEST_P(FedoseevRunPerfTestProcesses2, ExtraLargeSystem) { ExecuteTest(GetParam()); }

namespace {

using MPITask = FedoseevGaussianMethodHorizontalStripSchemeMPI;
using SEQTask = FedoseevGaussianMethodHorizontalStripSchemeSEQ;

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, MPITask, SEQTask>(PPC_SETTINGS_example_processes_2);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);
const auto kPerfTestName = FedoseevRunPerfTestProcesses2::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, FedoseevRunPerfTestProcesses2, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace fedoseev_gaussian_method_horizontal_strip_scheme