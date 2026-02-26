#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "fedoseev_gaussian_method_horizontal_strip_scheme/common/include/common.hpp"
#include "fedoseev_gaussian_method_horizontal_strip_scheme/mpi/include/ops_mpi.hpp"
#include "fedoseev_gaussian_method_horizontal_strip_scheme/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace fedoseev_gaussian_method_horizontal_strip_scheme {

InType GeneratePerformanceTestSystem(int n);

class FedoseevRunPerfTestProcesses2 : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kSmallSize_ = 100;
  const int kMediumSize_ = 500;
  const int kLargeSize_ = 1000;
  const int kExtraLargeSize_ = 2000;

  InType input_data_;

  void SetUp() override {
    const auto *test_info = ::testing::UnitTest::GetInstance()->current_test_info();
    std::string test_name = test_info->name();

    int size = kMediumSize_;
    if (test_name.find("Small") != std::string::npos) {
      size = kSmallSize_;
    } else if (test_name.find("Medium") != std::string::npos) {
      size = kMediumSize_;
    } else if (test_name.find("Large") != std::string::npos) {
      size = kLargeSize_;
    } else if (test_name.find("ExtraLarge") != std::string::npos) {
      size = kExtraLargeSize_;
    }

    input_data_ = GeneratePerformanceTestSystem(size);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const auto &a = input_data_;
    return !output_data.empty() && output_data.size() == a.size();
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

InType GeneratePerformanceTestSystem(int n) {
  static std::mt19937 gen(42);
  std::uniform_real_distribution<> dis(1.0, 100.0);

  InType a(n, std::vector<double>(n + 1));

  for (int i = 0; i < n; ++i) {
    double sum = 0.0;
    for (int j = 0; j < n; ++j) {
      a[i][j] = dis(gen);
      sum += std::abs(a[i][j]);
    }
    a[i][i] = sum + 1.0;
    a[i][n] = dis(gen);
  }
  return a;
}

TEST_P(FedoseevRunPerfTestProcesses2, SmallSystem) {
  ExecuteTest(GetParam());
}
TEST_P(FedoseevRunPerfTestProcesses2, MediumSystem) {
  ExecuteTest(GetParam());
}
TEST_P(FedoseevRunPerfTestProcesses2, LargeSystem) {
  ExecuteTest(GetParam());
}
TEST_P(FedoseevRunPerfTestProcesses2, ExtraLargeSystem) {
  ExecuteTest(GetParam());
}

namespace {

using MPITask = FedoseevGaussianMethodHorizontalStripSchemeMPI;
using SEQTask = FedoseevGaussianMethodHorizontalStripSchemeSEQ;

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, MPITask, SEQTask>(PPC_SETTINGS_example_processes_2);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);
const auto kPerfTestName = FedoseevRunPerfTestProcesses2::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, FedoseevRunPerfTestProcesses2, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace fedoseev_gaussian_method_horizontal_strip_scheme
