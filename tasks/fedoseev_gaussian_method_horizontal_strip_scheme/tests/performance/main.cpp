#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <random>
#include <string>
#include <vector>

#include "fedoseev_gaussian_method_horizontal_strip_scheme/common/include/common.hpp"
#include "fedoseev_gaussian_method_horizontal_strip_scheme/mpi/include/ops_mpi.hpp"
#include "fedoseev_gaussian_method_horizontal_strip_scheme/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace fedoseev_gaussian_method_horizontal_strip_scheme {
namespace {

InType GeneratePerformanceTestSystem(int n) {
  static std::mt19937 gen(std::random_device{}());
  std::uniform_real_distribution<> dis(1.0, 100.0);

  std::vector<std::vector<double>> augmented_matrix(static_cast<size_t>(n),
                                                    std::vector<double>(static_cast<size_t>(n) + 1));

  for (int i = 0; i < n; ++i) {
    double sum = 0.0;
    for (int j = 0; j < n; ++j) {
      augmented_matrix[static_cast<size_t>(i)][static_cast<size_t>(j)] = dis(gen);
      sum += std::abs(augmented_matrix[static_cast<size_t>(i)][static_cast<size_t>(j)]);
    }
    augmented_matrix[static_cast<size_t>(i)][static_cast<size_t>(i)] = sum + 1.0;
    augmented_matrix[static_cast<size_t>(i)][static_cast<size_t>(n)] = dis(gen);
  }

  return augmented_matrix;
}

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
    const auto &augmented_matrix = input_data_;
    return !output_data.empty() && output_data.size() == augmented_matrix.size();
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

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

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, FedoseevTestTaskMPI, FedoseevTestTaskSEQ>(PPC_SETTINGS_example_processes_2);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = FedoseevRunPerfTestProcesses2::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, FedoseevRunPerfTestProcesses2, kGtestValues, kPerfTestName);

}  // namespace
}  // namespace fedoseev_gaussian_method_horizontal_strip_scheme
