#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "cheremkhin_a_gaus_vert/common/include/common.hpp"
#include "cheremkhin_a_gaus_vert/mpi/include/ops_mpi.hpp"
#include "cheremkhin_a_gaus_vert/seq/include/ops_seq.hpp"
#include "performance/include/performance.hpp"
#include "util/include/perf_test_util.hpp"

namespace cheremkhin_a_gaus_vert {

namespace {

Input MakeDiagonallyDominantSystem(int n) {
  Input in;
  in.n = n;
  in.a.resize(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0);
  in.b.resize(static_cast<std::size_t>(n), 0.0);

  std::vector<double> x_true(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    x_true[static_cast<std::size_t>(i)] = static_cast<double>(i + 1);
  }

  for (int row = 0; row < n; ++row) {
    double row_sum = 0.0;
    for (int col = 0; col < n; ++col) {
      double v = (row == col) ? 0.0 : (static_cast<double>((row + 1) + (col + 2)) / 100.0);
      in.a[(static_cast<std::size_t>(row) * static_cast<std::size_t>(n)) + static_cast<std::size_t>(col)] = v;
      row_sum += std::abs(v);
    }
    const double diag = row_sum + 1.0;
    in.a[(static_cast<std::size_t>(row) * static_cast<std::size_t>(n)) + static_cast<std::size_t>(row)] = diag;
  }

  for (int row = 0; row < n; ++row) {
    double s = 0.0;
    for (int col = 0; col < n; ++col) {
      s += in.a[(static_cast<std::size_t>(row) * static_cast<std::size_t>(n)) + static_cast<std::size_t>(col)] *
           x_true[static_cast<std::size_t>(col)];
    }
    in.b[static_cast<std::size_t>(row)] = s;
  }

  return in;
}

double MaxResidual(const Input &in, const std::vector<double> &x) {
  const int n = in.n;
  double max_r = 0.0;
  for (int row = 0; row < n; ++row) {
    double s = 0.0;
    for (int col = 0; col < n; ++col) {
      s += in.a[(static_cast<std::size_t>(row) * static_cast<std::size_t>(n)) + static_cast<std::size_t>(col)] *
           x[static_cast<std::size_t>(col)];
    }
    const double rr = std::abs(s - in.b[static_cast<std::size_t>(row)]);
    max_r = std::max(max_r, rr);
  }
  return max_r;
}

}  // namespace

class GausVertRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  static constexpr int kN = 1200;
  static constexpr double kMaxResidual = 1e-5;

  InType input_data_{};

  void SetUp() override {
    input_data_ = MakeDiagonallyDominantSystem(kN);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (std::cmp_not_equal(output_data.size(), input_data_.n)) {
      return false;
    }
    const double residual = MaxResidual(input_data_, output_data);
    if (residual >= kMaxResidual) {
      ADD_FAILURE() << "Residual too large: " << residual << ", expected < " << kMaxResidual;
      return false;
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

  void SetPerfAttributes(ppc::performance::PerfAttr &perf_attrs) override {
    const auto t0 = std::chrono::high_resolution_clock::now();
    perf_attrs.current_timer = [t0] {
      auto now = std::chrono::high_resolution_clock::now();
      auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now - t0).count();
      return static_cast<double>(ns) * 1e-9;
    };
    perf_attrs.num_running = 1;
  }
};

TEST_P(GausVertRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, CheremkhinAGausVertMPI, CheremkhinAGausVertSEQ>(
    PPC_SETTINGS_cheremkhin_a_gaus_vert);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = GausVertRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, GausVertRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace cheremkhin_a_gaus_vert
