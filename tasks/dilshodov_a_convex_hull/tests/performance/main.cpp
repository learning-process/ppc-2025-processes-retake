#include <gtest/gtest.h>
#include <mpi.h>

#include <cstddef>

#include "dilshodov_a_convex_hull/common/include/common.hpp"
#include "dilshodov_a_convex_hull/mpi/include/ops_mpi.hpp"
#include "dilshodov_a_convex_hull/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace dilshodov_a_convex_hull {

class DilshodovAConvexHullPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
  static constexpr int kWidth = 1000;
  static constexpr int kHeight = 1000;
  InType input_data_;
  OutType expected_;

  void SetUp() override {
    input_data_.clear();
    input_data_.push_back(kWidth);
    input_data_.push_back(kHeight);
    input_data_.resize((static_cast<std::size_t>(kWidth) * kHeight) + 2, 0);

    for (int py = 0; py < kHeight; ++py) {
      for (int px = 0; px < kWidth; ++px) {
        input_data_[(static_cast<std::size_t>(py) * kWidth) + static_cast<std::size_t>(px) + 2] = 255;
      }
    }

    expected_ = {{0, 0}, {kWidth - 1, 0}, {kWidth - 1, kHeight - 1}, {0, kHeight - 1}};
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (ppc::util::IsUnderMpirun()) {
      int rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      if (rank != 0) {
        return true;
      }
    }
    if (output_data.size() < 3) {
      return false;
    }

    auto cross = [](const Point &o, const Point &a, const Point &b) {
      return ((a.x - o.x) * (b.y - o.y)) - ((a.y - o.y) * (b.x - o.x));
    };

    const auto n = output_data.size();
    for (std::size_t i = 0; i < n; ++i) {
      const Point &a = output_data[i];
      const Point &b = output_data[(i + 1) % n];
      const Point &c = output_data[(i + 2) % n];
      if (cross(a, b, c) < 0) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(DilshodovAConvexHullPerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, ConvexHullMPI, ConvexHullSEQ>(PPC_SETTINGS_dilshodov_a_convex_hull);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = DilshodovAConvexHullPerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, DilshodovAConvexHullPerfTest, kGtestValues, kPerfTestName);

}  // namespace dilshodov_a_convex_hull
