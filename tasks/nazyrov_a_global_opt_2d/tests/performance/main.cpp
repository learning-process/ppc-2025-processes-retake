#include <gtest/gtest.h>
#include <mpi.h>

#include <cmath>

#include "nazyrov_a_global_opt_2d/common/include/common.hpp"
#include "nazyrov_a_global_opt_2d/mpi/include/ops_mpi.hpp"
#include "nazyrov_a_global_opt_2d/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace nazyrov_a_global_opt_2d {

class NazyrovAGlobalOpt2dPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;

  void SetUp() override {
    input_data_ = OptInput();
    input_data_.func = [](double x, double y) {
      return std::sin(x * x) + std::cos(y * y) + (0.1 * x * x) + (0.1 * y * y);
    };
    input_data_.x_min = -5.0;
    input_data_.x_max = 5.0;
    input_data_.y_min = -5.0;
    input_data_.y_max = 5.0;
    input_data_.epsilon = 0.001;
    input_data_.r_param = 2.5;
    input_data_.max_iterations = 2000;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (ppc::util::IsUnderMpirun()) {
      int rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      if (rank != 0) {
        return true;
      }
    }
    return output_data.func_min < 1.0;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(NazyrovAGlobalOpt2dPerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, GlobalOpt2dMPI, GlobalOpt2dSEQ>(PPC_SETTINGS_nazyrov_a_global_opt_2d);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = NazyrovAGlobalOpt2dPerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, NazyrovAGlobalOpt2dPerfTest, kGtestValues, kPerfTestName);

}  // namespace nazyrov_a_global_opt_2d
