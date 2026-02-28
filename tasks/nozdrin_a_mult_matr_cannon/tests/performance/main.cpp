#include <gtest/gtest.h>
#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <tuple>
#include <vector>

#include "nozdrin_a_mult_matr_cannon/common/include/common.hpp"
#include "nozdrin_a_mult_matr_cannon/mpi/include/ops_mpi.hpp"
#include "nozdrin_a_mult_matr_cannon/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace nozdrin_a_mult_matr_cannon {

class NozdrinAMultMatrCannonPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  void SetUp() override {
    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    size_t rc = 1008;

    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized == 1) {
      int q = static_cast<int>(std::sqrt(world_size));
      if (q * q == world_size) {
        rc = (rc / q) * q;
      }
    }

    size_t size = rc * rc;

    std::vector<double> a(rc * rc);
    std::vector<double> b(rc * rc);

    for (size_t i = 0; i < size; i++) {
      a[i] = static_cast<double>(i % 100);
      b[i] = static_cast<double>((i + 1) % 100);
    }

    input_data_ = std::make_tuple(rc, a, b);

    std::vector<double> c(size, 0.0);
    LocalMatrixMultiply(a, b, c, static_cast<int>(rc));
    expected_output_ = c;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (expected_output_.size() != output_data.size()) {
      return false;
    }
    const double epsilon = 1e-7;
    for (size_t i = 0; i < expected_output_.size(); ++i) {
      if (std::abs(expected_output_[i] - output_data[i]) > epsilon) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

  static void LocalMatrixMultiply(const std::vector<double> &a, const std::vector<double> &b, std::vector<double> &c,
                                  int n) {
    for (int i = 0; i < n; ++i) {
      for (int k = 0; k < n; ++k) {
        double temp = a[(i * n) + k];
        for (int j = 0; j < n; ++j) {
          c[(i * n) + j] += temp * b[(k * n) + j];
        }
      }
    }
  }

 private:
  InType input_data_;
  std::vector<double> expected_output_;
};

TEST_P(NozdrinAMultMatrCannonPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, NozdrinAMultMatrCannonMPI, NozdrinAMultMatrCannonSEQ>(
    PPC_SETTINGS_nozdrin_a_mult_matr_cannon);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = NozdrinAMultMatrCannonPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, NozdrinAMultMatrCannonPerfTests, kGtestValues, kPerfTestName);

}  // namespace nozdrin_a_mult_matr_cannon
