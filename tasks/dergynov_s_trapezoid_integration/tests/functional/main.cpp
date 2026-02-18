#include <gtest/gtest.h>

#include <numbers>

#include "dergynov_s_trapezoid_integration/common/include/common.hpp"
#include "dergynov_s_trapezoid_integration/mpi/include/ops_mpi.hpp"
#include "dergynov_s_trapezoid_integration/seq/include/ops_seq.hpp"

using dergynov_s_trapezoid_integration::DergynovSTrapezoidIntegrationMPI;
using dergynov_s_trapezoid_integration::DergynovSTrapezoidIntegrationSEQ;
using dergynov_s_trapezoid_integration::GetExactIntegral;
using dergynov_s_trapezoid_integration::InType;

class TrapezoidIntegrationTest : public ::testing::Test {
 protected:
  template <typename T>
  void RunAndCheck(T &task, const InType &in) {
    ASSERT_TRUE(task.Validation());
    ASSERT_TRUE(task.PreProcessing());
    ASSERT_TRUE(task.Run());
    ASSERT_TRUE(task.PostProcessing());

    double res = task.GetOutput();
    double exact = GetExactIntegral(in);
    ASSERT_NEAR(res, exact, 1e-3);
  }
};

TEST_F(TrapezoidIntegrationTest, LinearFunctionSeq) {
  InType in{0.0, 10.0, 1000, 0};
  DergynovSTrapezoidIntegrationSEQ task(in);
  RunAndCheck(task, in);
}

TEST_F(TrapezoidIntegrationTest, QuadraticFunctionSeq) {
  InType in{0.0, 5.0, 2000, 1};
  DergynovSTrapezoidIntegrationSEQ task(in);
  RunAndCheck(task, in);
}

TEST_F(TrapezoidIntegrationTest, SinFunctionSeq) {
  InType in{0.0, std::numbers::pi, 2000, 2};
  DergynovSTrapezoidIntegrationSEQ task(in);
  RunAndCheck(task, in);
}

TEST_F(TrapezoidIntegrationTest, LinearFunctionMpi) {
  InType in{0.0, 10.0, 1000, 0};
  DergynovSTrapezoidIntegrationMPI task(in);
  RunAndCheck(task, in);
}

TEST_F(TrapezoidIntegrationTest, QuadraticFunctionMpi) {
  InType in{1.0, 4.0, 1500, 1};
  DergynovSTrapezoidIntegrationMPI task(in);
  RunAndCheck(task, in);
}

TEST_F(TrapezoidIntegrationTest, SinFunctionMpi) {
  InType in{0.0, std::numbers::pi / 2, 1500, 2};
  DergynovSTrapezoidIntegrationMPI task(in);
  RunAndCheck(task, in);
}
// namespace dergynov_s_trapezoid_integration
