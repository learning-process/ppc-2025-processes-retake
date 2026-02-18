#include <gtest/gtest.h>

#include "dergynov_s_trapezoid_integration/common/include/common.hpp"
#include "dergynov_s_trapezoid_integration/mpi/include/ops_mpi.hpp"
#include "dergynov_s_trapezoid_integration/seq/include/ops_seq.hpp"

using dergynov_s_trapezoid_integration::DergynovSTrapezoidIntegrationMPI;
using dergynov_s_trapezoid_integration::DergynovSTrapezoidIntegrationSEQ;
using dergynov_s_trapezoid_integration::InType;

class DergynovTrapezoidIntegrationPerfTest : public ::testing::Test {
 protected:
  template <typename T>
  void RunTestSequence(T &task) {
    ASSERT_TRUE(task.Validation());
    ASSERT_TRUE(task.PreProcessing());
    ASSERT_TRUE(task.Run());
    ASSERT_TRUE(task.PostProcessing());
  }
};

TEST_F(DergynovTrapezoidIntegrationPerfTest, SeqPerformance) {
  InType in{0.0, 100.0, 10'000'000, 0};
  DergynovSTrapezoidIntegrationSEQ task(in);
  RunTestSequence(task);
}

TEST_F(DergynovTrapezoidIntegrationPerfTest, MpiPerformance) {
  InType in{0.0, 100.0, 10'000'000, 0};
  DergynovSTrapezoidIntegrationMPI task(in);
  RunTestSequence(task);
}
// namespace dergynov_s_trapezoid_integration
