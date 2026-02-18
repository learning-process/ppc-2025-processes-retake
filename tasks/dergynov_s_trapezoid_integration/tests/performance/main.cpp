#include <gtest/gtest.h>

#include "dergynov_s_trapezoid_integration/mpi/include/ops_mpi.hpp"
#include "dergynov_s_trapezoid_integration/seq/include/ops_seq.hpp"

using dergynov_s_trapezoid_integration::DergynovSTrapezoidIntegrationMPI;
using dergynov_s_trapezoid_integration::DergynovSTrapezoidIntegrationSEQ;
using dergynov_s_trapezoid_integration::InType;

TEST(dergynov_s_trapezoid_integration_perf, seq_performance) {
  InType in{0.0, 100.0, 10'000'000, 0};
  DergynovSTrapezoidIntegrationSEQ task(in);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
}

TEST(dergynov_s_trapezoid_integration_perf, mpi_performance) {
  InType in{0.0, 100.0, 10'000'000, 0};
  DergynovSTrapezoidIntegrationMPI task(in);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
}
// namespace dergynov_s_trapezoid_integration
