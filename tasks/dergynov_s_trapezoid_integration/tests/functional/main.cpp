#include <gtest/gtest.h>

#include "dergynov_s_trapezoid_integration/mpi/include/ops_mpi.hpp"
#include "dergynov_s_trapezoid_integration/seq/include/ops_seq.hpp"

using dergynov_s_trapezoid_integration::DergynovSTrapezoidIntegrationMPI;
using dergynov_s_trapezoid_integration::DergynovSTrapezoidIntegrationSEQ;
using dergynov_s_trapezoid_integration::GetExactIntegral;
using dergynov_s_trapezoid_integration::InType;

TEST(dergynov_s_trapezoid_integration, linear_function_seq) {
  InType in{0.0, 10.0, 1000, 0};  // f(x)=x
  DergynovSTrapezoidIntegrationSEQ task(in);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  double res = task.GetOutput();
  double exact = GetExactIntegral(in);
  ASSERT_NEAR(res, exact, 1e-3);
}

TEST(dergynov_s_trapezoid_integration, quadratic_function_seq) {
  InType in{0.0, 5.0, 2000, 1};  // f(x)=x^2
  DergynovSTrapezoidIntegrationSEQ task(in);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  double res = task.GetOutput();
  double exact = GetExactIntegral(in);
  ASSERT_NEAR(res, exact, 1e-3);
}

TEST(dergynov_s_trapezoid_integration, sin_function_seq) {
  InType in{0.0, 3.1415926535, 2000, 2};  // f(x)=sin(x)
  DergynovSTrapezoidIntegrationSEQ task(in);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  double res = task.GetOutput();
  double exact = GetExactIntegral(in);
  ASSERT_NEAR(res, exact, 1e-3);
}

TEST(dergynov_s_trapezoid_integration, linear_function_mpi) {
  InType in{0.0, 10.0, 1000, 0};
  DergynovSTrapezoidIntegrationMPI task(in);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  double res = task.GetOutput();
  double exact = GetExactIntegral(in);
  ASSERT_NEAR(res, exact, 1e-3);
}

TEST(dergynov_s_trapezoid_integration, quadratic_function_mpi) {
  InType in{1.0, 4.0, 1500, 1};
  DergynovSTrapezoidIntegrationMPI task(in);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  double res = task.GetOutput();
  double exact = GetExactIntegral(in);
  ASSERT_NEAR(res, exact, 1e-3);
}

TEST(dergynov_s_trapezoid_integration, sin_function_mpi) {
  InType in{0.0, 1.5707963267, 1500, 2};
  DergynovSTrapezoidIntegrationMPI task(in);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  double res = task.GetOutput();
  double exact = GetExactIntegral(in);
  ASSERT_NEAR(res, exact, 1e-3);
}
// namespace dergynov_s_trapezoid_integration
