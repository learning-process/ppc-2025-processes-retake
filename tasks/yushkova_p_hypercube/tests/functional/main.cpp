#include <gtest/gtest.h>
#include <mpi.h>

#include <cstdint>

#include "util/include/util.hpp"
#include "yushkova_p_hypercube/common/include/common.hpp"
#include "yushkova_p_hypercube/mpi/include/ops_mpi.hpp"
#include "yushkova_p_hypercube/seq/include/ops_seq.hpp"

namespace yushkova_p_hypercube {
namespace {

bool IsPowerOfTwo(int value) {
  return value > 0 && (value & (value - 1)) == 0;
}

OutType ReferenceEdges(InType n) {
  return static_cast<OutType>(n) * (static_cast<std::uint64_t>(1) << (n - 1));
}

template <class Task>
bool RunTask(Task &task) {
  return task.Validation() && task.PreProcessing() && task.Run() && task.PostProcessing();
}

}  // namespace

class YushkovaPHypercubeFunctional : public ::testing::TestWithParam<InType> {};

TEST_P(YushkovaPHypercubeFunctional, SeqReturnsCorrectEdgeCount) {
  const InType n = GetParam();
  YushkovaPHypercubeSEQ task(n);
  ASSERT_TRUE(RunTask(task));
  EXPECT_EQ(task.GetOutput(), ReferenceEdges(n));
}

TEST_P(YushkovaPHypercubeFunctional, MpiMatchesSeqAndReference) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }

  const InType n = GetParam();

  int world_size = 0;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (!IsPowerOfTwo(world_size)) {
    GTEST_SKIP();
  }

  YushkovaPHypercubeMPI mpi_task(n);
  ASSERT_TRUE(RunTask(mpi_task));

  if (rank == 0) {
    YushkovaPHypercubeSEQ seq_task(n);
    ASSERT_TRUE(RunTask(seq_task));
    EXPECT_EQ(mpi_task.GetOutput(), seq_task.GetOutput());
    EXPECT_EQ(mpi_task.GetOutput(), ReferenceEdges(n));
  }
}

INSTANTIATE_TEST_SUITE_P(BasicCases, YushkovaPHypercubeFunctional, ::testing::Values(1, 2, 3, 4, 5, 8, 12));

TEST(YushkovaPHypercubeValidation, SeqRejectsInvalidInput) {
  YushkovaPHypercubeSEQ bad_zero(0);
  YushkovaPHypercubeSEQ bad_negative(-2);
  YushkovaPHypercubeSEQ bad_large(63);

  EXPECT_FALSE(bad_zero.Validation());
  EXPECT_FALSE(bad_negative.Validation());
  EXPECT_FALSE(bad_large.Validation());
}

TEST(YushkovaPHypercubeValidation, MpiValidationDependsOnWorldAndInput) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }

  int world_size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  {
    YushkovaPHypercubeMPI valid_input(6);
    EXPECT_EQ(valid_input.Validation(), IsPowerOfTwo(world_size));
  }
  {
    YushkovaPHypercubeMPI invalid_input(0);
    EXPECT_FALSE(invalid_input.Validation());
  }

  ppc::util::DestructorFailureFlag::Unset();
}

}  // namespace yushkova_p_hypercube
