#include <gtest/gtest.h>

#include "nozdrin_a_scalar_mult_vectors/common/include/common.hpp"
#include "nozdrin_a_scalar_mult_vectors/mpi/include/ops_mpi.hpp"
#include "nozdrin_a_scalar_mult_vectors/seq/include/ops_seq.hpp"

using nozdrin_a_scalar_mult_vectors::GenerateRandomVector;
using nozdrin_a_scalar_mult_vectors::InType;
using nozdrin_a_scalar_mult_vectors::NozdrinAScalarMultVectorsMPI;
using nozdrin_a_scalar_mult_vectors::NozdrinAScalarMultVectorsSEQ;

namespace {
template <typename T>
void ValidateAndPreProcess(T &task) {
    ASSERT_TRUE(task.Validation());
    ASSERT_TRUE(task.PreProcessing());
}

template <typename T>
void RunAndPostProcess(T &task) {
    ASSERT_TRUE(task.Run());
    ASSERT_TRUE(task.PostProcessing());
}
}  // namespace

class NozdrinScalarMultVectorsPerfTest : public ::testing::Test {
 protected:
    template <typename T>
    void RunSequence(T &task) {
        ValidateAndPreProcess(task);
        RunAndPostProcess(task);
    }
};

TEST_F(NozdrinScalarMultVectorsPerfTest, SeqPerformance) {
    InType in{GenerateRandomVector(1'000'000), GenerateRandomVector(1'000'000)};
    NozdrinAScalarMultVectorsSEQ task(in);
    RunSequence(task);
}

TEST_F(NozdrinScalarMultVectorsPerfTest, MpiPerformance) {
    InType in{GenerateRandomVector(1'000'000), GenerateRandomVector(1'000'000)};
    NozdrinAScalarMultVectorsMPI task(in);
    RunSequence(task);
}

// namespace nozdrin_a_scalar_mult_vectors
