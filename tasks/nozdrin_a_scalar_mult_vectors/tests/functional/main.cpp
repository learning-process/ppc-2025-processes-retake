#include <gtest/gtest.h>

#include "nozdrin_a_scalar_mult_vectors/common/include/common.hpp"
#include "nozdrin_a_scalar_mult_vectors/mpi/include/ops_mpi.hpp"
#include "nozdrin_a_scalar_mult_vectors/seq/include/ops_seq.hpp"
#include "util/include/util.hpp"

using nozdrin_a_scalar_mult_vectors::AnalyticDotProduct;
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

template <typename T>
void CheckResult(T &task, const InType &in) {
    const double expected = AnalyticDotProduct(in.a, in.b);
    ASSERT_NEAR(task.GetOutput(), expected, 1e-9);
}
}  // namespace

class NozdrinScalarMultVectorsTest : public ::testing::Test {
 protected:
    void SetUp() override {
        ppc::util::DestructorFailureFlag::Unset();
    }

    void TearDown() override {
        ppc::util::DestructorFailureFlag::Unset();
    }

    template <typename T>
    void RunAndCheck(T &task, const InType &in) {
        ValidateAndPreProcess(task);
        RunAndPostProcess(task);
        CheckResult(task, in);
    }
};

TEST_F(NozdrinScalarMultVectorsTest, SeqSmallVector) {
    InType in{GenerateRandomVector(100), GenerateRandomVector(100)};
    NozdrinAScalarMultVectorsSEQ task(in);
    RunAndCheck(task, in);
}

TEST_F(NozdrinScalarMultVectorsTest, SeqDeterministic) {
    InType in{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    NozdrinAScalarMultVectorsSEQ task(in);
    RunAndCheck(task, in);
}

TEST_F(NozdrinScalarMultVectorsTest, MpiDeterministic) {
    InType in{{1.0, 2.0, 3.0, 4.0}, {5.0, 6.0, 7.0, 8.0}};
    NozdrinAScalarMultVectorsMPI task(in);
    RunAndCheck(task, in);
}

TEST_F(NozdrinScalarMultVectorsTest, ValidationFailsOnDifferentSizes) {
    InType in{{1.0, 2.0}, {3.0}};
    NozdrinAScalarMultVectorsSEQ task(in);
    ASSERT_FALSE(task.Validation());
    ppc::util::DestructorFailureFlag::Unset();
}

// namespace nozdrin_a_scalar_mult_vectors
