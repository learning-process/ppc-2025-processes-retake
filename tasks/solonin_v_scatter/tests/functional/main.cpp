#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <cstddef>
#include <numeric>
#include <string>
#include <tuple>
#include <vector>

#include "solonin_v_scatter/common/include/common.hpp"
#include "solonin_v_scatter/mpi/include/ops_mpi.hpp"
#include "solonin_v_scatter/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace solonin_v_scatter {

class SoloninVScatterFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &tp) {
    return std::to_string(std::get<0>(tp)) + "_scatter";
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    data_ = std::get<1>(params);
    send_count_ = std::get<2>(params);
    root_ = std::get<3>(params);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Expected: each rank gets its chunk
    expected_.assign(data_.begin() + rank * send_count_,
                     data_.begin() + rank * send_count_ + send_count_);
  }

  bool CheckTestOutputData(OutType &out) final {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (out.size() != static_cast<size_t>(send_count_)) return false;
    for (int i = 0; i < send_count_; i++) {
      if (out[i] != expected_[i]) return false;
    }
    return true;
  }

  InType GetTestInputData() final {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == root_) {
      return std::make_tuple(data_, send_count_, root_);
    }
    return std::make_tuple(std::vector<int>(), send_count_, root_);
  }

 private:
  std::vector<int> data_;
  std::vector<int> expected_;
  int send_count_{1};
  int root_{0};
};

namespace {

// Build test data: total = send_count * num_procs elements, values = 0..total-1
std::vector<int> MakeData(int send_count, int num_procs) {
  std::vector<int> v(send_count * num_procs);
  std::iota(v.begin(), v.end(), 0);
  return v;
}

// Helper to create TestType for a given world size
// (For CI: we target 4 processes max, so data always has 4*count elements)
constexpr int kMaxProcs = 4;

const std::array<TestType, 10> kTests = {
    std::make_tuple(1, MakeData(1, kMaxProcs), 1, 0),
    std::make_tuple(2, MakeData(2, kMaxProcs), 2, 0),
    std::make_tuple(3, MakeData(4, kMaxProcs), 4, 0),
    std::make_tuple(4, MakeData(8, kMaxProcs), 8, 0),
    std::make_tuple(5, MakeData(16, kMaxProcs), 16, 0),
    std::make_tuple(6, MakeData(32, kMaxProcs), 32, 0),
    std::make_tuple(7, MakeData(64, kMaxProcs), 64, 0),
    std::make_tuple(8, MakeData(100, kMaxProcs), 100, 0),
    std::make_tuple(9, MakeData(256, kMaxProcs), 256, 0),
    std::make_tuple(10, MakeData(512, kMaxProcs), 512, 0),
};

TEST_P(SoloninVScatterFuncTests, FunctionalTests) { ExecuteTest(GetParam()); }

const auto kTaskList = std::tuple_cat(
    ppc::util::AddFuncTask<SoloninVScatterMPI, InType>(kTests, PPC_SETTINGS_solonin_v_scatter),
    ppc::util::AddFuncTask<SoloninVScatterSEQ, InType>(kTests, PPC_SETTINGS_solonin_v_scatter));

const auto kGtestValues = ppc::util::ExpandToValues(kTaskList);
const auto kTestName = SoloninVScatterFuncTests::PrintFuncTestName<SoloninVScatterFuncTests>;

INSTANTIATE_TEST_SUITE_P(ScatterSuite, SoloninVScatterFuncTests, kGtestValues, kTestName);

TEST(SoloninVScatterValidation, RejectsEmptyBuffer) {
  SoloninVScatterSEQ task(std::make_tuple(std::vector<int>(), 1, 0));
  EXPECT_FALSE(task.Validation());
}

TEST(SoloninVScatterValidation, RejectsZeroCount) {
  SoloninVScatterSEQ task(std::make_tuple(std::vector<int>{1, 2, 3}, 0, 0));
  EXPECT_FALSE(task.Validation());
}

TEST(SoloninVScatterValidation, AcceptsValidInput) {
  SoloninVScatterSEQ task(std::make_tuple(std::vector<int>{1, 2, 3, 4}, 2, 0));
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());
}

}  // namespace

}  // namespace solonin_v_scatter
