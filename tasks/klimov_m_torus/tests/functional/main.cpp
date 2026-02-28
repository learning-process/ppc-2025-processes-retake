#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <string>
#include <tuple>
#include <vector>

#include "klimov_m_torus/common/include/common.hpp"
#include "klimov_m_torus/mpi/include/ops_mpi.hpp"
#include "klimov_m_torus/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace klimov_m_torus {

class TorusNetworkTest : public ppc::util::BaseRunFuncTests<InType, OutType, TestParam> {
 public:
  static std::string PrintTestParam(const TestParam &test_param) {
    int size = std::get<0>(test_param);
    return "size" + std::to_string(size);
  }

 protected:
  void SetUp() override {
    TestParam params = std::get<2>(GetParam());
    int data_size = std::get<0>(params);

    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized != 0) {
      MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
      MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    } else {
      world_size_ = 1;
      rank_ = 0;
    }

    source_ = 0;
    dest_ = (world_size_ > 1) ? (world_size_ - 1) : 0;

    input_.source = source_;
    input_.dest = dest_;
    input_.payload.clear();
    for (int i = 0; i < data_size; ++i) {
      input_.payload.push_back(i + 1);
    }
    expected_payload_ = input_.payload;
  }

  InType GetTestInputData() override {
    return input_;
  }

  bool CheckTestOutputData(OutType &out) override {
    const std::string &task_name = std::get<1>(GetParam());
    const bool is_seq = (task_name.find("seq") != std::string::npos);

    return is_seq ? CheckSequential(out) : CheckParallel(out);
  }

 private:
  [[nodiscard]] bool CheckSequential(const OutType &out) const {
    if (out.payload != expected_payload_) return false;
    if (out.path.empty()) return false;
    if (out.path.front() != source_) return false;
    if (out.path.back() != dest_) return false;
    return true;
  }

  [[nodiscard]] bool CheckParallel(const OutType &out) const {
    if (rank_ != dest_) {
      return out.payload.empty() && out.path.empty();
    }
    if (out.payload != expected_payload_) return false;
    if (out.path.empty()) return false;
    if (out.path.front() != source_) return false;
    if (out.path.back() != dest_) return false;
    return true;
  }

  InType input_{};
  std::vector<int> expected_payload_;
  int world_size_{1};
  int rank_{0};
  int source_{0};
  int dest_{0};
};

namespace {

const std::array<TestParam, 5> kTestParams = {
    std::make_tuple(1), std::make_tuple(4), std::make_tuple(8), std::make_tuple(16), std::make_tuple(32),
};

const auto kTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<TorusNetworkMpi, InType>(kTestParams, "tasks/klimov_m_torus/settings.json"),
                   ppc::util::AddFuncTask<TorusSequential, InType>(kTestParams, "tasks/klimov_m_torus/settings.json"));

const auto kValues = ppc::util::ExpandToValues(kTasksList);
const auto kNamePrinter = TorusNetworkTest::PrintFuncTestName<TorusNetworkTest>;

TEST_P(TorusNetworkTest, TorusDataTransfer) {
  ExecuteTest(GetParam());
}

INSTANTIATE_TEST_SUITE_P(TorusTests, TorusNetworkTest, kValues, kNamePrinter);

}  // namespace
}  // namespace klimov_m_torus