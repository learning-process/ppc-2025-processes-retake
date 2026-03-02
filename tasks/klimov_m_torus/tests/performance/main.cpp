#include <gtest/gtest.h>
#include <mpi.h>

#include <string>

#include "klimov_m_torus/common/include/common.hpp"
#include "klimov_m_torus/mpi/include/ops_mpi.hpp"
#include "klimov_m_torus/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace klimov_m_torus {

class TorusPerformanceTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  InType test_data_;
  bool data_ready_ = false;
  int world_size_ = 1;
  int rank_ = 0;
  bool is_seq_mode_ = false;

  void SetUp() override {
    std::string task_name = std::get<1>(GetParam());
    is_seq_mode_ = (task_name.find("seq") != std::string::npos);

    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized != 0) {
      MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
      MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    }

    PrepareTestData();
  }

  void PrepareTestData() {
    if (data_ready_) {
      return;
    }

    const int data_size = 10000000;

    test_data_.sender = 0;
    if (is_seq_mode_) {
      test_data_.receiver = 0;
    } else {
      test_data_.receiver = (world_size_ > 1) ? (world_size_ - 1) : 0;
    }

    test_data_.data.resize(data_size);
    for (int i = 0; i < data_size; ++i) {
      test_data_.data[i] = i + 1;
    }

    data_ready_ = true;
  }

  InType GetTestInputData() override {
    return test_data_;
  }

  bool CheckTestOutputData(OutType &out) override {
    std::string task_name = std::get<1>(GetParam());
    bool is_mpi = (task_name.find("mpi") != std::string::npos);

    if (is_mpi) {
      if (rank_ != test_data_.receiver) {
        return out.received_data.empty() && out.route.empty();
      }
      if (out.received_data.size() != test_data_.data.size()) {
        return false;
      }
      if (out.received_data.empty()) {
        return true;
      }
      return (out.received_data.front() == test_data_.data.front() &&
              out.received_data.back() == test_data_.data.back());
    }

    if (rank_ == 0) {
      if (out.received_data.size() != test_data_.data.size()) {
        return false;
      }
      if (out.received_data.empty()) {
        return true;
      }
      return (out.received_data.front() == test_data_.data.front() &&
              out.received_data.back() == test_data_.data.back());
    }
    return true;
  }
};

namespace {

const auto kPerfTasksTuples = ppc::util::MakeAllPerfTasks<InType, TorusMeshCommunicator, TorusReferenceImpl>(
    "tasks/klimov_m_torus/settings.json");

const auto kPerfValues = ppc::util::TupleToGTestValues(kPerfTasksTuples);
const auto kPerfNamePrinter = TorusPerformanceTest::CustomPerfTestName;

TEST_P(TorusPerformanceTest, TorusPerformance) {
  ExecuteTest(GetParam());
}

INSTANTIATE_TEST_SUITE_P(TorusPerformanceTests, TorusPerformanceTest, kPerfValues, kPerfNamePrinter);

}  // namespace
}  // namespace klimov_m_torus
