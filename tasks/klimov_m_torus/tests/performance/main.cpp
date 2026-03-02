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
  InType test_data;
  bool data_ready = false;
  int world_size = 1;
  int rank = 0;
  bool is_seq_mode = false;

  void SetUp() override {
    std::string task_name = std::get<1>(GetParam());
    is_seq_mode = (task_name.find("seq") != std::string::npos);

    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized != 0) {
      MPI_Comm_size(MPI_COMM_WORLD, &world_size);
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }

    PrepareTestData();
  }

  void PrepareTestData() {
    if (data_ready) {
      return;
    }

    const int data_size = 10000000;

    test_data.sender = 0;
    if (is_seq_mode) {
      test_data.receiver = 0;
    } else {
      test_data.receiver = (world_size > 1) ? (world_size - 1) : 0;
    }

    test_data.data.resize(data_size);
    for (int i = 0; i < data_size; ++i) {
      test_data.data[i] = i + 1;
    }

    data_ready = true;
  }

  InType GetTestInputData() override {
    return test_data;
  }

  bool CheckTestOutputData(OutType &out) override {
    std::string task_name = std::get<1>(GetParam());
    bool is_mpi = (task_name.find("mpi") != std::string::npos);

    if (is_mpi) {
      if (rank != test_data.receiver) {
        return out.received_data.empty() && out.route.empty();
      }
      if (out.received_data.size() != test_data.data.size()) {
        return false;
      }
      if (out.received_data.empty()) {
        return true;
      }
      return (out.received_data.front() == test_data.data.front() && out.received_data.back() == test_data.data.back());
    }

    if (rank == 0) {
      if (out.received_data.size() != test_data.data.size()) {
        return false;
      }
      if (out.received_data.empty()) {
        return true;
      }
      return (out.received_data.front() == test_data.data.front() && out.received_data.back() == test_data.data.back());
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
