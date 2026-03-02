#include <gtest/gtest.h>
#include <mpi.h>

#include <cstddef>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "luchnikov_e_gener_transm_from_all_to_one_gather/common/include/common.hpp"
#include "luchnikov_e_gener_transm_from_all_to_one_gather/mpi/include/ops_mpi.hpp"
#include "luchnikov_e_gener_transm_from_all_to_one_gather/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace luchnikov_e_gener_transm_from_all_to_one_gather {

class LuchnikovEGenerTransmFromAllToOneGatherPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  static constexpr size_t kCount = 1000;
  InType input_data_{};

  void SetUp() override {
    input_data_.resize(kCount);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(1, 1000);

    for (size_t i = 0; i < kCount; ++i) {
      input_data_[i] = dist(gen);
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // Получаем указатель на задачу через защищённый метод базового класса
    auto *task_ptr = this->GetTask();

    // Проверяем тип задачи через dynamic_cast
    if (dynamic_cast<LuchnikovEGenerTransmFromAllToOneGatherMPI *>(task_ptr) != nullptr) {
      int rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      if (rank != 0) {
        return true;
      }
    }

    OutType expected = input_data_;
    std::sort(expected.begin(), expected.end());
    return expected == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(LuchnikovEGenerTransmFromAllToOneGatherPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

namespace {

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, LuchnikovEGenerTransmFromAllToOneGatherMPI,
                                                       LuchnikovEGenerTransmFromAllToOneGatherSEQ>(
    PPC_SETTINGS_luchnikov_e_gener_transm_from_all_to_one_gather);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = LuchnikovEGenerTransmFromAllToOneGatherPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, LuchnikovEGenerTransmFromAllToOneGatherPerfTestProcesses, kGtestValues,
                         kPerfTestName);

}  // namespace

}  // namespace luchnikov_e_gener_transm_from_all_to_one_gather
