#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <random>
#include <string>
#include <vector>

#include "luchnikov_e_gener_transm_from_all_to_one_gather/common/include/common.hpp"
#include "luchnikov_e_gener_transm_from_all_to_one_gather/mpi/include/ops_mpi.hpp"
#include "luchnikov_e_gener_transm_from_all_to_one_gather/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace luchnikov_e_gener_transm_from_all_to_one_gather {

class LuchnikovEGenerTransmFromAllToOneGatherFuncTestsProcesses
    : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    size_t size = std::get<0>(params);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(1, 100);

    input_data_.resize(size);
    for (size_t i = 0; i < size; ++i) {
      input_data_[i] = dist(gen);
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (this->GetTask()->GetTypeOfTask() == ppc::task::TypeOfTask::kMPI) {
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

 private:
  InType input_data_;
};

namespace {

TEST_P(LuchnikovEGenerTransmFromAllToOneGatherFuncTestsProcesses, GatherTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 10> kTestParam = {
    std::make_tuple(1, "1"),   std::make_tuple(2, "2"),   std::make_tuple(3, "3"),   std::make_tuple(5, "5"),
    std::make_tuple(7, "7"),   std::make_tuple(10, "10"), std::make_tuple(15, "15"), std::make_tuple(20, "20"),
    std::make_tuple(25, "25"), std::make_tuple(30, "30")};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<LuchnikovEGenerTransmFromAllToOneGatherMPI, InType>(
                       kTestParam, PPC_SETTINGS_luchnikov_e_gener_transm_from_all_to_one_gather),
                   ppc::util::AddFuncTask<LuchnikovEGenerTransmFromAllToOneGatherSEQ, InType>(
                       kTestParam, PPC_SETTINGS_luchnikov_e_gener_transm_from_all_to_one_gather));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = LuchnikovEGenerTransmFromAllToOneGatherFuncTestsProcesses::PrintFuncTestName<
    LuchnikovEGenerTransmFromAllToOneGatherFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(AllToGatherTests, LuchnikovEGenerTransmFromAllToOneGatherFuncTestsProcesses, kGtestValues,
                         kPerfTestName);

}  // namespace

}  // namespace luchnikov_e_gener_transm_from_all_to_one_gather
