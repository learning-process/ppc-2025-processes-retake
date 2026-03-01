#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <cstddef>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "denisov_a_ring/common/include/common.hpp"
#include "denisov_a_ring/mpi/include/ops_mpi.hpp"
#include "denisov_a_ring/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace denisov_a_ring {

class DenisovARingFuncTest : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &param) {
    return std::to_string(std::get<0>(param)) + "_" + std::get<1>(param);
  }

 protected:
  void SetUp() override {
    int world_size = 1;

    if (ppc::util::IsUnderMpirun()) {
      MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    } else {
      world_size = ppc::util::GetNumProc();
    }

    const auto &params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    int vec_len = std::get<0>(params);

    input_.source = 0;
    input_.destination = (world_size > 1 ? world_size - 1 : 0);
    input_.data.resize(static_cast<std::size_t>(vec_len));

    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> dist(-1000, 1000);

    for (int i = 0; i < vec_len; ++i) {
      input_.data[static_cast<std::size_t>(i)] = dist(rng);
    }

    expected_ = input_.data;
  }

  bool CheckTestOutputData(OutType &out) final {
    if (ppc::util::IsUnderMpirun()) {
      int rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      if (rank != 0) {
        return true;
      }
    }
    return out == expected_;
  }

  InType GetTestInputData() final {
    return input_;
  }

 private:
  InType input_{};
  OutType expected_;
};

namespace {

TEST_P(DenisovARingFuncTest, RingTransfer) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 6> kTestParams = {
    std::make_tuple(1, "single"),   std::make_tuple(5, "small"),       std::make_tuple(50, "medium_50"),
    std::make_tuple(100, "medium"), std::make_tuple(500, "large_500"), std::make_tuple(1000, "large"),
};

const auto kTasks =
    std::tuple_cat(ppc::util::AddFuncTask<RingTopologyMPI, InType>(kTestParams, PPC_SETTINGS_denisov_a_ring),
                   ppc::util::AddFuncTask<RingTopologySEQ, InType>(kTestParams, PPC_SETTINGS_denisov_a_ring));

const auto kValues = ppc::util::ExpandToValues(kTasks);

const auto kNameGen = DenisovARingFuncTest::PrintFuncTestName<DenisovARingFuncTest>;

INSTANTIATE_TEST_SUITE_P(RingTopologyTests, DenisovARingFuncTest, kValues, kNameGen);

}  // namespace

}  // namespace denisov_a_ring
