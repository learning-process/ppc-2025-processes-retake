#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <cstddef>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "nazyrov_a_broadcast/common/include/common.hpp"
#include "nazyrov_a_broadcast/mpi/include/ops_mpi.hpp"
#include "nazyrov_a_broadcast/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace nazyrov_a_broadcast {

class NazyrovABroadcastFuncTest : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int test_case = std::get<0>(params);

    int size = 1;
    if (ppc::util::IsUnderMpirun()) {
      MPI_Comm_size(MPI_COMM_WORLD, &size);
    } else {
      size = ppc::util::GetNumProc();
    }

    switch (test_case) {
      case 0:
        input_data_.root = 0;
        input_data_.data = {1, 2, 3, 4, 5};
        break;
      case 1:
        input_data_.root = 0;
        input_data_.data = {42};
        break;
      case 2: {
        input_data_.root = 0;
        input_data_.data.resize(100);
        std::mt19937 gen(std::random_device{}());
        std::uniform_int_distribution<int> dist(-1000, 1000);
        for (std::size_t i = 0; i < input_data_.data.size(); ++i) {
          input_data_.data[i] = dist(gen);
        }
      } break;
      case 3:
        input_data_.root = (size > 1) ? size - 1 : 0;
        input_data_.data = {10, 20, 30, 40, 50};
        break;
      case 4: {
        input_data_.root = (size > 2) ? size / 2 : 0;
        input_data_.data.resize(1000);
        std::mt19937 gen(std::random_device{}());
        std::uniform_int_distribution<int> dist(-10000, 10000);
        for (std::size_t i = 0; i < input_data_.data.size(); ++i) {
          input_data_.data[i] = dist(gen);
        }
      } break;
      case 5:
        input_data_.root = 0;
        input_data_.data = {-1, -2, -3, -4, -5, -6, -7, -8};
        break;
      default:
        input_data_.root = 0;
        input_data_.data = {0};
        break;
    }

    expected_output_ = input_data_.data;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (ppc::util::IsUnderMpirun()) {
      int rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      if (rank != 0) {
        return true;
      }
    }
    return output_data == expected_output_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_{};
  OutType expected_output_;
};

namespace {

TEST_P(NazyrovABroadcastFuncTest, TreeBroadcast) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 6> kTestParam = {
    std::make_tuple(0, "basic_5"),   std::make_tuple(1, "single_elem"), std::make_tuple(2, "random_100"),
    std::make_tuple(3, "last_root"), std::make_tuple(4, "mid_root"),    std::make_tuple(5, "negative"),
};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<BroadcastMPI, InType>(kTestParam, PPC_SETTINGS_nazyrov_a_broadcast),
                   ppc::util::AddFuncTask<BroadcastSEQ, InType>(kTestParam, PPC_SETTINGS_nazyrov_a_broadcast));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kFuncTestName = NazyrovABroadcastFuncTest::PrintFuncTestName<NazyrovABroadcastFuncTest>;

INSTANTIATE_TEST_SUITE_P(BroadcastTests, NazyrovABroadcastFuncTest, kGtestValues, kFuncTestName);

}  // namespace

}  // namespace nazyrov_a_broadcast
