#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <random>
#include <string>
#include <tuple>

#include "nazyrov_a_min_val_vec/common/include/common.hpp"
#include "nazyrov_a_min_val_vec/mpi/include/ops_mpi.hpp"
#include "nazyrov_a_min_val_vec/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace nazyrov_a_min_val_vec {

class NazyrovAMinValVecFuncTest : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int test_case = std::get<0>(params);

    switch (test_case) {
      case 0:
        input_data_ = {5, 3, 1, 4, 2};
        expected_ = 1;
        break;
      case 1:
        input_data_ = {42};
        expected_ = 42;
        break;
      case 2:
        input_data_ = {-10, -20, -5, -1, -100};
        expected_ = -100;
        break;
      case 3:
        input_data_ = {7, 7, 7, 7, 7};
        expected_ = 7;
        break;
      case 4: {
        input_data_.resize(1000);
        std::mt19937 gen(std::random_device{}());
        std::uniform_int_distribution<int> dist(-10000, 10000);
        for (std::size_t i = 0; i < input_data_.size(); ++i) {
          input_data_[i] = dist(gen);
        }
        expected_ = *std::ranges::min_element(input_data_);
      } break;
      case 5:
        input_data_ = {100, 50, 25, 10, 5, 1, 0, -1};
        expected_ = -1;
        break;
      case 6: {
        input_data_.resize(500);
        for (std::size_t i = 0; i < input_data_.size(); ++i) {
          input_data_[i] = static_cast<int>(input_data_.size() - i);
        }
        expected_ = 1;
      } break;
      case 7:
        input_data_ = {-999999, 0, 999999};
        expected_ = -999999;
        break;
      default:
        input_data_ = {0};
        expected_ = 0;
        break;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (ppc::util::IsUnderMpirun()) {
      int rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      if (rank != 0) {
        return true;
      }
    }
    return output_data == expected_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_{0};
};

namespace {

TEST_P(NazyrovAMinValVecFuncTest, MinValVec) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 8> kTestParam = {
    std::make_tuple(0, "basic"),         std::make_tuple(1, "single"),      std::make_tuple(2, "all_negative"),
    std::make_tuple(3, "all_equal"),     std::make_tuple(4, "random_1000"), std::make_tuple(5, "descending"),
    std::make_tuple(6, "ascending_500"), std::make_tuple(7, "extremes"),
};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<MinValVecMPI, InType>(kTestParam, PPC_SETTINGS_nazyrov_a_min_val_vec),
                   ppc::util::AddFuncTask<MinValVecSEQ, InType>(kTestParam, PPC_SETTINGS_nazyrov_a_min_val_vec));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kFuncTestName = NazyrovAMinValVecFuncTest::PrintFuncTestName<NazyrovAMinValVecFuncTest>;

INSTANTIATE_TEST_SUITE_P(MinValVecTests, NazyrovAMinValVecFuncTest, kGtestValues, kFuncTestName);

}  // namespace

}  // namespace nazyrov_a_min_val_vec
