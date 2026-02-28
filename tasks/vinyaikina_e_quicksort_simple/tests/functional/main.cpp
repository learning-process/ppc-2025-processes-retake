#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"
#include "vinyaikina_e_quicksort_simple/common/include/common.hpp"
#include "vinyaikina_e_quicksort_simple/mpi/include/ops_mpi.hpp"
#include "vinyaikina_e_quicksort_simple/seq/include/ops_seq.hpp"

namespace vinyaikina_e_quicksort_simple {

class VinyaikinaEQuicksortSimpleFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int size = std::get<0>(params);
    input_data_.resize(size);
    for (int i = 0; i < size; i++) {
      input_data_[i] = size - i;
    }
    expected_output_ = input_data_;
    std::ranges::sort(expected_output_);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return expected_output_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_;
};

namespace {

TEST_P(VinyaikinaEQuicksortSimpleFuncTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {std::make_tuple(3, "3"), std::make_tuple(5, "5"), std::make_tuple(7, "7")};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<VinyaikinaEQuicksortSimpleMPI, InType>(
                                               kTestParam, PPC_SETTINGS_vinyaikina_e_quicksort_simple),
                                           ppc::util::AddFuncTask<VinyaikinaEQuicksortSimpleSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_vinyaikina_e_quicksort_simple));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = VinyaikinaEQuicksortSimpleFuncTests::PrintFuncTestName<VinyaikinaEQuicksortSimpleFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, VinyaikinaEQuicksortSimpleFuncTests, kGtestValues, kPerfTestName);

TEST(VinyaikinaEQuicksortSimpleSEQEdge, EmptyArray) {
  std::vector<int> input;
  auto task = std::make_shared<VinyaikinaEQuicksortSimpleSEQ>(input);
  ASSERT_TRUE(task->Validation());
  ASSERT_TRUE(task->PreProcessing());
  ASSERT_TRUE(task->Run());
  ASSERT_TRUE(task->PostProcessing());
  ASSERT_TRUE(task->GetOutput().empty());
}

TEST(VinyaikinaEQuicksortSimpleSEQEdge, SingleElement) {
  std::vector<int> input = {42};
  auto task = std::make_shared<VinyaikinaEQuicksortSimpleSEQ>(input);
  ASSERT_TRUE(task->Validation());
  ASSERT_TRUE(task->PreProcessing());
  ASSERT_TRUE(task->Run());
  ASSERT_TRUE(task->PostProcessing());
  std::vector<int> expected = {42};
  ASSERT_EQ(task->GetOutput(), expected);
}

TEST(VinyaikinaEQuicksortSimpleSEQEdge, AlreadySorted) {
  std::vector<int> input = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  auto task = std::make_shared<VinyaikinaEQuicksortSimpleSEQ>(input);
  ASSERT_TRUE(task->Validation());
  ASSERT_TRUE(task->PreProcessing());
  ASSERT_TRUE(task->Run());
  ASSERT_TRUE(task->PostProcessing());
  ASSERT_EQ(task->GetOutput(), input);
}

TEST(VinyaikinaEQuicksortSimpleSEQEdge, ReverseSorted) {
  std::vector<int> input = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
  auto task = std::make_shared<VinyaikinaEQuicksortSimpleSEQ>(input);
  ASSERT_TRUE(task->Validation());
  ASSERT_TRUE(task->PreProcessing());
  ASSERT_TRUE(task->Run());
  ASSERT_TRUE(task->PostProcessing());
  std::vector<int> expected = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  ASSERT_EQ(task->GetOutput(), expected);
}

TEST(VinyaikinaEQuicksortSimpleSEQEdge, AllSameElements) {
  std::vector<int> input = {5, 5, 5, 5, 5};
  auto task = std::make_shared<VinyaikinaEQuicksortSimpleSEQ>(input);
  ASSERT_TRUE(task->Validation());
  ASSERT_TRUE(task->PreProcessing());
  ASSERT_TRUE(task->Run());
  ASSERT_TRUE(task->PostProcessing());
  ASSERT_EQ(task->GetOutput(), input);
}

TEST(VinyaikinaEQuicksortSimpleSEQEdge, NegativeNumbers) {
  std::vector<int> input = {-3, -1, -7, -5, -2};
  auto task = std::make_shared<VinyaikinaEQuicksortSimpleSEQ>(input);
  ASSERT_TRUE(task->Validation());
  ASSERT_TRUE(task->PreProcessing());
  ASSERT_TRUE(task->Run());
  ASSERT_TRUE(task->PostProcessing());
  std::vector<int> expected = {-7, -5, -3, -2, -1};
  ASSERT_EQ(task->GetOutput(), expected);
}

TEST(VinyaikinaEQuicksortSimpleSEQEdge, MixedPositiveNegative) {
  std::vector<int> input = {3, -1, 4, -1, 5, -9, 2, -6};
  auto task = std::make_shared<VinyaikinaEQuicksortSimpleSEQ>(input);
  ASSERT_TRUE(task->Validation());
  ASSERT_TRUE(task->PreProcessing());
  ASSERT_TRUE(task->Run());
  ASSERT_TRUE(task->PostProcessing());
  std::vector<int> expected = {-9, -6, -1, -1, 2, 3, 4, 5};
  ASSERT_EQ(task->GetOutput(), expected);
}

TEST(VinyaikinaEQuicksortSimpleSEQEdge, WithDuplicates) {
  std::vector<int> input = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5};
  auto task = std::make_shared<VinyaikinaEQuicksortSimpleSEQ>(input);
  ASSERT_TRUE(task->Validation());
  ASSERT_TRUE(task->PreProcessing());
  ASSERT_TRUE(task->Run());
  ASSERT_TRUE(task->PostProcessing());
  std::vector<int> expected = {1, 1, 2, 3, 3, 4, 5, 5, 5, 6, 9};
  ASSERT_EQ(task->GetOutput(), expected);
}

TEST(VinyaikinaEQuicksortSimpleSEQEdge, TwoElements) {
  std::vector<int> input = {2, 1};
  auto task = std::make_shared<VinyaikinaEQuicksortSimpleSEQ>(input);
  ASSERT_TRUE(task->Validation());
  ASSERT_TRUE(task->PreProcessing());
  ASSERT_TRUE(task->Run());
  ASSERT_TRUE(task->PostProcessing());
  std::vector<int> expected = {1, 2};
  ASSERT_EQ(task->GetOutput(), expected);
}

TEST(VinyaikinaEQuicksortSimpleSEQEdge, TwoElementsSorted) {
  std::vector<int> input = {1, 2};
  auto task = std::make_shared<VinyaikinaEQuicksortSimpleSEQ>(input);
  ASSERT_TRUE(task->Validation());
  ASSERT_TRUE(task->PreProcessing());
  ASSERT_TRUE(task->Run());
  ASSERT_TRUE(task->PostProcessing());
  ASSERT_EQ(task->GetOutput(), input);
}

TEST(VinyaikinaEQuicksortSimpleSEQEdge, LargerArray) {
  const int sz = 500;
  std::vector<int> input(sz);
  for (int i = 0; i < sz; i++) {
    input[i] = sz - i;
  }
  auto task = std::make_shared<VinyaikinaEQuicksortSimpleSEQ>(input);
  ASSERT_TRUE(task->Validation());
  ASSERT_TRUE(task->PreProcessing());
  ASSERT_TRUE(task->Run());
  ASSERT_TRUE(task->PostProcessing());
  std::vector<int> expected(sz);
  for (int i = 0; i < sz; i++) {
    expected[i] = i + 1;
  }
  ASSERT_EQ(task->GetOutput(), expected);
}

TEST(VinyaikinaEQuicksortSimpleMPIEdge, EmptyArray) {
  std::vector<int> input;
  auto task = std::make_shared<VinyaikinaEQuicksortSimpleMPI>(input);
  ASSERT_TRUE(task->Validation());
  ASSERT_TRUE(task->PreProcessing());
  ASSERT_TRUE(task->Run());
  ASSERT_TRUE(task->PostProcessing());
  ASSERT_TRUE(task->GetOutput().empty());
}

TEST(VinyaikinaEQuicksortSimpleMPIEdge, SingleElement) {
  std::vector<int> input = {42};
  auto task = std::make_shared<VinyaikinaEQuicksortSimpleMPI>(input);
  ASSERT_TRUE(task->Validation());
  ASSERT_TRUE(task->PreProcessing());
  ASSERT_TRUE(task->Run());
  ASSERT_TRUE(task->PostProcessing());
  std::vector<int> expected = {42};
  ASSERT_EQ(task->GetOutput(), expected);
}

TEST(VinyaikinaEQuicksortSimpleMPIEdge, AlreadySorted) {
  std::vector<int> input = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  auto task = std::make_shared<VinyaikinaEQuicksortSimpleMPI>(input);
  ASSERT_TRUE(task->Validation());
  ASSERT_TRUE(task->PreProcessing());
  ASSERT_TRUE(task->Run());
  ASSERT_TRUE(task->PostProcessing());
  ASSERT_EQ(task->GetOutput(), input);
}

TEST(VinyaikinaEQuicksortSimpleMPIEdge, NegativeNumbers) {
  std::vector<int> input = {-3, -1, -7, -5, -2};
  auto task = std::make_shared<VinyaikinaEQuicksortSimpleMPI>(input);
  ASSERT_TRUE(task->Validation());
  ASSERT_TRUE(task->PreProcessing());
  ASSERT_TRUE(task->Run());
  ASSERT_TRUE(task->PostProcessing());
  std::vector<int> expected = {-7, -5, -3, -2, -1};
  ASSERT_EQ(task->GetOutput(), expected);
}

TEST(VinyaikinaEQuicksortSimpleMPIEdge, WithDuplicates) {
  std::vector<int> input = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5};
  auto task = std::make_shared<VinyaikinaEQuicksortSimpleMPI>(input);
  ASSERT_TRUE(task->Validation());
  ASSERT_TRUE(task->PreProcessing());
  ASSERT_TRUE(task->Run());
  ASSERT_TRUE(task->PostProcessing());
  std::vector<int> expected = {1, 1, 2, 3, 3, 4, 5, 5, 5, 6, 9};
  ASSERT_EQ(task->GetOutput(), expected);
}

TEST(VinyaikinaEQuicksortSimpleMPIEdge, AllSameElements) {
  std::vector<int> input = {7, 7, 7, 7, 7, 7};
  auto task = std::make_shared<VinyaikinaEQuicksortSimpleMPI>(input);
  ASSERT_TRUE(task->Validation());
  ASSERT_TRUE(task->PreProcessing());
  ASSERT_TRUE(task->Run());
  ASSERT_TRUE(task->PostProcessing());
  ASSERT_EQ(task->GetOutput(), input);
}

TEST(VinyaikinaEQuicksortSimpleMPIEdge, TwoElements) {
  std::vector<int> input = {2, 1};
  auto task = std::make_shared<VinyaikinaEQuicksortSimpleMPI>(input);
  ASSERT_TRUE(task->Validation());
  ASSERT_TRUE(task->PreProcessing());
  ASSERT_TRUE(task->Run());
  ASSERT_TRUE(task->PostProcessing());
  std::vector<int> expected = {1, 2};
  ASSERT_EQ(task->GetOutput(), expected);
}

TEST(VinyaikinaEQuicksortSimpleMPIEdge, LargerArray) {
  const int sz = 500;
  std::vector<int> input(sz);
  for (int i = 0; i < sz; i++) {
    input[i] = sz - i;
  }
  auto task = std::make_shared<VinyaikinaEQuicksortSimpleMPI>(input);
  ASSERT_TRUE(task->Validation());
  ASSERT_TRUE(task->PreProcessing());
  ASSERT_TRUE(task->Run());
  ASSERT_TRUE(task->PostProcessing());
  std::vector<int> expected(sz);
  for (int i = 0; i < sz; i++) {
    expected[i] = i + 1;
  }
  ASSERT_EQ(task->GetOutput(), expected);
}

}  // namespace

}  // namespace vinyaikina_e_quicksort_simple
