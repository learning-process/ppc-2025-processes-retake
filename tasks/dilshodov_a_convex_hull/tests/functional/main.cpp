#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "dilshodov_a_convex_hull/common/include/common.hpp"
#include "dilshodov_a_convex_hull/mpi/include/ops_mpi.hpp"
#include "dilshodov_a_convex_hull/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace dilshodov_a_convex_hull {

namespace {

bool PointLess(const Point &a, const Point &b) {
  return a.x < b.x || (a.x == b.x && a.y < b.y);
}

bool CompareHulls(std::vector<Point> actual, std::vector<Point> expected) {
  if (actual.size() != expected.size()) {
    return false;
  }
  std::ranges::sort(actual, PointLess);
  std::ranges::sort(expected, PointLess);
  return actual == expected;
}

InType MakeImage(int width, int height, const std::vector<std::pair<int, int>> &pixels) {
  InType img;
  img.push_back(width);
  img.push_back(height);
  img.resize(static_cast<std::size_t>(width * height) + 2, 0);
  for (const auto &[x, y] : pixels) {
    img[(static_cast<std::size_t>(y) * static_cast<std::size_t>(width)) + static_cast<std::size_t>(x) + 2] = 255;
  }
  return img;
}

}  // namespace

class DilshodovAConvexHullFuncTest : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    auto test_case = std::get<0>(params);

    switch (test_case) {
      case 0:
        input_data_ = MakeImage(5, 5, {{0, 0}, {4, 0}, {0, 4}, {4, 4}, {2, 2}});
        expected_ = {{0, 0}, {4, 0}, {4, 4}, {0, 4}};
        break;
      case 1:
        input_data_ = MakeImage(3, 3, {{0, 0}, {2, 0}, {1, 2}});
        expected_ = {{0, 0}, {2, 0}, {1, 2}};
        break;
      case 2:
        input_data_ = MakeImage(10, 10, {{1, 1}, {5, 1}, {9, 1}, {1, 9}, {5, 9}, {9, 9}, {5, 5}});
        expected_ = {{1, 1}, {9, 1}, {9, 9}, {1, 9}};
        break;
      case 3: {
        int w = 10;
        int h = 10;
        std::vector<std::pair<int, int>> px;
        for (int bx = 0; bx < w; ++bx) {
          px.emplace_back(bx, 0);
          px.emplace_back(bx, h - 1);
        }
        for (int by = 1; by < h - 1; ++by) {
          px.emplace_back(0, by);
          px.emplace_back(w - 1, by);
        }
        input_data_ = MakeImage(w, h, px);
        expected_ = {{0, 0}, {9, 0}, {9, 9}, {0, 9}};
      } break;
      case 4:
        input_data_ = MakeImage(5, 5, {{0, 0}, {1, 0}, {2, 0}, {3, 0}, {4, 0}});
        expected_ = {};
        break;
      case 5:
        input_data_ = MakeImage(5, 5, {{2, 2}});
        expected_ = {};
        break;
      case 6:
        input_data_ = MakeImage(20, 20, {{0, 0}, {19, 0}, {0, 19}, {19, 19}, {10, 10}, {5, 5}, {15, 5}, {5, 15}});
        expected_ = {{0, 0}, {19, 0}, {19, 19}, {0, 19}};
        break;
      case 7: {
        int w = 50;
        int h = 50;
        std::vector<std::pair<int, int>> px;
        for (int fx = 0; fx < w; ++fx) {
          for (int fy = 0; fy < h; ++fy) {
            px.emplace_back(fx, fy);
          }
        }
        input_data_ = MakeImage(w, h, px);
        expected_ = {{0, 0}, {49, 0}, {49, 49}, {0, 49}};
      } break;
      default:
        input_data_ = MakeImage(3, 3, {{1, 1}});
        expected_ = {};
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
    return CompareHulls(output_data, expected_);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_;
};

namespace {

TEST_P(DilshodovAConvexHullFuncTest, ConvexHullGraham) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 8> kTestParam = {
    std::make_tuple(0, "square_with_center"),   std::make_tuple(1, "triangle"),
    std::make_tuple(2, "sparse_points"),        std::make_tuple(3, "border_rectangle"),
    std::make_tuple(4, "collinear_horizontal"), std::make_tuple(5, "single_point"),
    std::make_tuple(6, "large_square"),         std::make_tuple(7, "filled_square"),
};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<ConvexHullMPI, InType>(kTestParam, PPC_SETTINGS_dilshodov_a_convex_hull),
                   ppc::util::AddFuncTask<ConvexHullSEQ, InType>(kTestParam, PPC_SETTINGS_dilshodov_a_convex_hull));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kFuncTestName = DilshodovAConvexHullFuncTest::PrintFuncTestName<DilshodovAConvexHullFuncTest>;

INSTANTIATE_TEST_SUITE_P(ConvexHullTests, DilshodovAConvexHullFuncTest, kGtestValues, kFuncTestName);

}  // namespace

}  // namespace dilshodov_a_convex_hull
