#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <ranges>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "Nazarova_K_Conv_hull_const_Grah/common/include/common.hpp"
#include "Nazarova_K_Conv_hull_const_Grah/mpi/include/ops_mpi.hpp"
#include "Nazarova_K_Conv_hull_const_Grah/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace nazarova_k_conv_hull_const_grah_processes {
namespace {

inline bool LessPivot(const Point& a, const Point& b) {
  if (a.y != b.y) {
    return a.y < b.y;
  }
  return a.x < b.x;
}

inline bool LessXY(const Point& a, const Point& b) {
  if (a.x != b.x) {
    return a.x < b.x;
  }
  return a.y < b.y;
}

inline std::int64_t Cross(const Point& o, const Point& a, const Point& b) {
  const std::int64_t ax = static_cast<std::int64_t>(a.x) - static_cast<std::int64_t>(o.x);
  const std::int64_t ay = static_cast<std::int64_t>(a.y) - static_cast<std::int64_t>(o.y);
  const std::int64_t bx = static_cast<std::int64_t>(b.x) - static_cast<std::int64_t>(o.x);
  const std::int64_t by = static_cast<std::int64_t>(b.y) - static_cast<std::int64_t>(o.y);
  return (ax * by) - (ay * bx);
}

std::vector<Point> ReferenceHull(std::vector<Point> pts) {
  std::ranges::sort(pts, LessXY);
  pts.erase(std::ranges::unique(pts).begin(), pts.end());

  if (pts.size() <= 1U) {
    return pts;
  }

  std::vector<Point> lower;
  for (const auto& p : pts) {
    while (lower.size() >= 2U && Cross(lower[lower.size() - 2U], lower[lower.size() - 1U], p) <= 0) {
      lower.pop_back();
    }
    lower.push_back(p);
  }

  std::vector<Point> upper;
  for (std::size_t idx = pts.size(); idx > 0; idx--) {
    const auto& p = pts[idx - 1U];
    while (upper.size() >= 2U && Cross(upper[upper.size() - 2U], upper[upper.size() - 1U], p) <= 0) {
      upper.pop_back();
    }
    upper.push_back(p);
  }

  lower.pop_back();
  upper.pop_back();

  std::vector<Point> hull;
  hull.reserve(lower.size() + upper.size());
  hull.insert(hull.end(), lower.begin(), lower.end());
  hull.insert(hull.end(), upper.begin(), upper.end());
  return hull;
}

std::int64_t TwiceArea(const std::vector<Point>& h) {
  if (h.size() < 3U) {
    return 0;
  }
  std::int64_t area2 = 0;
  for (std::size_t i = 0; i < h.size(); i++) {
    const std::size_t j = (i + 1U) % h.size();
    area2 += (static_cast<std::int64_t>(h[i].x) * static_cast<std::int64_t>(h[j].y)) -
             (static_cast<std::int64_t>(h[i].y) * static_cast<std::int64_t>(h[j].x));
  }
  return area2;
}

std::vector<Point> CanonicalHull(std::vector<Point> h) {
  if (h.size() <= 1U) {
    return h;
  }

  if (h.size() == 2U && LessPivot(h[1], h[0])) {
    std::swap(h[0], h[1]);
    return h;
  }

  const auto it = std::ranges::min_element(h, LessPivot);
  std::ranges::rotate(h, it);

  if (TwiceArea(h) < 0) {
    std::ranges::reverse(h.begin() + 1, h.end());
  }

  return h;
}

bool HullEqual(const std::vector<Point>& a, const std::vector<Point>& b) {
  return CanonicalHull(a) == CanonicalHull(b);
}

Input MakeCase(const TestType& params) {
  const int n = std::get<0>(params);
  const std::string label = std::get<1>(params);

  Input in;
  if (label == "empty") {
    return in;
  }
  if (label == "single") {
    in.points = {Point{.x=0, .y=0}};
    return in;
  }
  if (label == "square_with_inside") {
    // Corners + points on edges + inside + duplicates.
    in.points = {Point{.x=0, .y=0}, Point{.x=0, .y=10}, Point{.x=10, .y=10}, Point{.x=10, .y=0},
                 Point{.x=0, .y=0},  Point{.x=10, .y=10},  // duplicates
                 Point{.x=5, .y=0},  Point{.x=0, .y=5}, Point{.x=10, .y=5}, Point{.x=5, .y=10},  // on edges
                 Point{.x=5, .y=5},  Point{.x=6, .y=6}, Point{.x=4, .y=7}};  // inside
    return in;
  }
  if (label == "collinear") {
    in.points = {Point{.x=0, .y=0}, Point{.x=1, .y=0}, Point{.x=2, .y=0}, Point{.x=3, .y=0}, Point{.x=-1, .y=0}, Point{.x=2, .y=0}, Point{.x=0, .y=0}};
    return in;
  }

  // random_n
  // NOLINTNEXTLINE(cert-msc51-cpp)
  std::mt19937 gen(123U + static_cast<unsigned>(n));
  std::uniform_int_distribution<int> dist(-1000, 1000);
  in.points.resize(static_cast<std::size_t>(n));
  for (auto& p : in.points) {
    p.x = dist(gen);
    p.y = dist(gen);
  }
  return in;
}

}  // namespace

class NazarovaKConvHullConstGrahRunFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType& test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    const TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = MakeCase(params);
    expected_ = CanonicalHull(ReferenceHull(input_data_.points));
  }

  bool CheckTestOutputData(OutType& output_data) final {
    return HullEqual(output_data, expected_);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_{};
  OutType expected_;
};

namespace {

TEST_P(NazarovaKConvHullConstGrahRunFuncTests, BuildConvexHull) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 6> kTestParam = {std::make_tuple(0, "empty"),
                                            std::make_tuple(1, "single"),
                                            std::make_tuple(0, "square_with_inside"),
                                            std::make_tuple(0, "collinear"),
                                            std::make_tuple(10, "random_n"),
                                            std::make_tuple(100, "random_n")};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<NazarovaKConvHullConstGrahMPI, InType>(
                                               kTestParam, PPC_SETTINGS_Nazarova_K_Conv_hull_const_Grah),
                                           ppc::util::AddFuncTask<NazarovaKConvHullConstGrahSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_Nazarova_K_Conv_hull_const_Grah));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    NazarovaKConvHullConstGrahRunFuncTests::PrintFuncTestName<NazarovaKConvHullConstGrahRunFuncTests>;

INSTANTIATE_TEST_SUITE_P(ConvHullGrahamTests, NazarovaKConvHullConstGrahRunFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace nazarova_k_conv_hull_const_grah_processes
