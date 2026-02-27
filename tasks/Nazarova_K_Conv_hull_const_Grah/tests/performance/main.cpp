#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <ranges>
#include <utility>
#include <vector>

#include "Nazarova_K_Conv_hull_const_Grah/common/include/common.hpp"
#include "Nazarova_K_Conv_hull_const_Grah/mpi/include/ops_mpi.hpp"
#include "Nazarova_K_Conv_hull_const_Grah/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace nazarova_k_conv_hull_const_grah_processes {
namespace {

inline bool LessPivot(const Point& a, const Point& b) {
  if (a.y != b.y) {
    return a.y < b.y;
  }
  return a.x < b.x;
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

}  // namespace

class NazarovaKConvHullConstGrahRunPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {

  static constexpr int kN = 1000000;
  InType input_data_{};
  OutType expected_;

  void SetUp() override {

    std::mt19937 gen(777U);

    static constexpr int kR = 1000000;
    std::uniform_int_distribution<int> dist(1, kR - 1);

    input_data_.points.resize(static_cast<std::size_t>(kN));
    input_data_.points[0] = Point{.x = 0, .y = 0};
    input_data_.points[1] = Point{.x = kR, .y = 0};
    input_data_.points[2] = Point{.x = kR, .y = kR};
    input_data_.points[3] = Point{.x = 0, .y = kR};

    for (std::size_t i = 4; i < input_data_.points.size(); i++) {
      input_data_.points[i] = Point{.x = dist(gen), .y = dist(gen)};
    }

    expected_ = CanonicalHull(std::vector<Point>{Point{.x = 0, .y = 0}, Point{.x = kR, .y = 0}, Point{.x = kR, .y = kR},
                                                Point{.x = 0, .y = kR}});
  }

  bool CheckTestOutputData(OutType& output_data) final {
    return HullEqual(output_data, expected_);
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(NazarovaKConvHullConstGrahRunPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, NazarovaKConvHullConstGrahMPI, NazarovaKConvHullConstGrahSEQ>(
        PPC_SETTINGS_Nazarova_K_Conv_hull_const_Grah);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = NazarovaKConvHullConstGrahRunPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, NazarovaKConvHullConstGrahRunPerfTests, kGtestValues, kPerfTestName);

}  // namespace nazarova_k_conv_hull_const_grah_processes
