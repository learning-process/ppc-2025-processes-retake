#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace dilshodov_a_convex_hull {

struct Point {
  int x;
  int y;
};

inline bool operator==(const Point &a, const Point &b) {
  return a.x == b.x && a.y == b.y;
}

using InType = std::vector<int>;
using OutType = std::vector<Point>;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace dilshodov_a_convex_hull
