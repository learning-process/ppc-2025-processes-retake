#pragma once

#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace nazarova_k_conv_hull_const_grah_processes {

struct Point {
  int x = 0;
  int y = 0;
};

inline bool operator==(const Point &a, const Point &b) {
  return a.x == b.x && a.y == b.y;
}

inline bool operator!=(const Point &a, const Point &b) {
  return !(a == b);
}

struct Input {
  std::vector<Point> points;
};

using InType = Input;
using OutType = std::vector<Point>;             // convex hull in CCW order, without repeating the first point
using TestType = std::tuple<int, std::string>;  // (n, label) for gtest naming
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace nazarova_k_conv_hull_const_grah_processes
