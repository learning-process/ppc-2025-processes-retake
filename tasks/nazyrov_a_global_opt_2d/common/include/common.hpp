#pragma once

#include <cmath>
#include <functional>
#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace nazyrov_a_global_opt_2d {

struct OptInput {
  std::function<double(double, double)> func;

  double x_min{0.0};
  double x_max{1.0};
  double y_min{0.0};
  double y_max{1.0};

  double epsilon{0.01};
  double r_param{2.0};
  int max_iterations{1000};

  OptInput() : func(nullptr) {}
};

struct OptResult {
  double x_opt{0.0};
  double y_opt{0.0};
  double func_min{0.0};
  int iterations{0};
  bool converged{false};

  OptResult() = default;

  bool operator==(const OptResult &other) const {
    constexpr double kTol = 1e-2;
    return std::abs(x_opt - other.x_opt) < kTol && std::abs(y_opt - other.y_opt) < kTol &&
           std::abs(func_min - other.func_min) < kTol;
  }
};

using InType = OptInput;
using OutType = OptResult;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace nazyrov_a_global_opt_2d
