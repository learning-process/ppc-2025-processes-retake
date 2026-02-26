#pragma once

#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace fedoseev_gaussian_method_horizontal_strip_scheme {

using InType = std::vector<std::vector<double>>;
using OutType = std::vector<double>;
using TestType = std::tuple<int, int>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace fedoseev_gaussian_method_horizontal_strip_scheme
