#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace tsarkov_k_monte_carlo_integration {

using InType = std::vector<int>;
using OutType = double;
using TestType = std::tuple<InType, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace tsarkov_k_monte_carlo_integration
