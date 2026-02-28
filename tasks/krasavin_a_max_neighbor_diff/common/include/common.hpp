#pragma once

#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace krasavin_a_max_neighbor_diff {

using InType = std::vector<int>;
using OutType = int;
using TestType = std::tuple<std::vector<int>, int>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace krasavin_a_max_neighbor_diff
