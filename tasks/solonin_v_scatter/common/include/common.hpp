#pragma once
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace solonin_v_scatter {

using InType = std::tuple<std::vector<int>, int, int>;
using OutType = std::vector<int>;
using TestType = std::tuple<int, std::vector<int>, int, int>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace solonin_v_scatter
