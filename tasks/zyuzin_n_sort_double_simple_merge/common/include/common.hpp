#pragma once

#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace zyuzin_n_sort_double_simple_merge {

using InType = std::vector<double>;
using OutType = std::vector<double>;
using TestType = std::tuple<int, std::vector<double>, std::vector<double>>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace zyuzin_n_sort_double_simple_merge
