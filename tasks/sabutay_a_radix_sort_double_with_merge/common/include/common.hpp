#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace sabutay_a_radix_sort_double_with_merge {

using InType = std::vector<double>;
using OutType = std::vector<double>;
using TestType = std::tuple<std::vector<double>, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace sabutay_a_radix_sort_double_with_merge
