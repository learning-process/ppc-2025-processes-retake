#pragma once

#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace marov_radix_sort_double {

using InType = std::vector<double>;
using OutType = std::vector<double>;
using TestType = std::tuple<std::vector<double>, std::vector<double>>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace marov_radix_sort_double
