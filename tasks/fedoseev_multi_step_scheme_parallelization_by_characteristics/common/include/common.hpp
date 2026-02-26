#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace fedoseev_multi_step_scheme_parallelization_by_characteristics {

using InType = int;
using OutType = double;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace fedoseev_multi_step_scheme_parallelization_by_characteristics
