#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace luchnikov_e_gener_transm_from_all_to_one_gather {

using InType = int;
using OutType = int;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace luchnikov_e_gener_transm_from_all_to_one_gather
