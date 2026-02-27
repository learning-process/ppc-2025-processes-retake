#pragma once

#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace luchnikov_e_gener_transm_from_all_to_one_gather {

using InType = std::vector<int>;
using OutType = std::vector<int>;
using TestType = std::tuple<size_t, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace luchnikov_e_gener_transm_from_all_to_one_gather
