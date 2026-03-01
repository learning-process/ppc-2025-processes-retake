#pragma once

#include "task/include/task.hpp"

#include <string>
#include <tuple>
#include <vector>

namespace solonin_v_col_min_matrix {

using InType = int;
using OutType = std::vector<int>;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace solonin_v_col_min_matrix
