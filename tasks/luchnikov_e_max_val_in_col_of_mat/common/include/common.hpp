#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace luchnikov_e_max_val_in_col_of_mat {

using InType = std::vector<std::vector<int>>;
using OutType = std::vector<int>;
using TestType = std::tuple<int, int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace luchnikov_e_max_val_in_col_of_mat
