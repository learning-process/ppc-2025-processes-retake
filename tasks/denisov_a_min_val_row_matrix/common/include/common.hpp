#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace denisov_a_min_val_row_matrix {

using InType = std::vector<std::vector<int>>;
using OutType = std::vector<int>;
using TestType = std::tuple<int, int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace denisov_a_min_val_row_matrix
