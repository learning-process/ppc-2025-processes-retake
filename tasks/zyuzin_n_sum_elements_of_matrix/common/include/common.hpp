#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace zyuzin_n_sum_elements_of_matrix {

using InType = std::tuple<int, int, std::vector<double>>;
using OutType = double;
using TestType = std::string;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace zyuzin_n_sum_elements_of_matrix
