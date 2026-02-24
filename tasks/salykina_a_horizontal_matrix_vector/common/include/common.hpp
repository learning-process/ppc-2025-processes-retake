#pragma once

#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace salykina_a_horizontal_matrix_vector {

using InType = std::tuple<std::vector<double>, int, int, std::vector<double>>;
using OutType = std::vector<double>;
using TestType = std::tuple<int, int>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace salykina_a_horizontal_matrix_vector
