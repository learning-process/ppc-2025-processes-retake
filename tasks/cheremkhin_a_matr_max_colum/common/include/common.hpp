#pragma once

#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace cheremkhin_a_matr_max_colum {

using InType = std::vector<std::vector<int>>;
using OutType = std::vector<int>;
using TestType = std::tuple<std::vector<std::vector<int>>, std::vector<int>>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace cheremkhin_a_matr_max_colum
