#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace k_sokolov_min_val_matrix {

using InType = int;
using OutType = int;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace k_sokolov_min_val_matrix
