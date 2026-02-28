#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace sokolov_k_shell_simple_merge {

using InType = int;
using OutType = int;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace sokolov_k_shell_simple_merge
