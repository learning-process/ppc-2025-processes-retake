#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace yushkova_p_hypercube {

using InType = std::tuple<int, int, int>;
using OutType = int;
using TestType = std::tuple<InType, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace yushkova_p_hypercube
