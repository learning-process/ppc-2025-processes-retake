#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace vinyaikina_e_vert_ribbon_scheme {

using InType = int;
using OutType = int;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace vinyaikina_e_vert_ribbon_scheme
