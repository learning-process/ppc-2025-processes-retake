#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace marov_count_letters {

using InType = std::string;
using OutType = int;
using TestType = std::tuple<std::string, int>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace marov_count_letters
