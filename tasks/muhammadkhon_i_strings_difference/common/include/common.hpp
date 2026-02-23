#pragma once

#include <string>
#include <utility>

#include "task/include/task.hpp"

namespace muhammadkhon_i_strings_difference {

using InType = std::pair<std::string, std::string>;
using OutType = int;
using TestType = std::pair<std::string, int>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace muhammadkhon_i_strings_difference
