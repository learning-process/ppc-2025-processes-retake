#pragma once

#include <string>
#include <utility>

#include "task/include/task.hpp"

namespace kulikov_d_coun_number_char {

using InType = std::pair<std::string, std::string>;
using OutType = int;
using TestType = std::pair<std::string, int>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace kulikov_d_coun_number_char
