#pragma once

#include <string>
#include <tuple>
#include <utility>

#include "task/include/task.hpp"

namespace tsarkov_k_lexicographic_string_compare {

using InType = std::pair<std::string, std::string>;
using OutType = int;
using TestType = std::tuple<std::string, std::string>;

using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace tsarkov_k_lexicographic_string_compare
