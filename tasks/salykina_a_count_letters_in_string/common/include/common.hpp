#pragma once
#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace salykina_a_count_letters_in_string {

using InType = std::string;
using OutType = int;
using TestType = std::tuple<std::string, int>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace salykina_a_count_letters_in_string
