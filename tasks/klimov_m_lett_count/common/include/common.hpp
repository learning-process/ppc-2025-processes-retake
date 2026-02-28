#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace klimov_m_lett_count {

using InputType = std::string;

using OutputType = int;

using TestParam = std::tuple<std::tuple<std::string, int>, std::string>;

using BaseTask = ppc::task::Task<InputType, OutputType>;

}  // namespace klimov_m_lett_count
