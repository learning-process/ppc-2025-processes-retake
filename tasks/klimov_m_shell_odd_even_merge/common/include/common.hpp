#pragma once

#include <vector>
#include <string>

#include "task/include/task.hpp"

namespace klimov_m_shell_odd_even_merge {

using InputType = std::vector<int>;
using OutputType = std::vector<int>;
using TestParam = std::string;  // имя файла с тестовыми данными
using BaseTask = ppc::task::Task<InputType, OutputType>;

}  // namespace klimov_m_shell_odd_even_merge