#pragma once

#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace nazarova_k_char_count_processes {

struct Input {
  std::vector<char> text;
  char target = '\0';
};

using InType = Input;
using OutType = int;
using TestType = std::tuple<int, char, bool>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace nazarova_k_char_count_processes
