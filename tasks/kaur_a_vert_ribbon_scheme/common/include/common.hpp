#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace kaur_a_vert_ribbon_scheme {

struct TaskData {
  std::vector<double> matrix;
  std::vector<double> vector;
  int rows{0};
  int cols{0};
};

using InType = TaskData;
using OutType = std::vector<double>;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace kaur_a_vert_ribbon_scheme