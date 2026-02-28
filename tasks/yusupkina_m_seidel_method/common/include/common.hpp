#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace yusupkina_m_seidel_method {
struct InputData {
  std::vector<double> matrix;      
  std::vector<double> rhs;         
  int n;                           
};

using InType = InputData;
using OutType = std::vector<double>;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace yusupkina_m_seidel_method
