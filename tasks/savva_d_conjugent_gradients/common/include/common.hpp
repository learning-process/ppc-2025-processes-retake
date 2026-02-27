#pragma once

#include <string>
#include <vector>

#include "task/include/task.hpp"

namespace savva_d_conjugent_gradients {

struct InputSystem {
  int n = 0;
  std::vector<double> a;
  std::vector<double> b;
};

using InType = InputSystem;
using OutType = std::vector<double>;

struct TestParams {
  InputSystem in{};
  OutType out;
  std::string name;
};

using TestType = TestParams;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace savva_d_conjugent_gradients
