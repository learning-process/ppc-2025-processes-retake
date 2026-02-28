#pragma once

#include <string>
#include <vector>

#include "task/include/task.hpp"

namespace savva_d_zeidel_method {

struct SeidelInput {
  int n = 0;
  std::vector<double> a;
  std::vector<double> b;
};

using InType = SeidelInput;
using OutType = std::vector<double>;

struct TestParams {
  SeidelInput in{};
  OutType out;
  std::string name;
};

// using TestType = std::tuple<SeidelInput, OutType, std::string>;
using TestType = TestParams;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace savva_d_zeidel_method
