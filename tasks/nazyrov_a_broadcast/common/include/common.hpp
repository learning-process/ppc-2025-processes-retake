#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace nazyrov_a_broadcast {

struct BcastInput {
  int root = 0;
  std::vector<int> data;
};

using InType = BcastInput;
using OutType = std::vector<int>;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace nazyrov_a_broadcast
