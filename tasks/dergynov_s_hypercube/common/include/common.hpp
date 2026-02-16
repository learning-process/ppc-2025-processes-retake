#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace dergynov_s_hypercube {

using InType = std::vector<int>;  // {source, dest, data_size}
using OutType = int;              // сумма элементов
using TestType = std::tuple<std::vector<int>, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace dergynov_s_hypercube
