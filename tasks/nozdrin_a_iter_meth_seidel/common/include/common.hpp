#pragma once

#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace nozdrin_a_iter_meth_seidel {

using InType = std::tuple<std::size_t, std::vector<double>, std::vector<double>, double>;
using OutType = std::vector<double>;
using TestType = std::tuple<int, std::string, double>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace nozdrin_a_iter_meth_seidel
