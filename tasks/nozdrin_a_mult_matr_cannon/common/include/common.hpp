#pragma once

#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace nozdrin_a_mult_matr_cannon {

using InType = std::tuple<size_t, std::vector<double>, std::vector<double>>;
using OutType = std::vector<double>;
using TestType = std::tuple<size_t, int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace nozdrin_a_mult_matr_cannon
