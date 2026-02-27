#pragma once

#include <tuple>
#include <utility>
#include <vector>

#include "task/include/task.hpp"

namespace zyuzin_n_multiplication_matrix_horiz {

using InType = std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>;
using OutType = std::vector<std::vector<double>>;
using TestType = std::tuple<int, std::vector<std::vector<double>>, std::vector<std::vector<double>>,
                            std::vector<std::vector<double>>>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace zyuzin_n_multiplication_matrix_horiz
