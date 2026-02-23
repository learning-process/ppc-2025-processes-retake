#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace muhammadkhon_i_batcher_sort {

using InType = std::vector<int>;
using OutType = std::vector<int>;
using TestType = std::tuple<InType, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace muhammadkhon_i_batcher_sort
