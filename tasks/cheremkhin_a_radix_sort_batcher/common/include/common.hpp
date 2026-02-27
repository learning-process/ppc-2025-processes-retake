#pragma once

#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace cheremkhin_a_radix_sort_batcher {

using InType = std::vector<int>;
using OutType = std::vector<int>;
using TestType = std::tuple<InType, OutType>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace cheremkhin_a_radix_sort_batcher
