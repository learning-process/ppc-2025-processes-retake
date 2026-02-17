#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace kichanova_k_shellsort_batcher {

using InType = int;
using OutType = int;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace kichanova_k_shellsort_batcher