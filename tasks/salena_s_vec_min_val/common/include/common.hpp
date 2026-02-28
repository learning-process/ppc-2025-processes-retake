#pragma once
#include <vector>
#include "task/include/task.hpp"

namespace salena_s_vec_min_val {

using InType = std::vector<int>;
using OutType = int;


using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace salena_s_vec_min_val