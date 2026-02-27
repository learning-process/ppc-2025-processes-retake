#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace sabutay_vector_sign_changes {

using InType = double;
using OutType = int;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace sabutay_vector_sign_changes
