#pragma once

#include <cstdint>
#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace yushkova_p_hypercube {

using InType = int;
using OutType = std::uint64_t;
using TestType = std::tuple<InType, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace yushkova_p_hypercube
