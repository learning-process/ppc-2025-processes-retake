#pragma once
#include <tuple>
#include <vector>
#include "task/include/task.hpp"

namespace solonin_v_scatter {

// InType: (send_buf, send_count, root)
// OutType: recv_buf (each process receives send_count elements)
using InType = std::tuple<std::vector<int>, int, int>;
using OutType = std::vector<int>;
using TestType = std::tuple<int, std::vector<int>, int, int>;  // (id, data, send_count, root)
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace solonin_v_scatter
