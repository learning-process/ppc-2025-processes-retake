#pragma once

#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace klimov_m_torus {

struct TransferRequest {
  int source;             
  int dest;           
  std::vector<int> payload; 
};

struct TransferResult {
  std::vector<int> payload; 
  std::vector<int> path; 
};

using InType = TransferRequest;
using OutType = TransferResult;
using TestParam = std::tuple<int>; 
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace klimov_m_torus