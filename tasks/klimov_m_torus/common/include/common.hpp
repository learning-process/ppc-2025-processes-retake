#pragma once

#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace klimov_m_torus {

struct TransferRequest {
  int sender;                
  int receiver;            
  std::vector<int> data;     
};

struct TransferResult {
  std::vector<int> received_data;  
  std::vector<int> route;        
};

using InType = TransferRequest;
using OutType = TransferResult;
using TestParam = std::tuple<int>; 
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace klimov_m_torus