#include "nazyrov_a_broadcast/seq/include/ops_seq.hpp"

#include <utility>

#include "nazyrov_a_broadcast/common/include/common.hpp"

namespace nazyrov_a_broadcast {

BroadcastSEQ::BroadcastSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  InType copy = in;
  GetInput() = std::move(copy);
  GetOutput() = {};
}

bool BroadcastSEQ::ValidationImpl() {
  const auto &input = GetInput();
  return input.root >= 0 && !input.data.empty();
}

bool BroadcastSEQ::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

bool BroadcastSEQ::RunImpl() {
  GetOutput() = GetInput().data;
  return true;
}

bool BroadcastSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace nazyrov_a_broadcast
