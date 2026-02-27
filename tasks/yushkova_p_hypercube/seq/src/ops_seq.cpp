#include "yushkova_p_hypercube/seq/include/ops_seq.hpp"

#include <cstdint>

#include "yushkova_p_hypercube/common/include/common.hpp"

namespace yushkova_p_hypercube {

YushkovaPHypercubeSEQ::YushkovaPHypercubeSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool YushkovaPHypercubeSEQ::ValidationImpl() {
  const InType n = GetInput();
  return n > 0 && n < 63;
}

bool YushkovaPHypercubeSEQ::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool YushkovaPHypercubeSEQ::RunImpl() {
  const InType n = GetInput();
  const std::uint64_t vertices = static_cast<std::uint64_t>(1) << n;
  std::uint64_t edges = 0;

  for (std::uint64_t vertex = 0; vertex < vertices; ++vertex) {
    for (int bit = 0; bit < n; ++bit) {
      const std::uint64_t neighbor = vertex ^ (static_cast<std::uint64_t>(1) << bit);
      if (vertex < neighbor) {
        ++edges;
      }
    }
  }

  GetOutput() = static_cast<OutType>(edges);
  return true;
}

bool YushkovaPHypercubeSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace yushkova_p_hypercube
