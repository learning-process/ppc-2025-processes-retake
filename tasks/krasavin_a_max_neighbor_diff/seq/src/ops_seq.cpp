#include "krasavin_a_max_neighbor_diff/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>
#include "krasavin_a_max_neighbor_diff/common/include/common.hpp"

namespace krasavin_a_max_neighbor_diff {

KrasavinAMaxNeighborDiffSEQ::KrasavinAMaxNeighborDiffSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool KrasavinAMaxNeighborDiffSEQ::ValidationImpl() {
  return true;
}

bool KrasavinAMaxNeighborDiffSEQ::PreProcessingImpl() {
  return true;
}

bool KrasavinAMaxNeighborDiffSEQ::RunImpl() {
  const std::vector<int> &vec = GetInput();
  std::size_t n = vec.size();
  if (n < 2) {
    GetOutput() = 0;
    return true;
  }

  int max_diff = 0;
  for (std::size_t i = 0; i < n - 1; i++) {
    int diff = std::abs(vec[i + 1] - vec[i]);
    max_diff = std::max(diff, max_diff);
  }

  GetOutput() = max_diff;
  return true;
}

bool KrasavinAMaxNeighborDiffSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace krasavin_a_max_neighbor_diff