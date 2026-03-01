#include "solonin_v_col_min_matrix/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "solonin_v_col_min_matrix/common/include/common.hpp"

namespace solonin_v_col_min_matrix {

namespace {
inline InType Generate(int64_t i, int64_t j) {
  uint64_t seed = (i * 100000007ULL + j * 1000000009ULL) ^ 42ULL;
  seed ^= seed >> 12;
  seed ^= seed << 25;
  seed ^= seed >> 27;
  uint64_t value = seed * 0x2545F4914F6CDD1DULL;
  return static_cast<InType>((value % 2000001) - 1000000);
}
}  // namespace

SoloninVMinMatrixSEQ::SoloninVMinMatrixSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool SoloninVMinMatrixSEQ::ValidationImpl() { return (GetInput() > 0) && (GetOutput().empty()); }

bool SoloninVMinMatrixSEQ::PreProcessingImpl() {
  GetOutput().clear();
  GetOutput().reserve(GetInput());
  return true;
}

bool SoloninVMinMatrixSEQ::RunImpl() {
  InType n = GetInput();
  if (n == 0) {
    return false;
  }
  GetOutput().clear();
  GetOutput().reserve(n);
  for (InType j = 0; j < n; j++) {
    InType min_val = Generate(static_cast<int64_t>(0), static_cast<int64_t>(j));
    for (InType i = 1; i < n; i++) {
      InType val = Generate(static_cast<int64_t>(i), static_cast<int64_t>(j));
      min_val = std::min(min_val, val);
    }
    GetOutput().push_back(min_val);
  }
  return !GetOutput().empty() && (GetOutput().size() == static_cast<size_t>(n));
}

bool SoloninVMinMatrixSEQ::PostProcessingImpl() {
  return !GetOutput().empty() && (GetOutput().size() == static_cast<size_t>(GetInput()));
}

}  // namespace solonin_v_col_min_matrix
