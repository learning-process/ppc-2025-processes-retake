#include "yushkova_p_min_in_matrix/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "yushkova_p_min_in_matrix/common/include/common.hpp"

namespace yushkova_p_min_in_matrix {

namespace {

inline int GenerateValue(int64_t i, int64_t j) {
  constexpr int64_t kA = 1103515245LL;
  constexpr int64_t kC = 12345LL;
  constexpr int64_t kM = 2147483648LL;

  int64_t seed = ((i % kM) * (100000007LL % kM) + (j % kM) * (1000000009LL % kM)) % kM;
  seed = (seed ^ 42LL) % kM;
  int64_t val = ((kA % kM) * (seed % kM) + kC) % kM;

  return static_cast<int>((val % 2000001LL) - 1000000LL);
}

}  // namespace

YushkovaPMinInMatrixSEQ::YushkovaPMinInMatrixSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool YushkovaPMinInMatrixSEQ::ValidationImpl() {
  return GetInput() > 0 && GetOutput().empty();
}

bool YushkovaPMinInMatrixSEQ::PreProcessingImpl() {
  auto &out = GetOutput();
  out.clear();
  out.reserve(GetInput());
  return true;
}

bool YushkovaPMinInMatrixSEQ::RunImpl() {
  const InType n = GetInput();
  if (n <= 0) {
    return false;
  }

  auto &out = GetOutput();
  out.clear();
  out.reserve(n);

  for (InType row = 0; row < n; ++row) {
    InType row_min = GenerateValue(static_cast<int64_t>(row), 0);

    for (InType col = 1; col < n; ++col) {
      const InType val = GenerateValue(static_cast<int64_t>(row), static_cast<int64_t>(col));
      row_min = std::min(row_min, val);
    }

    out.push_back(row_min);
  }

  return out.size() == static_cast<std::size_t>(n);
}

bool YushkovaPMinInMatrixSEQ::PostProcessingImpl() {
  return GetOutput().size() == static_cast<std::size_t>(GetInput());
}

}  // namespace yushkova_p_min_in_matrix
