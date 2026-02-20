#include "sabutay_vector_sign_changes/seq/include/ops_seq.hpp"

#include <cstddef>

#include "sabutay_vector_sign_changes/common/include/common.hpp"

namespace sabutay_vector_sign_changes {

namespace {

inline int Sign(double x) {
  if (x > 0.0) {
    return 1;
  }
  if (x < 0.0) {
    return -1;
  }
  return 0;
}

inline double GetElement(std::size_t i) {
  if (i % 5 == 0) {
    return 0.0;
  }
  if (i % 2 == 0) {
    return 1.0;
  }
  return -1.0;
}

}  // namespace

SabutayVectorSignChangesSEQ::SabutayVectorSignChangesSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool SabutayVectorSignChangesSEQ::ValidationImpl() {
  return GetInput() >= 0;
}

bool SabutayVectorSignChangesSEQ::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool SabutayVectorSignChangesSEQ::RunImpl() {
  const InType n = GetInput();
  if (n <= 0) {
    GetOutput() = 0;
    return true;
  }

  int prev = 0;
  int cnt = 0;

  for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) {
    const int s = Sign(GetElement(i));
    if (s == 0) {
      continue;
    }
    if (prev != 0 && s != prev) {
      ++cnt;
    }
    prev = s;
  }

  GetOutput() = cnt;
  return true;
}

bool SabutayVectorSignChangesSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace sabutay_vector_sign_changes
