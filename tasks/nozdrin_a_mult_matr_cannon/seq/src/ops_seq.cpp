#include "nozdrin_a_mult_matr_cannon/seq/include/ops_seq.hpp"

#include <cstddef>
#include <vector>

#include "nozdrin_a_mult_matr_cannon/common/include/common.hpp"

namespace nozdrin_a_mult_matr_cannon {

NozdrinAMultMatrCannonSEQ::NozdrinAMultMatrCannonSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool NozdrinAMultMatrCannonSEQ::ValidationImpl() {
  const size_t n = std::get<0>(GetInput());
  const auto &a = std::get<1>(GetInput());
  const auto &b = std::get<2>(GetInput());

  return n > 0 && a.size() == n * n && b.size() == n * n;
}

bool NozdrinAMultMatrCannonSEQ::PreProcessingImpl() {
  GetOutput() = {};
  return true;
}

bool NozdrinAMultMatrCannonSEQ::RunImpl() {
  const auto n = std::get<0>(GetInput());
  const auto &a = std::get<1>(GetInput());
  const auto &b = std::get<2>(GetInput());
  std::vector<double> c(n * n, 0.0);

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      double sum = 0.0;
      for (size_t k = 0; k < n; ++k) {
        sum += a[(i * n) + k] * b[(k * n) + j];
      }
      c[(i * n) + j] = sum;
    }
  }

  GetOutput() = c;
  return true;
}

bool NozdrinAMultMatrCannonSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace nozdrin_a_mult_matr_cannon
