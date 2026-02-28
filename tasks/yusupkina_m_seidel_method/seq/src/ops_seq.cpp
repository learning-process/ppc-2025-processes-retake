#include "yusupkina_m_seidel_method/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "yusupkina_m_seidel_method/common/include/common.hpp"

namespace yusupkina_m_seidel_method {

YusupkinaMSeidelMethodSEQ::YusupkinaMSeidelMethodSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}

bool YusupkinaMSeidelMethodSEQ::ValidationImpl() {
  const auto &in = GetInput();
  if (in.n <= 0) {
    return false;
  }

  if (in.matrix.size() != static_cast<size_t>(in.n) * in.n) {
    return false;
  }
  if (in.rhs.size() != static_cast<size_t>(in.n)) {
    return false;
  }

  for (int i = 0; i < in.n; i++) {
    double sum = 0.0;
    for (int j = 0; j < in.n; j++) {
      if (i != j) {
        sum += std::abs(in.matrix[(i * in.n) + j]);
      }
    }
    if (std::abs(in.matrix[(i * in.n) + i]) <= sum) {
      return false;
    }
  }
  return true;
}

bool YusupkinaMSeidelMethodSEQ::PreProcessingImpl() {
  auto &out = GetOutput();
  const auto &in = GetInput();
  out.assign(in.n, 0.0);
  return true;
}

bool YusupkinaMSeidelMethodSEQ::RunImpl() {
  const auto &in = GetInput();
  auto &x = GetOutput();
  const int n = in.n;

  if (n == 0) {
    x.clear();
    return true;
  }
  const int max_iter = 1000;
  const double eps = 1e-6;
  const auto &a = in.matrix;
  const auto &b = in.rhs;

  for (int iter = 0; iter < max_iter; iter++) {
    double max_error = 0.0;
    for (int i = 0; i < n; i++) {
      double sum = 0.0;
      for (int j = 0; j < n; j++) {
        if (i != j) {
          sum += a[(i * n) + j] * x[j];
        }
      }

      double new_xi = (b[i] - sum) / a[(i * n) + i];
      double error = std::abs(new_xi - x[i]);
      max_error = std::max(max_error, error);
      x[i] = new_xi;
    }
    if (max_error < eps) {
      break;
    }
  }

  return true;
}

bool YusupkinaMSeidelMethodSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace yusupkina_m_seidel_method
