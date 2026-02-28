#include "savva_d_zeidel_method/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "savva_d_zeidel_method/common/include/common.hpp"

namespace savva_d_zeidel_method {

SavvaDZeidelSEQ::SavvaDZeidelSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<double>{};
}

bool SavvaDZeidelSEQ::ValidationImpl() {
  const auto &in = GetInput();

  if (in.n < 0 || in.a.size() != static_cast<size_t>(in.n) * in.n || in.b.size() != static_cast<size_t>(in.n)) {
    return false;
  }

  for (int i = 0; i < in.n; ++i) {
    double sum = 0.0;
    for (int j = 0; j < in.n; ++j) {
      if (i != j) {
        sum += std::abs(in.a[(i * in.n) + j]);
      }
    }
    if (std::abs(in.a[(i * in.n) + i]) <= sum) {
      return false;
    }
  }

  return true;
}

bool SavvaDZeidelSEQ::PreProcessingImpl() {
  GetOutput().assign(GetInput().n, 0.0);
  return true;
}

bool SavvaDZeidelSEQ::RunImpl() {
  const auto &input = GetInput();
  auto &x = GetOutput();

  const int n = input.n;
  if (n == 0) {
    x = std::vector<double>{};
    return true;
  }
  const int max_iter = 10000;
  const double eps = 1e-9;

  for (int iter = 0; iter < max_iter; ++iter) {
    double max_error = 0.0;
    for (int i = 0; i < n; ++i) {
      double sum = 0.0;
      for (int j = 0; j < n; ++j) {
        if (i != j) {
          sum += input.a[(i * n) + j] * x[j];
        }
      }
      double new_xi = (input.b[i] - sum) / input.a[(i * n) + i];
      max_error = std::max(max_error, std::abs(new_xi - x[i]));
      x[i] = new_xi;
    }
    if (max_error < eps) {
      break;
    }
  }

  return true;
}

bool SavvaDZeidelSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace savva_d_zeidel_method
