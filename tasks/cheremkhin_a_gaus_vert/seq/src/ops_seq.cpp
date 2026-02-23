#include "cheremkhin_a_gaus_vert/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "cheremkhin_a_gaus_vert/common/include/common.hpp"

namespace cheremkhin_a_gaus_vert {

namespace {

constexpr double kEps = 1e-12;

inline double &At(std::vector<double> &a, int n, int r, int c) {
  return a[(static_cast<std::size_t>(r) * static_cast<std::size_t>(n)) + static_cast<std::size_t>(c)];
}

}  // namespace

CheremkhinAGausVertSEQ::CheremkhinAGausVertSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool CheremkhinAGausVertSEQ::ValidationImpl() {
  const auto &in = GetInput();
  if (in.n <= 0) {
    return false;
  }
  const auto n = static_cast<std::size_t>(in.n);
  return in.a.size() == n * n && in.b.size() == n;
}

bool CheremkhinAGausVertSEQ::PreProcessingImpl() {
  return true;
}

bool CheremkhinAGausVertSEQ::RunImpl() {
  const auto &in = GetInput();
  const int n = in.n;

  std::vector<double> a = in.a;
  std::vector<double> b = in.b;
  std::vector<double> x(static_cast<std::size_t>(n), 0.0);

  for (int k = 0; k < n; ++k) {
    int pivot_row = k;
    double best = std::abs(At(a, n, k, k));
    for (int row = k + 1; row < n; ++row) {
      const double v = std::abs(At(a, n, row, k));
      if (v > best) {
        best = v;
        pivot_row = row;
      }
    }

    if (best < kEps) {
      return false;
    }

    if (pivot_row != k) {
      for (int col = 0; col < n; ++col) {
        std::swap(At(a, n, k, col), At(a, n, pivot_row, col));
      }
      std::swap(b[static_cast<std::size_t>(k)], b[static_cast<std::size_t>(pivot_row)]);
    }

    const double pivot = At(a, n, k, k);
    for (int row = k + 1; row < n; ++row) {
      const double m = At(a, n, row, k) / pivot;
      At(a, n, row, k) = 0.0;
      for (int col = k + 1; col < n; ++col) {
        At(a, n, row, col) -= m * At(a, n, k, col);
      }
      b[static_cast<std::size_t>(row)] -= m * b[static_cast<std::size_t>(k)];
    }
  }

  for (int i = n - 1; i >= 0; --i) {
    double sum = b[static_cast<std::size_t>(i)];
    for (int j = i + 1; j < n; ++j) {
      sum -= At(a, n, i, j) * x[static_cast<std::size_t>(j)];
    }
    const double diag = At(a, n, i, i);
    if (std::abs(diag) < kEps) {
      return false;
    }
    x[static_cast<std::size_t>(i)] = sum / diag;
  }

  GetOutput() = x;
  return true;
}

bool CheremkhinAGausVertSEQ::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace cheremkhin_a_gaus_vert
