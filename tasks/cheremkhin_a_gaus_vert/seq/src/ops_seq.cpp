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

int FindPivotRow(const std::vector<double> &a, int n, int k, double &best_abs) {
  int pivot_row = k;
  best_abs = std::abs(a[(static_cast<std::size_t>(k) * static_cast<std::size_t>(n)) + static_cast<std::size_t>(k)]);
  for (int row = k + 1; row < n; ++row) {
    const double v =
        std::abs(a[(static_cast<std::size_t>(row) * static_cast<std::size_t>(n)) + static_cast<std::size_t>(k)]);
    if (v > best_abs) {
      best_abs = v;
      pivot_row = row;
    }
  }
  return pivot_row;
}

void SwapRows(std::vector<double> &a, std::vector<double> &b, int n, int r1, int r2) {
  if (r1 == r2) {
    return;
  }
  for (int col = 0; col < n; ++col) {
    std::swap(At(a, n, r1, col), At(a, n, r2, col));
  }
  std::swap(b[static_cast<std::size_t>(r1)], b[static_cast<std::size_t>(r2)]);
}

bool ApplyPivoting(std::vector<double> &a, std::vector<double> &b, int n, int k) {
  double best_abs = 0.0;
  const int pivot_row = FindPivotRow(a, n, k, best_abs);
  if (best_abs < kEps) {
    return false;
  }
  SwapRows(a, b, n, k, pivot_row);
  return true;
}

void EliminateBelowPivot(std::vector<double> &a, std::vector<double> &b, int n, int k) {
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

bool ForwardElimination(std::vector<double> &a, std::vector<double> &b, int n) {
  for (int k = 0; k < n; ++k) {
    if (!ApplyPivoting(a, b, n, k)) {
      return false;
    }
    EliminateBelowPivot(a, b, n, k);
  }
  return true;
}

bool BackSubstitution(const std::vector<double> &a, const std::vector<double> &b, int n, std::vector<double> &x) {
  for (int i = n - 1; i >= 0; --i) {
    double sum = b[static_cast<std::size_t>(i)];
    for (int j = i + 1; j < n; ++j) {
      sum -= a[(static_cast<std::size_t>(i) * static_cast<std::size_t>(n)) + static_cast<std::size_t>(j)] *
             x[static_cast<std::size_t>(j)];
    }
    const double diag = a[(static_cast<std::size_t>(i) * static_cast<std::size_t>(n)) + static_cast<std::size_t>(i)];
    if (std::abs(diag) < kEps) {
      return false;
    }
    x[static_cast<std::size_t>(i)] = sum / diag;
  }
  return true;
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

  if (!ForwardElimination(a, b, n)) {
    return false;
  }

  if (!BackSubstitution(a, b, n, x)) {
    return false;
  }

  GetOutput() = x;
  return true;
}

bool CheremkhinAGausVertSEQ::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace cheremkhin_a_gaus_vert
