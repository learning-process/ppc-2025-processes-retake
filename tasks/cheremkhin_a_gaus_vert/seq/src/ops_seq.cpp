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

inline std::size_t Idx(std::size_t n, int r, int c) {
  return (static_cast<std::size_t>(r) * n) + static_cast<std::size_t>(c);
}

inline double *RowPtr(std::vector<double> &a, std::size_t n, int r) {
  return a.data() + (static_cast<std::size_t>(r) * n);
}

inline int FindPivotRow(const std::vector<double> &a, int n, int k, double &best_abs) {
  const auto sn = static_cast<std::size_t>(n);
  int pivot_row = k;
  best_abs = std::abs(a[Idx(sn, k, k)]);
  for (int row = k + 1; row < n; ++row) {
    const double v = std::abs(a[Idx(sn, row, k)]);
    if (v > best_abs) {
      best_abs = v;
      pivot_row = row;
    }
  }
  return pivot_row;
}

inline void SwapRows(std::vector<double> &a, std::vector<double> &b, int n, int r1, int r2) {
  if (r1 == r2) {
    return;
  }
  const auto sn = static_cast<std::size_t>(n);
  auto *row1 = RowPtr(a, sn, r1);
  auto *row2 = RowPtr(a, sn, r2);
  std::swap_ranges(row1, row1 + sn, row2);
  std::swap(b[static_cast<std::size_t>(r1)], b[static_cast<std::size_t>(r2)]);
}

inline bool ApplyPivoting(std::vector<double> &a, std::vector<double> &b, int n, int k) {
  double best_abs = 0.0;
  const int pivot_row = FindPivotRow(a, n, k, best_abs);
  if (best_abs < kEps) {
    return false;
  }
  SwapRows(a, b, n, k, pivot_row);
  return true;
}

inline void EliminateBelowPivot(std::vector<double> &a, std::vector<double> &b, int n, int k) {
  const auto sn = static_cast<std::size_t>(n);
  auto *row_k = RowPtr(a, sn, k);
  const double pivot = row_k[k];
  const double inv_pivot = 1.0 / pivot;
  for (int row = k + 1; row < n; ++row) {
    auto *row_r = RowPtr(a, sn, row);
    const double m = row_r[k] * inv_pivot;
    row_r[k] = 0.0;
    auto *dst = row_r + (k + 1);
    const auto *src = row_k + (k + 1);
    for (int col = k + 1; col < n; ++col) {
      dst[col - (k + 1)] -= m * src[col - (k + 1)];
    }
    b[static_cast<std::size_t>(row)] -= m * b[static_cast<std::size_t>(k)];
  }
}

inline bool ForwardElimination(std::vector<double> &a, std::vector<double> &b, int n) {
  for (int k = 0; k < n; ++k) {
    if (!ApplyPivoting(a, b, n, k)) {
      return false;
    }
    EliminateBelowPivot(a, b, n, k);
  }
  return true;
}

inline bool BackSubstitution(const std::vector<double> &a, const std::vector<double> &b, int n,
                             std::vector<double> &x) {
  const auto sn = static_cast<std::size_t>(n);
  const auto *adata = a.data();
  const auto *bdata = b.data();
  auto *xdata = x.data();
  for (int i = n - 1; i >= 0; --i) {
    const auto si = static_cast<std::size_t>(i);
    const auto *row_i = adata + (si * sn);
    double sum = bdata[si];
    for (int j = i + 1; j < n; ++j) {
      sum -= row_i[j] * xdata[static_cast<std::size_t>(j)];
    }
    const double diag = row_i[i];
    if (std::abs(diag) < kEps) {
      return false;
    }
    xdata[si] = sum / diag;
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
