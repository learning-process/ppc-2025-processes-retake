#include "Nazarova_K_Gaussian_vert_scheme/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "Nazarova_K_Gaussian_vert_scheme/common/include/common.hpp"

namespace nazarova_k_gaussian_vert_scheme_processes {
namespace {

constexpr double kEps = 1e-12;

inline double &At(std::vector<double> &aug, int n, int row, int col) {
  return aug[(static_cast<std::size_t>(row) * static_cast<std::size_t>(n + 1)) + static_cast<std::size_t>(col)];
}

inline double AtConst(const std::vector<double> &aug, int n, int row, int col) {
  return aug[(static_cast<std::size_t>(row) * static_cast<std::size_t>(n + 1)) + static_cast<std::size_t>(col)];
}

std::pair<int, double> FindPivotRow(std::vector<double> &aug, int n, int k) {
  int pivot_row = k;
  double max_abs = std::abs(At(aug, n, k, k));
  for (int i = k + 1; i < n; i++) {
    const double v = std::abs(At(aug, n, i, k));
    if (v > max_abs) {
      max_abs = v;
      pivot_row = i;
    }
  }
  return {pivot_row, max_abs};
}

void SwapRows(std::vector<double> &aug, int n, int k, int pivot_row) {
  for (int j = k; j <= n; j++) {
    std::swap(At(aug, n, k, j), At(aug, n, pivot_row, j));
  }
}

void EliminateColumn(std::vector<double> &aug, int n, int k) {
  const double pivot = At(aug, n, k, k);
  for (int i = k + 1; i < n; i++) {
    const double factor = At(aug, n, i, k) / pivot;
    At(aug, n, i, k) = 0.0;
    for (int j = k + 1; j <= n; j++) {
      At(aug, n, i, j) -= factor * At(aug, n, k, j);
    }
  }
}

bool ForwardElimination(std::vector<double> &aug, int n) {
  for (int k = 0; k < n; k++) {
    const auto [pivot_row, max_abs] = FindPivotRow(aug, n, k);
    if (max_abs < kEps) {
      return false;
    }
    if (pivot_row != k) {
      SwapRows(aug, n, k, pivot_row);
    }
    EliminateColumn(aug, n, k);
  }
  return true;
}

bool BackSubstitution(const std::vector<double> &aug, int n, std::vector<double> &x) {
  for (int i = n - 1; i >= 0; i--) {
    double sum = 0.0;
    for (int j = i + 1; j < n; j++) {
      sum += AtConst(aug, n, i, j) * x[static_cast<std::size_t>(j)];
    }
    const double rhs = AtConst(aug, n, i, n);
    const double diag = AtConst(aug, n, i, i);
    if (std::abs(diag) < kEps) {
      return false;
    }
    x[static_cast<std::size_t>(i)] = (rhs - sum) / diag;
  }
  return true;
}

}  // namespace

NazarovaKGaussianVertSchemeSEQ::NazarovaKGaussianVertSchemeSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool NazarovaKGaussianVertSchemeSEQ::ValidationImpl() {
  const auto &in = GetInput();
  if (in.n <= 0) {
    return false;
  }
  const auto expected = static_cast<std::size_t>(in.n) * static_cast<std::size_t>(in.n + 1);
  return in.augmented.size() == expected;
}

bool NazarovaKGaussianVertSchemeSEQ::PreProcessingImpl() {
  n_ = GetInput().n;
  aug_ = GetInput().augmented;
  GetOutput().assign(static_cast<std::size_t>(n_), 0.0);
  return true;
}

bool NazarovaKGaussianVertSchemeSEQ::RunImpl() {
  if (!ForwardElimination(aug_, n_)) {
    return false;
  }
  return BackSubstitution(aug_, n_, GetOutput());
}

bool NazarovaKGaussianVertSchemeSEQ::PostProcessingImpl() {
  return GetOutput().size() == static_cast<std::size_t>(n_);
}

}  // namespace nazarova_k_gaussian_vert_scheme_processes
