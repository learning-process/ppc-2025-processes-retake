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

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
bool NazarovaKGaussianVertSchemeSEQ::RunImpl() {
  // Forward elimination with partial pivoting
  for (int k = 0; k < n_; k++) {
    int pivot_row = k;
    double max_abs = std::abs(At(aug_, n_, k, k));
    for (int i = k + 1; i < n_; i++) {
      const double v = std::abs(At(aug_, n_, i, k));
      if (v > max_abs) {
        max_abs = v;
        pivot_row = i;
      }
    }

    if (max_abs < kEps) {
      return false;  // singular / ill-conditioned
    }

    if (pivot_row != k) {
      for (int j = k; j <= n_; j++) {
        std::swap(At(aug_, n_, k, j), At(aug_, n_, pivot_row, j));
      }
    }

    const double pivot = At(aug_, n_, k, k);
    for (int i = k + 1; i < n_; i++) {
      const double factor = At(aug_, n_, i, k) / pivot;
      At(aug_, n_, i, k) = 0.0;
      for (int j = k + 1; j <= n_; j++) {
        At(aug_, n_, i, j) -= factor * At(aug_, n_, k, j);
      }
    }
  }

  // Back substitution
  auto &x = GetOutput();
  for (int i = n_ - 1; i >= 0; i--) {
    double sum = 0.0;
    for (int j = i + 1; j < n_; j++) {
      sum += At(aug_, n_, i, j) * x[static_cast<std::size_t>(j)];
    }
    const double rhs = At(aug_, n_, i, n_);
    const double diag = At(aug_, n_, i, i);
    if (std::abs(diag) < kEps) {
      return false;
    }
    x[static_cast<std::size_t>(i)] = (rhs - sum) / diag;
  }

  return true;
}

bool NazarovaKGaussianVertSchemeSEQ::PostProcessingImpl() {
  return GetOutput().size() == static_cast<std::size_t>(n_);
}

}  // namespace nazarova_k_gaussian_vert_scheme_processes
