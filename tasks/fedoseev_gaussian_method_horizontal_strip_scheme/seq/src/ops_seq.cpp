#include "fedoseev_gaussian_method_horizontal_strip_scheme/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "fedoseev_gaussian_method_horizontal_strip_scheme/common/include/common.hpp"

namespace fedoseev_gaussian_method_horizontal_strip_scheme {

FedoseevTestTaskSEQ::FedoseevTestTaskSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<double>();
}

bool FedoseevTestTaskSEQ::ValidationImpl() {
  const InType &augmented_matrix = GetInput();
  size_t n = augmented_matrix.size();

  if (n == 0) {
    return false;
  }

  return std::all_of(augmented_matrix.begin(), augmented_matrix.end(),
                     [n](const auto &row) { return row.size() == static_cast<size_t>(n) + 1; });

  return true;
}

bool FedoseevTestTaskSEQ::PreProcessingImpl() {
  const InType &augmented_matrix = GetInput();
  size_t n = augmented_matrix.size();

  for (size_t i = 0; i < n; ++i) {
    if (std::abs(augmented_matrix[i][i]) < 1e-10) {
      bool found = false;
      for (size_t j = i + 1; j < n; ++j) {
        if (std::abs(augmented_matrix[j][i]) > 1e-10) {
          found = true;
          break;
        }
      }
      if (!found) {
        return false;
      }
    }
  }

  return true;
}

bool FedoseevTestTaskSEQ::RunImpl() {
  InType augmented_matrix = GetInput();
  size_t n = augmented_matrix.size();
  std::vector<double> x(n, 0.0);

  for (size_t i = 0; i < n; ++i) {
    size_t pivot = i;
    for (size_t k = i + 1; k < n; ++k) {
      if (std::abs(augmented_matrix[k][i]) > std::abs(augmented_matrix[pivot][i])) {
        pivot = k;
      }
    }

    if (std::abs(augmented_matrix[pivot][i]) < 1e-12) {
      return false;
    }

    if (pivot != i) {
      std::swap(augmented_matrix[i], augmented_matrix[pivot]);
    }

    for (size_t k = i + 1; k < n; ++k) {
      double factor = augmented_matrix[k][i] / augmented_matrix[i][i];
      for (size_t j = i; j < n + 1; ++j) {
        augmented_matrix[k][j] -= factor * augmented_matrix[i][j];
      }
    }
  }

  for (size_t i = n - 1; i >= 0; --i) {
    x[i] = augmented_matrix[i][n];
    for (size_t j = i + 1; j < n; ++j) {
      x[i] -= augmented_matrix[i][j] * x[j];
    }
    x[i] /= augmented_matrix[i][i];
  }

  GetOutput() = x;
  return !GetOutput().empty();
}

bool FedoseevTestTaskSEQ::PostProcessingImpl() {
  const InType &augmented_matrix = GetInput();
  const auto &x = GetOutput();
  size_t n = augmented_matrix.size();

  double residual = 0.0;
  for (size_t i = 0; i < n; ++i) {
    double sum = 0.0;
    for (size_t j = 0; j < n; ++j) {
      sum += augmented_matrix[i][j] * x[j];
    }
    residual += std::abs(sum - augmented_matrix[i][n]);
  }

  return residual < 1e-6 * n;
}

}  // namespace fedoseev_gaussian_method_horizontal_strip_scheme
