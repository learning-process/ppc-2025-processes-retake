#include "fedoseev_gaussian_method_horizontal_strip_scheme/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "fedoseev_gaussian_method_horizontal_strip_scheme/common/include/common.hpp"

namespace fedoseev_gaussian_method_horizontal_strip_scheme {

FedoseevGaussianMethodHorizontalStripSchemeSEQ::FedoseevGaussianMethodHorizontalStripSchemeSEQ(
    const InType &input_data) {
  SetTypeOfTask(GetStaticTypeOfTask());

  InType &input_ref = GetInput();
  input_ref.clear();
  input_ref.reserve(input_data.size());

  for (const auto &row : input_data) {
    input_ref.push_back(row);
  }

  GetOutput().clear();
}

bool FedoseevGaussianMethodHorizontalStripSchemeSEQ::ValidationImpl() {
  const InType &matrix = GetInput();
  if (matrix.empty()) {
    return false;
  }

  const size_t n = matrix.size();
  const size_t cols = matrix[0].size();

  if (cols < n + 1) {
    return false;
  }

  for (size_t i = 1; i < n; ++i) {
    if (matrix[i].size() != cols) {
      return false;
    }
  }

  return true;
}

bool FedoseevGaussianMethodHorizontalStripSchemeSEQ::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

bool FedoseevGaussianMethodHorizontalStripSchemeSEQ::RunImpl() {
  InType augmented_matrix = GetInput();
  const size_t n = augmented_matrix.size();
  const size_t cols = augmented_matrix[0].size();

  if (!ForwardElimination(augmented_matrix, n, cols)) {
    return false;
  }

  GetOutput() = BackwardSubstitution(augmented_matrix, n, cols);
  return true;
}

bool FedoseevGaussianMethodHorizontalStripSchemeSEQ::ForwardElimination(InType &matrix, size_t n, size_t cols) {
  for (size_t k = 0; k < n; ++k) {
    const size_t pivot_row_idx = SelectPivotRow(matrix, k, n);
    if (pivot_row_idx != k) {
      std::swap(matrix[k], matrix[pivot_row_idx]);
    }

    if (std::abs(matrix[k][k]) < 1e-10) {
      return false;
    }

    EliminateRows(matrix, k, n, cols);
  }
  return true;
}

size_t FedoseevGaussianMethodHorizontalStripSchemeSEQ::SelectPivotRow(const InType &matrix, size_t k, size_t n) {
  size_t best_row = k;
  double max_value = std::abs(matrix[k][k]);

  for (size_t i = k + 1; i < n; ++i) {
    const double current_value = std::abs(matrix[i][k]);
    if (current_value > max_value) {
      max_value = current_value;
      best_row = i;
    }
  }

  return best_row;
}

void FedoseevGaussianMethodHorizontalStripSchemeSEQ::EliminateRows(InType &matrix, size_t k, size_t n, size_t cols) {
  for (size_t i = k + 1; i < n; ++i) {
    if (std::abs(matrix[i][k]) > 1e-10) {
      const double factor = matrix[i][k] / matrix[k][k];
      for (size_t j = k; j < cols; ++j) {
        matrix[i][j] -= factor * matrix[k][j];
      }
    }
  }
}

std::vector<double> FedoseevGaussianMethodHorizontalStripSchemeSEQ::BackwardSubstitution(const InType &matrix, size_t n,
                                                                                         size_t cols) {
  std::vector<double> solution(n, 0.0);

  for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
    double sum = 0.0;
    for (size_t j = static_cast<size_t>(i) + 1; j < n; ++j) {
      sum += matrix[static_cast<size_t>(i)][j] * solution[j];
    }

    solution[static_cast<size_t>(i)] =
        (matrix[static_cast<size_t>(i)][cols - 1] - sum) / matrix[static_cast<size_t>(i)][static_cast<size_t>(i)];
  }

  return solution;
}

bool FedoseevGaussianMethodHorizontalStripSchemeSEQ::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace fedoseev_gaussian_method_horizontal_strip_scheme
