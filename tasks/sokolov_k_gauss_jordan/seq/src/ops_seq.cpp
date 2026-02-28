#include "sokolov_k_gauss_jordan/seq/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

#include "sokolov_k_gauss_jordan/common/include/common.hpp"

namespace sokolov_k_gauss_jordan {

namespace {

void GenerateSystem(int n, std::vector<double> &matrix) {
  int cols = n + 1;
  matrix.assign(static_cast<size_t>(n) * cols, 0.0);
  unsigned int seed = static_cast<unsigned int>(n) * 1234567U;
  auto gen = [&seed]() -> double {
    seed = (seed * 1103515245U) + 12345U;
    return static_cast<double>(((seed >> 16) % 100) + 1);
  };
  for (int i = 0; i < n; i++) {
    double row_sum = 0.0;
    for (int j = 0; j < n; j++) {
      if (i != j) {
        matrix[(i * cols) + j] = gen() / 100.0;
        row_sum += std::abs(matrix[(i * cols) + j]);
      }
    }
    matrix[(i * cols) + i] = row_sum + gen();
  }
  for (int i = 0; i < n; i++) {
    double b_val = 0.0;
    for (int j = 0; j < n; j++) {
      b_val += matrix[(i * cols) + j] * (j + 1.0);
    }
    matrix[(i * cols) + n] = b_val;
  }
}

void GaussJordanStep(std::vector<double> &matrix, int n, int cols, int k) {
  int max_row = k;
  double max_val = std::abs(matrix[(k * cols) + k]);
  for (int i = k + 1; i < n; i++) {
    double val = std::abs(matrix[(i * cols) + k]);
    if (val > max_val) {
      max_val = val;
      max_row = i;
    }
  }

  if (max_row != k) {
    for (int j = 0; j < cols; j++) {
      double tmp = matrix[(k * cols) + j];
      matrix[(k * cols) + j] = matrix[(max_row * cols) + j];
      matrix[(max_row * cols) + j] = tmp;
    }
  }

  double pivot = matrix[(k * cols) + k];
  for (int j = k; j < cols; j++) {
    matrix[(k * cols) + j] /= pivot;
  }

  for (int i = 0; i < n; i++) {
    if (i != k) {
      double factor = matrix[(i * cols) + k];
      for (int j = k; j < cols; j++) {
        matrix[(i * cols) + j] -= factor * matrix[(k * cols) + j];
      }
    }
  }
}

}  // namespace

SokolovKGaussJordanSEQ::SokolovKGaussJordanSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool SokolovKGaussJordanSEQ::ValidationImpl() {
  return GetInput() > 0 && GetOutput() == 0;
}

bool SokolovKGaussJordanSEQ::PreProcessingImpl() {
  n_ = GetInput();
  if (n_ <= 0) {
    return false;
  }
  GenerateSystem(n_, matrix_);
  original_matrix_ = matrix_;
  solution_.assign(n_, 0.0);
  return true;
}

bool SokolovKGaussJordanSEQ::RunImpl() {
  if (n_ <= 0) {
    return false;
  }
  matrix_ = original_matrix_;
  int cols = n_ + 1;

  for (int k = 0; k < n_; k++) {
    GaussJordanStep(matrix_, n_, cols, k);
  }

  for (int i = 0; i < n_; i++) {
    solution_[i] = matrix_[(i * cols) + n_];
  }
  return true;
}

bool SokolovKGaussJordanSEQ::PostProcessingImpl() {
  const double eps = 1e-6;
  for (int i = 0; i < n_; i++) {
    if (std::abs(solution_[i] - (i + 1.0)) > eps) {
      return false;
    }
  }
  GetOutput() = GetInput();
  return true;
}

}  // namespace sokolov_k_gauss_jordan
