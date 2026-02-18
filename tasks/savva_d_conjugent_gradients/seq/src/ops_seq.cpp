#include "savva_d_conjugent_gradients/seq/include/ops_seq.hpp"

// #include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "savva_d_conjugent_gradients/common/include/common.hpp"

namespace savva_d_conjugent_gradients {

SavvaDConjugentGradientsSEQ::SavvaDConjugentGradientsSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<double>{};
}

bool SavvaDConjugentGradientsSEQ::ValidationImpl() {
  const auto &in = GetInput();

  if (in.n < 0 || in.a.size() != static_cast<size_t>(in.n) * in.n || in.b.size() != static_cast<size_t>(in.n)) {
    return false;
  }

  for (int i = 0; i < in.n; ++i) {
    for (int j = i + 1; j < in.n; ++j) {
      if (std::abs(in.a[(i * in.n) + j] - in.a[(j * in.n) + i]) > 1e-9) {
        return false;
      }
    }
  }

  return true;
}

bool SavvaDConjugentGradientsSEQ::PreProcessingImpl() {
  GetOutput().assign(GetInput().n, 0.0);
  return true;
}

void SavvaDConjugentGradientsSEQ::ComputeAp(const std::vector<double> &a, const std::vector<double> &p,
                                            std::vector<double> &ap, int n) {
  for (int i = 0; i < n; ++i) {
    double sum = 0.0;
    for (int j = 0; j < n; ++j) {
      sum += a[(i * n) + j] * p[j];
    }
    ap[i] = sum;
  }
}

void SavvaDConjugentGradientsSEQ::UpdateXR(std::vector<double> &x, std::vector<double> &r, const std::vector<double> &p,
                                           const std::vector<double> &ap, double alpha, int n) {
  for (int i = 0; i < n; ++i) {
    x[i] += alpha * p[i];
    r[i] -= alpha * ap[i];
  }
}

void SavvaDConjugentGradientsSEQ::UpdateP(std::vector<double> &p, const std::vector<double> &r, double beta, int n) {
  for (int i = 0; i < n; ++i) {
    p[i] = r[i] + (beta * p[i]);
  }
}

bool SavvaDConjugentGradientsSEQ::RunImpl() {
  const auto &input = GetInput();
  auto &x = GetOutput();

  const int n = input.n;
  if (n == 0) {
    x = std::vector<double>{};
    return true;
  }
  const int max_iter = 10000;
  const double eps = 1e-9;

  std::vector<double> r(n);
  std::vector<double> p(n);
  std::vector<double> ap(n);

  for (int i = 0; i < n; ++i) {
    r[i] = input.b[i];  // r0 = b - A*x0 => так как x0=0, то r0 = b
    p[i] = r[i];
  }

  double rs_old = 0.0;  // r^T * r
  for (int i = 0; i < n; ++i) {
    rs_old += r[i] * r[i];
  }

  for (int k = 0; k < max_iter; ++k) {
    // Проверка сходимости
    if (std::sqrt(rs_old) < eps) {
      break;
    }

    // 2.1 Вычисление ap = A * p
    ComputeAp(input.a, p, ap, n);

    // 2.2 Вычисление alpha = (r^T * r) / (p^T * A * p)
    // Знаменатель p^T * ap
    double p_ap = 0.0;
    for (int i = 0; i < n; ++i) {
      p_ap += p[i] * ap[i];
    }

    // Защита от деления на ноль (если матрица не положительно определена)
    if (std::abs(p_ap) < 1e-15) {
      return false;
    }

    double alpha = rs_old / p_ap;

    // 2.3 x = x + alpha * p
    // 2.4 r = r - alpha * ap
    UpdateXR(x, r, p, ap, alpha, n);

    // Вычисление новой невязки r^T * r (для следующего шага и проверки)
    double rs_new = 0.0;
    for (int i = 0; i < n; ++i) {
      rs_new += r[i] * r[i];
    }

    // Проверка условия выхода
    if (std::sqrt(rs_new) < eps) {
      break;
    }

    // 2.5 Вычисление beta
    double beta = rs_new / rs_old;

    // 2.6 p = r + beta * p
    UpdateP(p, r, beta, n);

    rs_old = rs_new;
  }

  return true;
}

bool SavvaDConjugentGradientsSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace savva_d_conjugent_gradients
