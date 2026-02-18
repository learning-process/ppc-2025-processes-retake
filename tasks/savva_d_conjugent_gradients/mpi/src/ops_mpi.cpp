#include "savva_d_conjugent_gradients/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
// #include <ranges>
#include <vector>

#include "savva_d_conjugent_gradients/common/include/common.hpp"

namespace savva_d_conjugent_gradients {

SavvaDConjugentGradientsMPI::SavvaDConjugentGradientsMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<double>{};
}

bool SavvaDConjugentGradientsMPI::ValidationImpl() {
  const auto &in = GetInput();

  if (in.n < 0 || in.a.size() != static_cast<size_t>(in.n) * static_cast<size_t>(in.n) ||
      in.b.size() != static_cast<size_t>(in.n)) {
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

bool SavvaDConjugentGradientsMPI::PreProcessingImpl() {
  GetOutput().assign(GetInput().n, 0.0);
  return true;
}

std::vector<double> SavvaDConjugentGradientsMPI::ComputeLocalAp(int n, int local_rows,
                                                                const std::vector<double> &local_a,
                                                                const std::vector<double> &p) {
  std::vector<double> local_ap(local_rows, 0.0);
  for (int i = 0; i < local_rows; ++i) {
    double sum = 0.0;
    for (int j = 0; j < n; ++j) {
      sum += local_a[(i * n) + j] * p[j];
    }
    local_ap[i] = sum;
  }
  return local_ap;
}

void SavvaDConjugentGradientsMPI::UpdateXR(std::vector<double> &x, std::vector<double> &r, const std::vector<double> &p,
                                           const std::vector<double> &global_ap, double alpha, int n) {
  for (int i = 0; i < n; ++i) {
    x[i] += alpha * p[i];
    r[i] -= alpha * global_ap[i];
  }
}

void SavvaDConjugentGradientsMPI::RunCGIterations(int n, int local_rows, int local_offset, std::vector<double> &r,
                                                  const std::vector<double> &local_a, std::vector<double> &vector_x,
                                                  const std::vector<int> &counts, const std::vector<int> &displs) {
  // Константы
  const int max_iter = 1000;
  const double eps = 1e-9;

  std::vector<double> p(n, 0.0);
  p = r;
  std::vector<double> global_ap(n, 0.0);
  // std::vector<double> local_ap(local_rows, 0.0);

  double rr_old = std::inner_product(r.begin(), r.end(), r.begin(), 0.0);

  for (int iter = 0; iter < max_iter; ++iter) {
    if (std::sqrt(rr_old) < eps) {
      break;
    }

    // A_local * p

    std::vector<double> local_ap = ComputeLocalAp(n, local_rows, local_a, p);

    // собираем глобальный ap
    std::ranges::fill(global_ap, 0.0);
    for (int i = 0; i < local_rows; ++i) {
      global_ap[local_offset + i] = local_ap[i];
    }
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, global_ap.data(), counts.data(), displs.data(), MPI_DOUBLE,
                   MPI_COMM_WORLD);

    double p_ap = std::inner_product(p.begin(), p.end(), global_ap.begin(), 0.0);

    if (std::abs(p_ap) < eps) {
      break;
    }

    double alpha = rr_old / p_ap;

    UpdateXR(vector_x, r, p, global_ap, alpha, n);

    double rr_new = std::inner_product(r.begin(), r.end(), r.begin(), 0.0);

    double beta = rr_new / rr_old;

    rr_old = rr_new;

    for (int i = 0; i < n; ++i) {
      p[i] = r[i] + (beta * p[i]);
    }
  }
}

bool SavvaDConjugentGradientsMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  const double *sendbuf_a = nullptr;  // будут ненулевыми только на 0 процессе

  int n = 0;
  if (rank == 0) {
    n = GetInput().n;
  }
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  std::vector<double> r(n, 0.0);

  if (rank == 0) {
    sendbuf_a = GetInput().a.data();
    const auto &full_b = GetInput().b;
    std::ranges::copy(full_b, r.begin());
  }

  MPI_Bcast(r.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (n == 0) {
    return true;
  }

  std::vector<int> counts(size);
  std::vector<int> displs(size);
  std::vector<int> counts_a(size);
  std::vector<int> displs_a(size);

  int rows_per_proc = n / size;
  int remainder = n % size;
  int offset = 0;
  int local_rows = 0;
  int local_offset = 0;

  for (int i = 0; i < size; ++i) {
    counts[i] = rows_per_proc + (i < remainder ? 1 : 0);
    displs[i] = offset;
    counts_a[i] = counts[i] * n;
    displs_a[i] = displs[i] * n;
    offset += counts[i];

    if (i == rank) {
      local_offset = displs[rank];
      local_rows = counts[rank];
    }
  }

  // Выделение памяти под локальные данные

  std::vector<double> local_a(static_cast<size_t>(local_rows) * static_cast<size_t>(n));

  // Рассылка данных
  MPI_Scatterv(sendbuf_a, counts_a.data(), displs_a.data(), MPI_DOUBLE, local_a.data(), local_rows * n, MPI_DOUBLE, 0,
               MPI_COMM_WORLD);

  auto &x = GetOutput();
  x.assign(n, 0.0);
  // Запуск алгоритма
  RunCGIterations(n, local_rows, local_offset, r, local_a, x, counts, displs);

  return true;
}

bool SavvaDConjugentGradientsMPI::PostProcessingImpl() {
  return true;
}

}  // namespace savva_d_conjugent_gradients
