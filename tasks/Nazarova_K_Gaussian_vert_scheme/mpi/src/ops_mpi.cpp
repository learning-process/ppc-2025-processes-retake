#include "Nazarova_K_Gaussian_vert_scheme/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "Nazarova_K_Gaussian_vert_scheme/common/include/common.hpp"

namespace nazarova_k_gaussian_vert_scheme_processes {
namespace {

constexpr double kEps = 1e-12;

inline int OwnerOfCol(int col, int num_proc, int total_cols) {
  return (((col + 1) * num_proc) - 1) / total_cols;
}

inline int ColStart(int rank, int num_proc, int total_cols) {
  return (rank * total_cols) / num_proc;
}

inline int ColEnd(int rank, int num_proc, int total_cols) {
  return ((rank + 1) * total_cols) / num_proc;
}

inline double &LocalAt(std::vector<double> &local_aug, int local_cols, int row, int local_col) {
  return local_aug[(static_cast<std::size_t>(row) * static_cast<std::size_t>(local_cols)) +
                   static_cast<std::size_t>(local_col)];
}

inline double LocalAtConst(const std::vector<double> &local_aug, int local_cols, int row, int local_col) {
  return local_aug[(static_cast<std::size_t>(row) * static_cast<std::size_t>(local_cols)) +
                   static_cast<std::size_t>(local_col)];
}

bool FindPivotAndBroadcast(int rank, int owner_k, int k, int n, int local_cols, int local_k,
                           std::vector<double> &local_aug, int &pivot_row, double &pivot_abs) {
  pivot_row = k;
  pivot_abs = 0.0;
  if (rank == owner_k) {
    pivot_abs = std::abs(LocalAt(local_aug, local_cols, k, local_k));
    for (int i = k + 1; i < n; i++) {
      const double v = std::abs(LocalAt(local_aug, local_cols, i, local_k));
      if (v > pivot_abs) {
        pivot_abs = v;
        pivot_row = i;
      }
    }
  }
  MPI_Bcast(&pivot_row, 1, MPI_INT, owner_k, MPI_COMM_WORLD);
  MPI_Bcast(&pivot_abs, 1, MPI_DOUBLE, owner_k, MPI_COMM_WORLD);
  return pivot_abs >= kEps;
}

void SwapPivotRowMPI(int k, int pivot_row, int local_cols, std::vector<double> &local_aug) {
  for (int lc = 0; lc < local_cols; lc++) {
    std::swap(LocalAt(local_aug, local_cols, k, lc), LocalAt(local_aug, local_cols, pivot_row, lc));
  }
}

std::vector<double> GatherPivotColumn(int rank, int owner_k, int k, int n, int local_cols, int local_k,
                                      std::vector<double> &local_aug) {
  std::vector<double> pivot_col_vals(static_cast<std::size_t>(n - k), 0.0);
  if (rank == owner_k) {
    for (int i = k; i < n; i++) {
      pivot_col_vals[static_cast<std::size_t>(i - k)] = LocalAt(local_aug, local_cols, i, local_k);
    }
  }
  MPI_Bcast(pivot_col_vals.data(), static_cast<int>(pivot_col_vals.size()), MPI_DOUBLE, owner_k, MPI_COMM_WORLD);
  return pivot_col_vals;
}

void EliminateColumnMPI(int rank, int owner_k, int k, int n, int col_start, int col_end, int local_cols, int local_k,
                        std::vector<double> &local_aug, const std::vector<double> &pivot_col_vals) {
  const double pivot = pivot_col_vals[0];
  const int first_upd_gcol = k + 1;
  for (int i = k + 1; i < n; i++) {
    const double factor = pivot_col_vals[static_cast<std::size_t>(i - k)] / pivot;
    for (int gcol = std::max(first_upd_gcol, col_start); gcol < col_end; gcol++) {
      const int lc = gcol - col_start;
      const double piv_val = LocalAt(local_aug, local_cols, k, lc);
      LocalAt(local_aug, local_cols, i, lc) -= factor * piv_val;
    }
    if (rank == owner_k) {
      LocalAt(local_aug, local_cols, i, local_k) = 0.0;
    }
  }
}

bool ForwardEliminationMPI(int rank, int size, int n, int col_start, int col_end, int local_cols,
                           std::vector<double> &local_aug) {
  const int total_cols = n + 1;
  for (int k = 0; k < n; k++) {
    const int owner_k = OwnerOfCol(k, size, total_cols);
    const int local_k = k - ColStart(owner_k, size, total_cols);

    int pivot_row = k;
    double pivot_abs = 0.0;
    if (!FindPivotAndBroadcast(rank, owner_k, k, n, local_cols, local_k, local_aug, pivot_row, pivot_abs)) {
      return false;
    }
    if (pivot_row != k) {
      SwapPivotRowMPI(k, pivot_row, local_cols, local_aug);
    }
    std::vector<double> pivot_col_vals = GatherPivotColumn(rank, owner_k, k, n, local_cols, local_k, local_aug);
    if (std::abs(pivot_col_vals[0]) < kEps) {
      return false;
    }
    EliminateColumnMPI(rank, owner_k, k, n, col_start, col_end, local_cols, local_k, local_aug, pivot_col_vals);
  }
  return true;
}

bool BackSubstitutionMPI(int rank, int size, int n, int col_start, int col_end, int local_cols,
                         const std::vector<double> &local_aug, std::vector<double> &x) {
  const int total_cols = n + 1;
  const int rhs_col = n;
  const int rhs_owner = OwnerOfCol(rhs_col, size, total_cols);

  for (int i = n - 1; i >= 0; i--) {
    const int owner_i = OwnerOfCol(i, size, total_cols);
    const int local_i = i - ColStart(owner_i, size, total_cols);

    double diag = 0.0;
    if (rank == owner_i) {
      diag = LocalAtConst(local_aug, local_cols, i, local_i);
    }
    MPI_Bcast(&diag, 1, MPI_DOUBLE, owner_i, MPI_COMM_WORLD);
    if (std::abs(diag) < kEps) {
      return false;
    }
    double rhs = 0.0;
    if (rank == rhs_owner) {
      rhs = LocalAtConst(local_aug, local_cols, i, rhs_col - col_start);
    }
    MPI_Bcast(&rhs, 1, MPI_DOUBLE, rhs_owner, MPI_COMM_WORLD);

    double local_sum = 0.0;
    for (int gcol = std::max(i + 1, col_start); gcol < std::min(n, col_end); gcol++) {
      const int lc = gcol - col_start;
      local_sum += LocalAtConst(local_aug, local_cols, i, lc) * x[static_cast<std::size_t>(gcol)];
    }
    double global_sum = 0.0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    x[static_cast<std::size_t>(i)] = (rhs - global_sum) / diag;
  }
  return true;
}

}  // namespace

NazarovaKGaussianVertSchemeMPI::NazarovaKGaussianVertSchemeMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool NazarovaKGaussianVertSchemeMPI::ValidationImpl() {
  const auto &in = GetInput();
  if (in.n <= 0) {
    return false;
  }
  const auto expected = static_cast<std::size_t>(in.n) * static_cast<std::size_t>(in.n + 1);
  return in.augmented.size() == expected;
}

bool NazarovaKGaussianVertSchemeMPI::PreProcessingImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  n_ = GetInput().n;
  const int total_cols = n_ + 1;
  col_start_ = ColStart(rank, size, total_cols);
  col_end_ = ColEnd(rank, size, total_cols);
  local_cols_ = col_end_ - col_start_;

  local_aug_.assign(static_cast<std::size_t>(n_) * static_cast<std::size_t>(local_cols_), 0.0);

  const auto &aug = GetInput().augmented;
  for (int i = 0; i < n_; i++) {
    for (int gcol = col_start_; gcol < col_end_; gcol++) {
      const auto src =
          (static_cast<std::size_t>(i) * static_cast<std::size_t>(total_cols)) + static_cast<std::size_t>(gcol);
      LocalAt(local_aug_, local_cols_, i, gcol - col_start_) = aug[src];
    }
  }

  GetOutput().assign(static_cast<std::size_t>(n_), 0.0);
  return true;
}

bool NazarovaKGaussianVertSchemeMPI::RunImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (!ForwardEliminationMPI(rank, size, n_, col_start_, col_end_, local_cols_, local_aug_)) {
    return false;
  }
  return BackSubstitutionMPI(rank, size, n_, col_start_, col_end_, local_cols_, local_aug_, GetOutput());
}

bool NazarovaKGaussianVertSchemeMPI::PostProcessingImpl() {
  return GetOutput().size() == static_cast<std::size_t>(n_);
}

}  // namespace nazarova_k_gaussian_vert_scheme_processes
