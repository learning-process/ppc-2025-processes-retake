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
  // Block distribution: [ColStart(rank), ColEnd(rank))
  // NOTE: simple (col * p) / total_cols breaks on boundaries when total_cols is not divisible by p.
  // Use a ceil-based mapping that matches ColStart/ColEnd.
  // owner(col) = ceil((col+1) * p / total_cols) - 1
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

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
bool NazarovaKGaussianVertSchemeMPI::RunImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const int total_cols = n_ + 1;
  const int rhs_col = n_;
  const int rhs_owner = OwnerOfCol(rhs_col, size, total_cols);

  // Forward elimination with partial pivoting
  std::vector<double> pivot_col_vals;  // rows k..n-1 of column k (broadcasted)
  for (int k = 0; k < n_; k++) {
    const int owner_k = OwnerOfCol(k, size, total_cols);
    const int local_k = k - ColStart(owner_k, size, total_cols);  // local index on owner_k

    int pivot_row = k;
    double pivot_abs = 0.0;
    if (rank == owner_k) {
      pivot_abs = std::abs(LocalAt(local_aug_, local_cols_, k, local_k));
      for (int i = k + 1; i < n_; i++) {
        const double v = std::abs(LocalAt(local_aug_, local_cols_, i, local_k));
        if (v > pivot_abs) {
          pivot_abs = v;
          pivot_row = i;
        }
      }
    }

    MPI_Bcast(&pivot_row, 1, MPI_INT, owner_k, MPI_COMM_WORLD);
    MPI_Bcast(&pivot_abs, 1, MPI_DOUBLE, owner_k, MPI_COMM_WORLD);
    if (pivot_abs < kEps) {
      return false;
    }

    if (pivot_row != k) {
      for (int lc = 0; lc < local_cols_; lc++) {
        std::swap(LocalAt(local_aug_, local_cols_, k, lc), LocalAt(local_aug_, local_cols_, pivot_row, lc));
      }
    }

    // Broadcast pivot column values (after row swap) for factor computation
    pivot_col_vals.assign(static_cast<std::size_t>(n_ - k), 0.0);
    if (rank == owner_k) {
      for (int i = k; i < n_; i++) {
        pivot_col_vals[static_cast<std::size_t>(i - k)] = LocalAt(local_aug_, local_cols_, i, local_k);
      }
    }
    MPI_Bcast(pivot_col_vals.data(), static_cast<int>(pivot_col_vals.size()), MPI_DOUBLE, owner_k, MPI_COMM_WORLD);

    const double pivot = pivot_col_vals[0];
    if (std::abs(pivot) < kEps) {
      return false;
    }

    // Eliminate rows below pivot
    const int first_upd_gcol = k + 1;
    for (int i = k + 1; i < n_; i++) {
      const double factor = pivot_col_vals[static_cast<std::size_t>(i - k)] / pivot;

      // Update only local columns with global col > k (including RHS if local)
      for (int gcol = std::max(first_upd_gcol, col_start_); gcol < col_end_; gcol++) {
        const int lc = gcol - col_start_;
        const double piv_val = LocalAt(local_aug_, local_cols_, k, lc);
        LocalAt(local_aug_, local_cols_, i, lc) -= factor * piv_val;
      }

      // Owner of pivot column sets eliminated element to 0 for stability
      if (rank == owner_k) {
        LocalAt(local_aug_, local_cols_, i, local_k) = 0.0;
      }
    }
  }

  // Back substitution
  auto &x = GetOutput();
  for (int i = n_ - 1; i >= 0; i--) {
    const int owner_i = OwnerOfCol(i, size, total_cols);
    const int local_i = i - ColStart(owner_i, size, total_cols);

    double diag = 0.0;
    if (rank == owner_i) {
      diag = LocalAt(local_aug_, local_cols_, i, local_i);
    }
    MPI_Bcast(&diag, 1, MPI_DOUBLE, owner_i, MPI_COMM_WORLD);
    if (std::abs(diag) < kEps) {
      return false;
    }

    double rhs = 0.0;
    if (rank == rhs_owner) {
      rhs = LocalAt(local_aug_, local_cols_, i, rhs_col - col_start_);
    }
    MPI_Bcast(&rhs, 1, MPI_DOUBLE, rhs_owner, MPI_COMM_WORLD);

    double local_sum = 0.0;
    for (int gcol = std::max(i + 1, col_start_); gcol < std::min(n_, col_end_); gcol++) {
      const int lc = gcol - col_start_;
      local_sum += LocalAt(local_aug_, local_cols_, i, lc) * x[static_cast<std::size_t>(gcol)];
    }
    double global_sum = 0.0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    x[static_cast<std::size_t>(i)] = (rhs - global_sum) / diag;
  }

  return true;
}

bool NazarovaKGaussianVertSchemeMPI::PostProcessingImpl() {
  return GetOutput().size() == static_cast<std::size_t>(n_);
}

}  // namespace nazarova_k_gaussian_vert_scheme_processes
