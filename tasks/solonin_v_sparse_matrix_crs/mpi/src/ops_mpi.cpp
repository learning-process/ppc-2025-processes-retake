#include "solonin_v_sparse_matrix_crs/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <ranges>
#include <tuple>
#include <utility>
#include <vector>

#include "solonin_v_sparse_matrix_crs/common/include/common.hpp"

namespace solonin_v_sparse_matrix_crs {

SoloninVSparseMulCRSMPI::SoloninVSparseMulCRSMPI(const InType &in)
    : vals_a_(std::get<0>(in)),
      cols_a_(std::get<1>(in)),
      ptr_a_(std::get<2>(in)),
      vals_b_(std::get<3>(in)),
      cols_b_(std::get<4>(in)),
      ptr_b_(std::get<5>(in)),
      rows_a_(std::get<6>(in)),
      cols_a_count_(std::get<7>(in)),
      cols_b_count_(std::get<8>(in)) {
  SetTypeOfTask(GetStaticTypeOfTask());
}

bool SoloninVSparseMulCRSMPI::ValidationImpl() {
  int init = 0;
  MPI_Initialized(&init);
  if (init == 0) return false;
  int sz = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &sz);
  return sz >= 1;
}

bool SoloninVSparseMulCRSMPI::PreProcessingImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
  local_row_ids_.clear();
  local_vals_a_.clear();
  local_cols_a_.clear();
  local_ptr_a_.clear();
  local_vals_c_.clear();
  local_cols_c_.clear();
  local_ptr_c_.clear();
  vals_c_.clear();
  cols_c_.clear();
  ptr_c_.clear();
  return true;
}

bool SoloninVSparseMulCRSMPI::RunImpl() {
  if (world_size_ == 1) return RunSequential();

  int ra = 0;
  int ca = 0;
  int cb = 0;
  if (!BroadcastSizes(ra, cb, ca)) return true;

  BroadcastB();
  DistributeA();
  ComputeLocal();
  GatherResults();
  return true;
}

bool SoloninVSparseMulCRSMPI::RunSequential() {
  if (rank_ != 0) return true;

  ptr_c_.resize(rows_a_ + 1, 0);
  std::vector<std::vector<double>> rv(rows_a_);
  std::vector<std::vector<int>> rc(rows_a_);

  for (int i = 0; i < rows_a_; i++) {
    int sa = ptr_a_[i];
    int ea = ptr_a_[i + 1];
    std::vector<double> tmp(cols_b_count_, 0.0);
    for (int k = sa; k < ea; k++) {
      double av = vals_a_[k];
      int col = cols_a_[k];
      int sb = ptr_b_[col];
      int eb = ptr_b_[col + 1];
      for (int j = sb; j < eb; j++) {
        tmp[cols_b_[j]] += av * vals_b_[j];
      }
    }
    for (int j = 0; j < cols_b_count_; j++) {
      if (std::abs(tmp[j]) > 1e-12) {
        rv[i].push_back(tmp[j]);
        rc[i].push_back(j);
      }
    }
    SortRow(rv[i], rc[i]);
    ptr_c_[i + 1] = ptr_c_[i] + static_cast<int>(rc[i].size());
  }

  for (int i = 0; i < rows_a_; i++) {
    vals_c_.insert(vals_c_.end(), rv[i].begin(), rv[i].end());
    cols_c_.insert(cols_c_.end(), rc[i].begin(), rc[i].end());
  }

  GetOutput() = std::make_tuple(vals_c_, cols_c_, ptr_c_);
  return true;
}

bool SoloninVSparseMulCRSMPI::BroadcastSizes(int &rows_a, int &cols_a, int &cols_b) {
  if (rank_ == 0) {
    rows_a = rows_a_;
    cols_a = cols_a_count_;
    cols_b = cols_b_count_;
  }
  std::array<int, 3> sz = {rows_a, cols_a, cols_b};
  MPI_Bcast(sz.data(), 3, MPI_INT, 0, MPI_COMM_WORLD);
  rows_a_ = sz[0];
  cols_a_count_ = sz[1];
  cols_b_count_ = sz[2];
  return rows_a_ > 0 && cols_a_count_ > 0 && cols_b_count_ > 0;
}

void SoloninVSparseMulCRSMPI::BroadcastB() {
  std::array<int, 3> bsz = {0, 0, 0};
  if (rank_ == 0) {
    bsz[0] = static_cast<int>(vals_b_.size());
    bsz[1] = static_cast<int>(cols_b_.size());
    bsz[2] = static_cast<int>(ptr_b_.size());
  }
  MPI_Bcast(bsz.data(), 3, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank_ != 0) {
    vals_b_.resize(bsz[0]);
    cols_b_.resize(bsz[1]);
    ptr_b_.resize(bsz[2]);
  }
  MPI_Bcast(vals_b_.data(), bsz[0], MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(cols_b_.data(), bsz[1], MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(ptr_b_.data(), bsz[2], MPI_INT, 0, MPI_COMM_WORLD);
}

void SoloninVSparseMulCRSMPI::DistributeA() {
  local_row_ids_.clear();
  for (int i = 0; i < rows_a_; i++) {
    if (i % world_size_ == rank_) local_row_ids_.push_back(i);
  }

  if (rank_ == 0) {
    for (int dest = 1; dest < world_size_; dest++) SendAToRank(dest);

    local_ptr_a_.resize(local_row_ids_.size() + 1, 0);
    for (size_t idx = 0; idx < local_row_ids_.size(); idx++) {
      int r = local_row_ids_[idx];
      int s = ptr_a_[r];
      int e = ptr_a_[r + 1];
      local_vals_a_.insert(local_vals_a_.end(), vals_a_.begin() + s, vals_a_.begin() + e);
      local_cols_a_.insert(local_cols_a_.end(), cols_a_.begin() + s, cols_a_.begin() + e);
      local_ptr_a_[idx + 1] = local_ptr_a_[idx] + (e - s);
    }
  } else {
    ReceiveAFromRoot();
  }
}

void SoloninVSparseMulCRSMPI::SendAToRank(int dest) {
  std::vector<int> rows;
  for (int i = 0; i < rows_a_; i++) {
    if (i % world_size_ == dest) rows.push_back(i);
  }
  int cnt = static_cast<int>(rows.size());
  MPI_Send(&cnt, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
  if (cnt == 0) return;
  MPI_Send(rows.data(), cnt, MPI_INT, dest, 1, MPI_COMM_WORLD);
  for (int r : rows) {
    int s = ptr_a_[r];
    int e = ptr_a_[r + 1];
    int nnz = e - s;
    MPI_Send(&nnz, 1, MPI_INT, dest, 2, MPI_COMM_WORLD);
    if (nnz > 0) {
      MPI_Send(vals_a_.data() + s, nnz, MPI_DOUBLE, dest, 3, MPI_COMM_WORLD);
      MPI_Send(cols_a_.data() + s, nnz, MPI_INT, dest, 4, MPI_COMM_WORLD);
    }
  }
}

void SoloninVSparseMulCRSMPI::ReceiveAFromRoot() {
  int cnt = 0;
  MPI_Recv(&cnt, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  if (cnt == 0) return;
  local_row_ids_.resize(cnt);
  MPI_Recv(local_row_ids_.data(), cnt, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  local_ptr_a_.resize(cnt + 1, 0);
  for (int i = 0; i < cnt; i++) {
    int nnz = 0;
    MPI_Recv(&nnz, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (nnz > 0) {
      std::vector<double> tv(nnz);
      std::vector<int> tc(nnz);
      MPI_Recv(tv.data(), nnz, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(tc.data(), nnz, MPI_INT, 0, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      local_vals_a_.insert(local_vals_a_.end(), tv.begin(), tv.end());
      local_cols_a_.insert(local_cols_a_.end(), tc.begin(), tc.end());
    }
    local_ptr_a_[i + 1] = local_ptr_a_[i] + nnz;
  }
}

void SoloninVSparseMulCRSMPI::ComputeLocal() {
  int n = static_cast<int>(local_row_ids_.size());
  std::vector<std::vector<double>> rv(n);
  std::vector<std::vector<int>> rc(n);
  local_ptr_c_.resize(n + 1, 0);

  for (int i = 0; i < n; i++) {
    ProcessLocalRow(i, rv[i], rc[i]);
    local_ptr_c_[i + 1] = local_ptr_c_[i] + static_cast<int>(rc[i].size());
  }

  for (int i = 0; i < n; i++) {
    local_vals_c_.insert(local_vals_c_.end(), rv[i].begin(), rv[i].end());
    local_cols_c_.insert(local_cols_c_.end(), rc[i].begin(), rc[i].end());
  }
}

void SoloninVSparseMulCRSMPI::ProcessLocalRow(int local_idx, std::vector<double> &rv,
                                               std::vector<int> &rc) {
  int sa = local_ptr_a_[local_idx];
  int ea = local_ptr_a_[local_idx + 1];
  std::vector<double> tmp(cols_b_count_, 0.0);

  for (int k = sa; k < ea; k++) {
    if (static_cast<size_t>(k) >= local_vals_a_.size()) break;
    double av = local_vals_a_[k];
    int col = local_cols_a_[k];
    if (col < 0 || col >= static_cast<int>(ptr_b_.size()) - 1) continue;
    int sb = ptr_b_[col];
    int eb = ptr_b_[col + 1];
    for (int j = sb; j < eb; j++) {
      if (static_cast<size_t>(j) >= vals_b_.size()) continue;
      int jcol = cols_b_[j];
      if (jcol >= 0 && jcol < cols_b_count_) tmp[jcol] += av * vals_b_[j];
    }
  }

  for (int j = 0; j < cols_b_count_; j++) {
    if (std::abs(tmp[j]) > 1e-12) {
      rv.push_back(tmp[j]);
      rc.push_back(j);
    }
  }
  SortRow(rv, rc);
}

void SoloninVSparseMulCRSMPI::GatherResults() {
  if (rank_ == 0) {
    std::vector<std::vector<double>> all_vals(rows_a_);
    std::vector<std::vector<int>> all_cols(rows_a_);

    for (size_t i = 0; i < local_row_ids_.size(); i++) {
      int gr = local_row_ids_[i];
      int s = local_ptr_c_[i];
      int e = local_ptr_c_[i + 1];
      for (int j = s; j < e; j++) {
        all_vals[gr].push_back(local_vals_c_[j]);
        all_cols[gr].push_back(local_cols_c_[j]);
      }
    }

    for (int src = 1; src < world_size_; src++) CollectFromRank(src, all_vals, all_cols);

    AssembleResult(all_vals, all_cols);
    GetOutput() = std::make_tuple(vals_c_, cols_c_, ptr_c_);
  } else {
    int cnt = static_cast<int>(local_row_ids_.size());
    MPI_Send(&cnt, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    if (cnt > 0) {
      MPI_Send(local_row_ids_.data(), cnt, MPI_INT, 0, 1, MPI_COMM_WORLD);
      MPI_Send(local_ptr_c_.data(), cnt + 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
      int nnz = static_cast<int>(local_vals_c_.size());
      if (nnz > 0) {
        MPI_Send(local_vals_c_.data(), nnz, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD);
        MPI_Send(local_cols_c_.data(), nnz, MPI_INT, 0, 4, MPI_COMM_WORLD);
      }
    }
    GetOutput() = std::make_tuple(std::vector<double>(), std::vector<int>(), std::vector<int>());
  }
}

void SoloninVSparseMulCRSMPI::CollectFromRank(int src, std::vector<std::vector<double>> &all_vals,
                                               std::vector<std::vector<int>> &all_cols) {
  int cnt = 0;
  MPI_Recv(&cnt, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  if (cnt == 0) return;
  std::vector<int> rows(cnt);
  MPI_Recv(rows.data(), cnt, MPI_INT, src, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  std::vector<int> lptr(cnt + 1);
  MPI_Recv(lptr.data(), cnt + 1, MPI_INT, src, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  int nnz = lptr[cnt];
  std::vector<double> lv(nnz);
  std::vector<int> lc(nnz);
  if (nnz > 0) {
    MPI_Recv(lv.data(), nnz, MPI_DOUBLE, src, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(lc.data(), nnz, MPI_INT, src, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  for (int i = 0; i < cnt; i++) {
    int gr = rows[i];
    for (int j = lptr[i]; j < lptr[i + 1]; j++) {
      all_vals[gr].push_back(lv[j]);
      all_cols[gr].push_back(lc[j]);
    }
  }
}

void SoloninVSparseMulCRSMPI::AssembleResult(std::vector<std::vector<double>> &all_vals,
                                              std::vector<std::vector<int>> &all_cols) {
  vals_c_.clear();
  cols_c_.clear();
  ptr_c_.resize(rows_a_ + 1, 0);
  for (int i = 0; i < rows_a_; i++) {
    SortRow(all_vals[i], all_cols[i]);
    vals_c_.insert(vals_c_.end(), all_vals[i].begin(), all_vals[i].end());
    cols_c_.insert(cols_c_.end(), all_cols[i].begin(), all_cols[i].end());
    ptr_c_[i + 1] = ptr_c_[i] + static_cast<int>(all_vals[i].size());
  }
}

void SoloninVSparseMulCRSMPI::SortRow(std::vector<double> &rv, std::vector<int> &rc) {
  if (rc.empty()) return;
  std::vector<std::pair<int, double>> pairs;
  pairs.reserve(rc.size());
  for (size_t i = 0; i < rc.size(); i++) pairs.emplace_back(rc[i], rv[i]);
  std::ranges::sort(pairs);
  static_cast<void>(std::ranges::begin(pairs));
  for (size_t i = 0; i < pairs.size(); i++) {
    rc[i] = pairs[i].first;
    rv[i] = pairs[i].second;
  }
}

bool SoloninVSparseMulCRSMPI::PostProcessingImpl() { return true; }

}  // namespace solonin_v_sparse_matrix_crs
