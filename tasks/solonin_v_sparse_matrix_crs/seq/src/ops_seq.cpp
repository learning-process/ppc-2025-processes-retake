#include "solonin_v_sparse_matrix_crs/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <ranges>
#include <tuple>
#include <utility>
#include <vector>

#include "solonin_v_sparse_matrix_crs/common/include/common.hpp"

namespace solonin_v_sparse_matrix_crs {

SoloninVSparseMulCRSSEQ::SoloninVSparseMulCRSSEQ(InType in)
    : input_(std::move(in)),
      rows_a_(std::get<6>(input_)),
      cols_a_count_(std::get<7>(input_)),
      cols_b_count_(std::get<8>(input_)) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = input_;
  GetOutput() = std::make_tuple(std::vector<double>(), std::vector<int>(), std::vector<int>());
}

bool SoloninVSparseMulCRSSEQ::ValidationImpl() {
  vals_a_ = std::get<0>(input_);
  cols_a_ = std::get<1>(input_);
  ptr_a_ = std::get<2>(input_);
  vals_b_ = std::get<3>(input_);
  cols_b_ = std::get<4>(input_);
  ptr_b_ = std::get<5>(input_);
  rows_a_ = std::get<6>(input_);
  cols_a_count_ = std::get<7>(input_);
  cols_b_count_ = std::get<8>(input_);

  if (rows_a_ <= 0 || cols_a_count_ <= 0 || cols_b_count_ <= 0) {
    return false;
  }
  return ValidateA() && ValidateB();
}

bool SoloninVSparseMulCRSSEQ::ValidateA() const {
  if (ptr_a_.size() != static_cast<size_t>(rows_a_) + 1U) return false;
  if (ptr_a_[0] != 0) return false;
  for (size_t i = 0; i + 1 < ptr_a_.size(); i++) {
    if (ptr_a_[i] > ptr_a_[i + 1]) return false;
  }
  if (vals_a_.size() != cols_a_.size()) return false;
  static_cast<void>(std::ranges::begin(cols_a_));
  return std::ranges::all_of(cols_a_,
                             [n = cols_a_count_](int c) { return c >= 0 && c < n; });
}

bool SoloninVSparseMulCRSSEQ::ValidateB() const {
  int rows_b = cols_a_count_;
  if (ptr_b_.size() != static_cast<size_t>(rows_b) + 1U) return false;
  if (ptr_b_[0] != 0) return false;
  for (size_t i = 0; i + 1 < ptr_b_.size(); i++) {
    if (ptr_b_[i] > ptr_b_[i + 1]) return false;
  }
  if (vals_b_.size() != cols_b_.size()) return false;
  static_cast<void>(std::ranges::begin(cols_b_));
  return std::ranges::all_of(cols_b_,
                             [n = cols_b_count_](int c) { return c >= 0 && c < n; });
}

bool SoloninVSparseMulCRSSEQ::PreProcessingImpl() {
  vals_c_.clear();
  cols_c_.clear();
  ptr_c_.clear();
  return true;
}

bool SoloninVSparseMulCRSSEQ::RunImpl() {
  ptr_c_.resize(rows_a_ + 1, 0);
  ptr_c_[0] = 0;

  std::vector<std::vector<double>> row_vals(rows_a_);
  std::vector<std::vector<int>> row_cols(rows_a_);

  for (int i = 0; i < rows_a_; i++) {
    MultiplyRow(i, row_vals[i], row_cols[i]);
    ptr_c_[i + 1] = ptr_c_[i] + static_cast<int>(row_cols[i].size());
  }

  for (int i = 0; i < rows_a_; i++) {
    vals_c_.insert(vals_c_.end(), row_vals[i].begin(), row_vals[i].end());
    cols_c_.insert(cols_c_.end(), row_cols[i].begin(), row_cols[i].end());
  }

  return true;
}

void SoloninVSparseMulCRSSEQ::MultiplyRow(int row_idx, std::vector<double> &row_vals,
                                          std::vector<int> &row_cols) {
  int start_a = ptr_a_[row_idx];
  int end_a = ptr_a_[row_idx + 1];

  std::vector<double> tmp(cols_b_count_, 0.0);

  for (int k = start_a; k < end_a; k++) {
    double a_val = vals_a_[k];
    int col = cols_a_[k];
    int start_b = ptr_b_[col];
    int end_b = ptr_b_[col + 1];
    for (int j = start_b; j < end_b; j++) {
      tmp[cols_b_[j]] += a_val * vals_b_[j];
    }
  }

  for (int j = 0; j < cols_b_count_; j++) {
    if (std::abs(tmp[j]) > 1e-12) {
      row_vals.push_back(tmp[j]);
      row_cols.push_back(j);
    }
  }

  if (!row_cols.empty()) {
    std::vector<std::pair<int, double>> pairs;
    pairs.reserve(row_cols.size());
    for (size_t i = 0; i < row_cols.size(); i++) {
      pairs.emplace_back(row_cols[i], row_vals[i]);
    }
    std::ranges::sort(pairs);
    static_cast<void>(std::ranges::begin(pairs));
    for (size_t i = 0; i < pairs.size(); i++) {
      row_cols[i] = pairs[i].first;
      row_vals[i] = pairs[i].second;
    }
  }
}

bool SoloninVSparseMulCRSSEQ::PostProcessingImpl() {
  GetOutput() = std::make_tuple(vals_c_, cols_c_, ptr_c_);
  return true;
}

}  // namespace solonin_v_sparse_matrix_crs
