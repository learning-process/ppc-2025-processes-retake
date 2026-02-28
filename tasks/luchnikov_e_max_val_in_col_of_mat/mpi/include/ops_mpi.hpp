#pragma once

#include <utility>
#include <vector>

#include "luchnikov_e_max_val_in_col_of_mat/common/include/common.hpp"

namespace luchnikov_e_max_val_in_col_of_mat {

class LuchnikovEMaxValInColOfMatMPI {
 public:
  explicit LuchnikovEMaxValInColOfMatMPI(const InType &in);

  void SetTypeOfTask(const std::string &type);
  const InType &GetInput() const;
  InType &GetInput();
  OutType &GetOutput();
  const OutType &GetOutput() const;
  static std::string GetStaticTypeOfTask();

  bool ValidationImpl();
  bool PreProcessingImpl();
  bool RunImpl();
  bool PostProcessingImpl();

 private:
  InType matrix_;
  int rows_ = 0;
  int cols_ = 0;
  int rank_ = 0;
  int size_ = 1;
  std::vector<int> result_;

  std::vector<int> RunSequential() const;
  std::vector<int> PrepareFlatMatrix() const;
  std::pair<std::vector<int>, std::vector<int>> CalculateDistribution() const;
  std::vector<int> ComputeLocalMax(const std::vector<int> &local_flat, int local_rows) const;
};

}  // namespace luchnikov_e_max_val_in_col_of_mat
