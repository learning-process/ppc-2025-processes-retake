#pragma once

#include <cstddef>
#include <vector>

#include "fedoseev_gaussian_method_horizontal_strip_scheme/common/include/common.hpp"
#include "task/include/task.hpp"

namespace fedoseev_gaussian_method_horizontal_strip_scheme {

class FedoseevGaussianMethodHorizontalStripSchemeMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit FedoseevGaussianMethodHorizontalStripSchemeMPI(const InType &input_data);

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  static int ValidateInputData(const InType &input_data);

  static void DistributeRows(const InType &matrix, size_t n, size_t cols, int rank, int size, InType &local_matrix,
                             std::vector<int> &global_to_local_map);

  static bool ForwardEliminationMPI(InType &local_matrix, const std::vector<int> &global_to_local_map, size_t n,
                                    size_t cols, int rank, int size);

  static void EliminateRowsMPI(InType &local_matrix, const std::vector<int> &global_to_local_map, size_t pivot_idx,
                               size_t n, size_t cols, const std::vector<double> &pivot_row);

  static size_t GetGlobalIndex(const std::vector<int> &global_to_local_map, size_t local_idx, size_t n);

  static std::vector<double> BackwardSubstitutionMPI(const InType &local_matrix,
                                                     const std::vector<int> &global_to_local_map, size_t n, size_t cols,
                                                     int rank, int size);
};

}  // namespace fedoseev_gaussian_method_horizontal_strip_scheme
