#pragma once

#include <cstdint>
#include <vector>

#include "task/include/task.hpp"
#include "zyuzin_n_sort_double_simple_merge/common/include/common.hpp"

namespace zyuzin_n_sort_double_simple_merge {

class ZyuzinNSortDoubleWithSimpleMergeMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit ZyuzinNSortDoubleWithSimpleMergeMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void ConvertDoublesToBits(int rank, const std::vector<double> &data, std::vector<std::uint64_t> &bits_data);
  static std::vector<std::uint64_t> SortBits(const std::vector<std::uint64_t> &bits);
  static std::vector<double> ConvertBitsToDoubles(const std::vector<std::uint64_t> &data);
  static std::vector<std::uint64_t> MergeSegments(const std::vector<std::uint64_t> &local_sorted_data, int rank,
                                                  int size);
  static std::vector<std::uint64_t> MergeTwoVectors(const std::vector<std::uint64_t> &a,
                                                    const std::vector<std::uint64_t> &b);
};

}  // namespace zyuzin_n_sort_double_simple_merge
