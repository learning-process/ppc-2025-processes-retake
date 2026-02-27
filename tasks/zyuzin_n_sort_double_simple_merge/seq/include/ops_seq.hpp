#pragma once

#include <cstdint>
#include <vector>

#include "task/include/task.hpp"
#include "zyuzin_n_sort_double_simple_merge/common/include/common.hpp"

namespace zyuzin_n_sort_double_simple_merge {

class ZyuzinNSortDoubleWithSimpleMergeSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit ZyuzinNSortDoubleWithSimpleMergeSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static std::vector<std::uint64_t> ConvertDoublesToBits(const std::vector<double> &data);
  static std::vector<std::uint64_t> SortBits(const std::vector<std::uint64_t> &bits);
  static std::vector<double> ConvertBitsToDoubles(const std::vector<std::uint64_t> &data);
  static std::vector<std::uint64_t> SimpleMerge(const std::vector<std::uint64_t> &all_bits,
                                                const std::vector<int> &counts, const std::vector<int> &displs);
};

}  // namespace zyuzin_n_sort_double_simple_merge
