#pragma once

#include <cstddef>
#include <vector>

#include "luchnikov_e_gener_transm_from_all_to_one_gather/common/include/common.hpp"
#include "task/include/task.hpp"

namespace luchnikov_e_gener_transm_from_all_to_one_gather {

class LuchnikovEGenerTransmFromAllToOneGatherSequential : public BaseTask {
 public:
  static constexpr auto GetStaticTypeOfTask() -> ppc::task::TypeOfTask {
    return ppc::task::TypeOfTask::kSequential;
  }

  explicit LuchnikovEGenerTransmFromAllToOneGatherSequential(const InType &input);

  LuchnikovEGenerTransmFromAllToOneGatherSequential(const LuchnikovEGenerTransmFromAllToOneGatherSequential &) = delete;
  auto operator=(const LuchnikovEGenerTransmFromAllToOneGatherSequential &)
      -> LuchnikovEGenerTransmFromAllToOneGatherSequential & = delete;
  LuchnikovEGenerTransmFromAllToOneGatherSequential(LuchnikovEGenerTransmFromAllToOneGatherSequential &&) = delete;
  auto operator=(LuchnikovEGenerTransmFromAllToOneGatherSequential &&)
      -> LuchnikovEGenerTransmFromAllToOneGatherSequential & = delete;
  ~LuchnikovEGenerTransmFromAllToOneGatherSequential() override = default;

 private:
  auto ValidationImpl() -> bool override;
  auto PreProcessingImpl() -> bool override;
  auto RunImpl() -> bool override;
  auto PostProcessingImpl() -> bool override;
};

}  // namespace luchnikov_e_gener_transm_from_all_to_one_gather
