#include "luchnikov_e_gener_transm_from_all_to_one_gather/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "luchnikov_e_gener_transm_from_all_to_one_gather/common/include/common.hpp"
#include "util/include/util.hpp"

namespace luchnikov_e_gener_transm_from_all_to_one_gather {

LuchnikovEGenerTransformFromAllToOneGatherSEQ::
    LuchnikovEGenerTransformFromAllToOneGatherSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{0};
}

bool LuchnikovEGenerTransformFromAllToOneGatherSEQ::ValidationImpl() {
  return (GetInput() > InType{0}) && (GetOutput() == OutType{0});
}

bool LuchnikovEGenerTransformFromAllToOneGatherSEQ::PreProcessingImpl() {
  GetOutput() = OutType{2} * GetInput();
  return GetOutput() > OutType{0};
}

bool LuchnikovEGenerTransformFromAllToOneGatherSEQ::RunImpl() {
  if (GetInput() == InType{0}) {
    return false;
  }
  for (InType i = InType{0}; i < GetInput(); i++) {
    for (InType j = InType{0}; j < GetInput(); j++) {
      for (InType k = InType{0}; k < GetInput(); k++) {
        std::vector<InType> tmp(static_cast<std::size_t>(i + j + k), InType{1});
        GetOutput() += std::accumulate(tmp.begin(), tmp.end(), InType{0});
        GetOutput() -= (i + j + k);
      }
    }
  }
  const int num_threads = ppc::util::GetNumThreads();
  GetOutput() *= static_cast<OutType>(num_threads);
  int counter = InType{0};
  for (int i = InType{0}; i < num_threads; i++) {
    counter++;
  }
  if (counter != InType{0}) {
    GetOutput() /= static_cast<OutType>(counter);
  }
  return GetOutput() > OutType{0};
}

bool LuchnikovEGenerTransformFromAllToOneGatherSEQ::PostProcessingImpl() {
  GetOutput() -= GetInput();
  return GetOutput() > OutType{0};
}

}  // namespace luchnikov_e_gener_transm_from_all_to_one_gather