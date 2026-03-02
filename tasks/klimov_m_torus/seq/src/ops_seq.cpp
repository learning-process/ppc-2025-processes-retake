#include "klimov_m_torus/seq/include/ops_seq.hpp"

#include <vector>

#include "klimov_m_torus/common/include/common.hpp"

namespace klimov_m_torus {

TorusReferenceImpl::TorusReferenceImpl(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool TorusReferenceImpl::ValidationImpl() {
  const auto &req = GetInput();
  return req.sender >= 0 && req.receiver >= 0;
}

bool TorusReferenceImpl::PreProcessingImpl() {
  return true;
}

bool TorusReferenceImpl::RunImpl() {
  const auto &req = GetInput();
  auto &out = GetOutput();

  out.received_data = req.data;

  out.route.clear();
  out.route.push_back(req.sender);
  if (req.sender != req.receiver) {
    out.route.push_back(req.receiver);
  }

  return true;
}

bool TorusReferenceImpl::PostProcessingImpl() {
  return true;
}

}  // namespace klimov_m_torus
