#include "klimov_m_torus/seq/include/ops_seq.hpp"

#include <vector>

#include "klimov_m_torus/common/include/common.hpp"

namespace klimov_m_torus {

TorusSequential::TorusSequential(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool TorusSequential::ValidationImpl() {
  const auto &req = GetInput();
  return req.source >= 0 && req.dest >= 0;
}

bool TorusSequential::PreProcessingImpl() {
  return true;
}

bool TorusSequential::RunImpl() {
  const auto &req = GetInput();
  auto &out = GetOutput();

  out.payload = req.payload;

  out.path.clear();
  out.path.push_back(req.source);
  if (req.source != req.dest) {
    out.path.push_back(req.dest);
  }

  return true;
}

bool TorusSequential::PostProcessingImpl() {
  return true;
}

}  // namespace klimov_m_torus