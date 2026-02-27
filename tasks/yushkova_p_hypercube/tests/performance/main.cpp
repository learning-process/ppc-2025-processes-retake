#include <gtest/gtest.h>

#include <cstdint>

#include "util/include/perf_test_util.hpp"
#include "yushkova_p_hypercube/common/include/common.hpp"
#include "yushkova_p_hypercube/mpi/include/ops_mpi.hpp"
#include "yushkova_p_hypercube/seq/include/ops_seq.hpp"

namespace yushkova_p_hypercube {
namespace {

OutType ReferenceEdges(InType n) {
  return static_cast<OutType>(n) * (static_cast<std::uint64_t>(1) << (n - 1));
}

}  // namespace

class YushkovaPHypercubePerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 public:
  void SetUp() override {
    input_n_ = 25;
  }

 protected:
  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == ReferenceEdges(input_n_);
  }

  InType GetTestInputData() final {
    return input_n_;
  }

 private:
  InType input_n_{};
};

TEST_F(YushkovaPHypercubePerfTests, SeqPipelineRun) {
  const auto seq_tasks =
      ppc::util::MakePerfTaskTuples<YushkovaPHypercubeSEQ, InType>(PPC_SETTINGS_yushkova_p_hypercube);
  ExecuteTest(std::get<0>(seq_tasks));
}

TEST_F(YushkovaPHypercubePerfTests, SeqTaskRun) {
  const auto seq_tasks =
      ppc::util::MakePerfTaskTuples<YushkovaPHypercubeSEQ, InType>(PPC_SETTINGS_yushkova_p_hypercube);
  ExecuteTest(std::get<1>(seq_tasks));
}

TEST_F(YushkovaPHypercubePerfTests, MpiPipelineRun) {
  const auto mpi_tasks =
      ppc::util::MakePerfTaskTuples<YushkovaPHypercubeMPI, InType>(PPC_SETTINGS_yushkova_p_hypercube);
  ExecuteTest(std::get<0>(mpi_tasks));
}

TEST_F(YushkovaPHypercubePerfTests, MpiTaskRun) {
  const auto mpi_tasks =
      ppc::util::MakePerfTaskTuples<YushkovaPHypercubeMPI, InType>(PPC_SETTINGS_yushkova_p_hypercube);
  ExecuteTest(std::get<1>(mpi_tasks));
}

}  // namespace yushkova_p_hypercube
