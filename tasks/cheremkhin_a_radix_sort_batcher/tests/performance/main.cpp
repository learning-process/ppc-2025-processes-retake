#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>

#include "cheremkhin_a_radix_sort_batcher/common/include/common.hpp"
#include "cheremkhin_a_radix_sort_batcher/mpi/include/ops_mpi.hpp"
#include "cheremkhin_a_radix_sort_batcher/seq/include/ops_seq.hpp"
#include "performance/include/performance.hpp"
#include "util/include/perf_test_util.hpp"

namespace cheremkhin_a_radix_sort_batcher {

namespace {

InType MakeInput(std::size_t n) {
  InType in;
  in.resize(n);

  std::uint32_t x = 123456789U;
  for (std::size_t i = 0; i < n; ++i) {
    x = (1103515245U * x) + 12345U;
    const std::int32_t v = static_cast<std::int32_t>(x ^ (x >> 16)) - static_cast<std::int32_t>(1U << 30);
    in[i] = static_cast<int>(v);
  }

  return in;
}

}  // namespace

class RadixSortBatcherRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  // Keep the test heavy enough but still runnable for multiple MPI counts.
  static constexpr std::size_t kN = 6000000U;

  InType input_data_;
  OutType correct_answer_;
  int rank_ = 0;
  bool mpi_inited_ = false;

  void SetUp() override {
    int inited = 0;
    MPI_Initialized(&inited);
    mpi_inited_ = (inited != 0);
    if (mpi_inited_) {
      MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    } else {
      rank_ = 0;
    }

    if (!mpi_inited_ || rank_ == 0) {
      input_data_ = MakeInput(kN);
      correct_answer_ = input_data_;
      std::ranges::sort(correct_answer_);
    } else {
      input_data_.clear();
      correct_answer_.clear();
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (mpi_inited_ && rank_ != 0) {
      return true;
    }
    return output_data == correct_answer_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

  void SetPerfAttributes(ppc::performance::PerfAttr &perf_attrs) override {
    const auto t0 = std::chrono::high_resolution_clock::now();
    perf_attrs.current_timer = [t0] {
      auto now = std::chrono::high_resolution_clock::now();
      auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now - t0).count();
      return static_cast<double>(ns) * 1e-9;
    };
    perf_attrs.num_running = 1;
  }
};

TEST_P(RadixSortBatcherRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, CheremkhinARadixSortBatcherMPI, CheremkhinARadixSortBatcherSEQ>(
        PPC_SETTINGS_cheremkhin_a_radix_sort_batcher);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = RadixSortBatcherRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, RadixSortBatcherRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace cheremkhin_a_radix_sort_batcher
