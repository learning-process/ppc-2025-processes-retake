#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "performance/include/performance.hpp"
#include "util/include/util.hpp"
#include "yushkova_p_min_in_matrix/common/include/common.hpp"
#include "yushkova_p_min_in_matrix/mpi/include/ops_mpi.hpp"
#include "yushkova_p_min_in_matrix/seq/include/ops_seq.hpp"

namespace yushkova_p_min_in_matrix {

namespace {

InType GenerateValue(int64_t i, int64_t j) {
  constexpr int64_t kA = 1103515245LL;
  constexpr int64_t kC = 12345LL;
  constexpr int64_t kM = 2147483648LL;

  int64_t seed = ((i % kM) * (100000007LL % kM) + (j % kM) * (1000000009LL % kM)) % kM;
  seed = (seed ^ 42LL) % kM;
  int64_t val = ((kA % kM) * (seed % kM) + kC) % kM;
  return static_cast<InType>((val % 2000001LL) - 1000000LL);
}

OutType BuildExpectedOutput(InType n) {
  OutType expected;
  expected.reserve(static_cast<size_t>(n));
  for (InType i = 0; i < n; ++i) {
    InType row_min = GenerateValue(static_cast<int64_t>(i), 0);
    for (InType j = 1; j < n; ++j) {
      row_min = std::min(row_min, GenerateValue(static_cast<int64_t>(i), static_cast<int64_t>(j)));
    }
    expected.push_back(row_min);
  }
  return expected;
}

ppc::performance::PerfAttr MakePerfAttr() {
  ppc::performance::PerfAttr attr;
  attr.num_running = 3;

  const auto t0 = std::chrono::high_resolution_clock::now();
  attr.current_timer = [t0] {
    const auto now = std::chrono::high_resolution_clock::now();
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now - t0).count();
    return static_cast<double>(ns) * 1e-9;
  };
  return attr;
}

bool IsRootOrSingleProcess() {
  if (!ppc::util::IsUnderMpirun()) {
    return true;
  }

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  return rank == 0;
}

template <typename TaskType>
OutType RunPerfPipeline(InType n, const std::string &test_id) {
  auto task = std::make_shared<TaskType>(n);
  ppc::performance::Perf<InType, OutType> perf(task);
  const ppc::performance::PerfAttr attr = MakePerfAttr();
  perf.PipelineRun(attr);
  if (IsRootOrSingleProcess()) {
    perf.PrintPerfStatistic(test_id);
  }
  return task->GetOutput();
}

template <typename TaskType>
OutType RunPerfTask(InType n, const std::string &test_id) {
  auto task = std::make_shared<TaskType>(n);
  ppc::performance::Perf<InType, OutType> perf(task);
  const ppc::performance::PerfAttr attr = MakePerfAttr();
  perf.TaskRun(attr);
  if (IsRootOrSingleProcess()) {
    perf.PrintPerfStatistic(test_id);
  }
  return task->GetOutput();
}

void CheckOutputForN(InType n, const OutType &output) {
  const OutType expected = BuildExpectedOutput(n);
  ASSERT_EQ(output.size(), expected.size());
  EXPECT_EQ(output, expected);
}

}  // namespace

TEST(YushkovaMinMatrixPerf, SeqPipelineRun) {
  constexpr InType kN = 512;
  const OutType output = RunPerfPipeline<YushkovaPMinInMatrixSEQ>(kN, "yushkova_p_min_in_matrix_seq_pipeline");
  CheckOutputForN(kN, output);
}

TEST(YushkovaMinMatrixPerf, SeqTaskRun) {
  constexpr InType kN = 512;
  const OutType output = RunPerfTask<YushkovaPMinInMatrixSEQ>(kN, "yushkova_p_min_in_matrix_seq_task");
  CheckOutputForN(kN, output);
}

TEST(YushkovaMinMatrixPerf, MpiPipelineRun) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }

  constexpr InType kN = 512;
  const OutType output = RunPerfPipeline<YushkovaPMinInMatrixMPI>(kN, "yushkova_p_min_in_matrix_mpi_pipeline");
  CheckOutputForN(kN, output);
}

TEST(YushkovaMinMatrixPerf, MpiTaskRun) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }

  constexpr InType kN = 512;
  const OutType output = RunPerfTask<YushkovaPMinInMatrixMPI>(kN, "yushkova_p_min_in_matrix_mpi_task");
  CheckOutputForN(kN, output);
}

}  // namespace yushkova_p_min_in_matrix
