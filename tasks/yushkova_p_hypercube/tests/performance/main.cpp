#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstddef>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "performance/include/performance.hpp"
#include "util/include/util.hpp"
#include "yushkova_p_hypercube/common/include/common.hpp"
#include "yushkova_p_hypercube/mpi/include/ops_mpi.hpp"
#include "yushkova_p_hypercube/seq/include/ops_seq.hpp"

namespace yushkova_p_hypercube {

namespace {

bool IsPowerOfTwo(int value) {
  return value > 0 && (value & (value - 1)) == 0;
}

bool IsRootOrSingleProcess() {
  if (!ppc::util::IsUnderMpirun()) {
    return true;
  }
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  return rank == 0;
}

int HammingDistance(int lhs, int rhs) {
  return std::popcount(static_cast<unsigned int>(lhs ^ rhs));
}

int RouteIntStep(int bit, int current_owner, int value) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const int next_owner = current_owner ^ (1 << bit);
  const int tag = 1200 + bit;
  if (rank == current_owner) {
    MPI_Send(&value, 1, MPI_INT, next_owner, tag, MPI_COMM_WORLD);
  }
  if (rank == next_owner) {
    MPI_Recv(&value, 1, MPI_INT, current_owner, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  return value;
}

int RouteIntThroughHypercube(int source, int destination, int payload) {
  int world_size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int current_owner = source;
  int value = payload;
  const int route_mask = source ^ destination;
  for (int bit = 0; (1 << bit) < world_size; ++bit) {
    if ((route_mask & (1 << bit)) == 0) {
      continue;
    }
    value = RouteIntStep(bit, current_owner, value);
    current_owner ^= (1 << bit);
  }

  MPI_Bcast(&value, 1, MPI_INT, destination, MPI_COMM_WORLD);
  return value;
}

double MeasureRouteLatency(int source, int destination, int repeats) {
  MPI_Barrier(MPI_COMM_WORLD);
  const double t0 = MPI_Wtime();
  for (int i = 0; i < repeats; ++i) {
    (void)RouteIntThroughHypercube(source, destination, i + 1);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  return (MPI_Wtime() - t0) / static_cast<double>(repeats);
}

double RunNeighborPairTraffic() {
  int rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const int partner = rank ^ 1;
  int received = -1;
  const int send_value = rank * 10 + 1;

  MPI_Barrier(MPI_COMM_WORLD);
  const double t0 = MPI_Wtime();
  if (partner < world_size) {
    if ((rank & 1) == 0) {
      MPI_Send(&send_value, 1, MPI_INT, partner, 2000, MPI_COMM_WORLD);
      MPI_Recv(&received, 1, MPI_INT, partner, 2001, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
      MPI_Recv(&received, 1, MPI_INT, partner, 2000, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Send(&send_value, 1, MPI_INT, partner, 2001, MPI_COMM_WORLD);
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);

  if (partner < world_size) {
    EXPECT_EQ(received, partner * 10 + 1);
  }
  return MPI_Wtime() - t0;
}

double HypercubeBroadcastVector(std::vector<int> &data, int root) {
  int rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const int virtual_rank = rank ^ root;

  MPI_Barrier(MPI_COMM_WORLD);
  const double t0 = MPI_Wtime();
  for (int step = 1, level = 0; step < world_size; step <<= 1, ++level) {
    if (virtual_rank < step) {
      const int virtual_partner = virtual_rank + step;
      if (virtual_partner < world_size) {
        const int partner = virtual_partner ^ root;
        const int count = static_cast<int>(data.size());
        MPI_Send(&count, 1, MPI_INT, partner, 2200 + level, MPI_COMM_WORLD);
        if (count > 0) {
          MPI_Send(data.data(), count, MPI_INT, partner, 2300 + level, MPI_COMM_WORLD);
        }
      }
      continue;
    }

    if (virtual_rank < (step << 1)) {
      const int virtual_partner = virtual_rank - step;
      const int partner = virtual_partner ^ root;
      int count = 0;
      MPI_Recv(&count, 1, MPI_INT, partner, 2200 + level, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      data.resize(static_cast<size_t>(count));
      if (count > 0) {
        MPI_Recv(data.data(), count, MPI_INT, partner, 2300 + level, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  return MPI_Wtime() - t0;
}

double MpiBroadcastVector(std::vector<int> &data, int root) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int count = (rank == root) ? static_cast<int>(data.size()) : 0;
  MPI_Barrier(MPI_COMM_WORLD);
  const double t0 = MPI_Wtime();
  MPI_Bcast(&count, 1, MPI_INT, root, MPI_COMM_WORLD);
  if (rank != root) {
    data.resize(static_cast<size_t>(count));
  }
  if (count > 0) {
    MPI_Bcast(data.data(), count, MPI_INT, root, MPI_COMM_WORLD);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  return MPI_Wtime() - t0;
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

OutType RunSeqPerfPipeline(const InType &input, const std::string &test_id) {
  auto task = std::make_shared<YushkovaPHypercubeSEQ>(input);
  ppc::performance::Perf<InType, OutType> perf(task);
  perf.PipelineRun(MakePerfAttr());
  if (IsRootOrSingleProcess()) {
    perf.PrintPerfStatistic(test_id);
  }
  return task->GetOutput();
}

OutType RunSeqPerfTask(const InType &input, const std::string &test_id) {
  auto task = std::make_shared<YushkovaPHypercubeSEQ>(input);
  ppc::performance::Perf<InType, OutType> perf(task);
  perf.TaskRun(MakePerfAttr());
  if (IsRootOrSingleProcess()) {
    perf.PrintPerfStatistic(test_id);
  }
  return task->GetOutput();
}

OutType RunMpiPerfPipeline(const InType &input, const std::string &test_id) {
  auto task = std::make_shared<YushkovaPHypercubeMPI>(input);
  ppc::performance::Perf<InType, OutType> perf(task);
  perf.PipelineRun(MakePerfAttr());
  if (IsRootOrSingleProcess()) {
    perf.PrintPerfStatistic(test_id);
  }
  return task->GetOutput();
}

OutType RunMpiPerfTask(const InType &input, const std::string &test_id) {
  auto task = std::make_shared<YushkovaPHypercubeMPI>(input);
  ppc::performance::Perf<InType, OutType> perf(task);
  perf.TaskRun(MakePerfAttr());
  if (IsRootOrSingleProcess()) {
    perf.PrintPerfStatistic(test_id);
  }
  return task->GetOutput();
}

void CheckPerfOutput(const InType &input, int output) {
  EXPECT_EQ(output, std::get<2>(input));
}

}  // namespace

TEST(HypercubePerformance, HopsAndLatencyGrowLogarithmically) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }

  int world_size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  if (!IsPowerOfTwo(world_size) || world_size < 8) {
    GTEST_SKIP();
  }

  constexpr int kRepeats = 200;
  const double near_latency = MeasureRouteLatency(0, 1, kRepeats);
  const double far_latency = MeasureRouteLatency(0, world_size - 1, kRepeats);

  ASSERT_GT(near_latency, 0.0);
  ASSERT_GT(far_latency, 0.0);
  EXPECT_EQ(HammingDistance(0, 1), 1);
}

TEST(HypercubePerformance, NeighborPairsSaturateNetwork) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }

  int world_size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  if (!IsPowerOfTwo(world_size) || world_size < 2) {
    GTEST_SKIP();
  }

  EXPECT_GT(RunNeighborPairTraffic(), 0.0);
}

TEST(HypercubePerformance, CompareHypercubeBroadcastWithMPIBcast) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }

  int world_size = 0;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (!IsPowerOfTwo(world_size) || world_size < 2) {
    GTEST_SKIP();
  }

  std::vector<int> base(1 << 14);
  if (rank == 0) {
    std::iota(base.begin(), base.end(), 10);
  } else {
    base.clear();
  }

  auto custom_data = base;
  auto mpi_data = base;
  const double custom_time = HypercubeBroadcastVector(custom_data, 0);
  const double mpi_time = MpiBroadcastVector(mpi_data, 0);
  EXPECT_TRUE(custom_data.size() == mpi_data.size() &&
              std::equal(custom_data.begin(), custom_data.end(), mpi_data.begin()) && custom_time > 0.0 &&
              mpi_time > 0.0);
}

TEST(YushkovaPHypercubePerf, SeqPipelineRun) {
  const InType input{0, 1, 42};
  const OutType output = RunSeqPerfPipeline(input, "yushkova_p_hypercube_seq_pipeline");
  CheckPerfOutput(input, output);
}

TEST(YushkovaPHypercubePerf, SeqTaskRun) {
  const InType input{2, 7, -15};
  const OutType output = RunSeqPerfTask(input, "yushkova_p_hypercube_seq_task");
  CheckPerfOutput(input, output);
}

TEST(YushkovaPHypercubePerf, MpiPipelineRun) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }

  int world_size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  const InType input{0, world_size > 0 ? world_size - 1 : 0, 123456};
  const OutType output = RunMpiPerfPipeline(input, "yushkova_p_hypercube_mpi_pipeline");
  CheckPerfOutput(input, output);
}

TEST(YushkovaPHypercubePerf, MpiTaskRun) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }

  int world_size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  const InType input{0, world_size > 0 ? world_size - 1 : 0, 123456};
  const OutType output = RunMpiPerfTask(input, "yushkova_p_hypercube_mpi_task");
  CheckPerfOutput(input, output);
}

}  // namespace yushkova_p_hypercube
