#include <gtest/gtest.h>
#include <mpi.h>

#include <bit>
#include <cstddef>
#include <cstring>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "util/include/util.hpp"
#include "yushkova_p_hypercube/common/include/common.hpp"
#include "yushkova_p_hypercube/mpi/include/ops_mpi.hpp"
#include "yushkova_p_hypercube/seq/include/ops_seq.hpp"

namespace yushkova_p_hypercube {

namespace {

bool IsPowerOfTwo(int value) {
  return value > 0 && (value & (value - 1)) == 0;
}

int HammingDistance(int lhs, int rhs) {
  return std::popcount(static_cast<unsigned int>(lhs ^ rhs));
}

bool CanSendDirectlyInHypercube(int from, int to) {
  return HammingDistance(from, to) == 1;
}

bool IsValidHypercubeWorld(int min_world_size) {
  int world_size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  return IsPowerOfTwo(world_size) && world_size >= min_world_size;
}

int RouteIntStep(int bit, int current_owner, int value) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const int next_owner = current_owner ^ (1 << bit);
  const int tag = 500 + bit;
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

void SendVectorToNeighbor(const std::vector<int>& data, int next_owner, int bit) {
  const int count = static_cast<int>(data.size());
  MPI_Send(&count, 1, MPI_INT, next_owner, 700 + bit, MPI_COMM_WORLD);
  if (count > 0) {
    MPI_Send(data.data(), count, MPI_INT, next_owner, 900 + bit, MPI_COMM_WORLD);
  }
}

std::vector<int> ReceiveVectorFromNeighbor(int current_owner, int bit) {
  int count = 0;
  MPI_Recv(&count, 1, MPI_INT, current_owner, 700 + bit, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  std::vector<int> data(static_cast<size_t>(count));
  if (count > 0) {
    MPI_Recv(data.data(), count, MPI_INT, current_owner, 900 + bit, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  return data;
}

std::vector<int> BroadcastVectorFromDestination(std::vector<int> data, int destination) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int final_count = 0;
  if (rank == destination) {
    final_count = static_cast<int>(data.size());
  }
  MPI_Bcast(&final_count, 1, MPI_INT, destination, MPI_COMM_WORLD);

  if (rank != destination) {
    data.resize(static_cast<size_t>(final_count));
  }
  if (final_count > 0) {
    MPI_Bcast(data.data(), final_count, MPI_INT, destination, MPI_COMM_WORLD);
  }

  return data;
}

std::vector<int> RouteVectorThroughHypercube(int source, int destination, const std::vector<int>& payload) {
  int rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  std::vector<int> data = (rank == source) ? payload : std::vector<int>{};
  int current_owner = source;
  const int route_mask = source ^ destination;

  for (int bit = 0; (1 << bit) < world_size; ++bit) {
    if ((route_mask & (1 << bit)) == 0) {
      continue;
    }

    const int next_owner = current_owner ^ (1 << bit);
    if (rank == current_owner) {
      SendVectorToNeighbor(data, next_owner, bit);
    }
    if (rank == next_owner) {
      data = ReceiveVectorFromNeighbor(current_owner, bit);
    }

    current_owner = next_owner;
  }

  return BroadcastVectorFromDestination(std::move(data), destination);
}

template <typename TaskType>
bool RunPipeline(TaskType& task) {
  return task.Validation() && task.PreProcessing() && task.Run() && task.PostProcessing();
}

OutType RunSeqTask(const InType& input) {
  auto task = std::make_shared<YushkovaPHypercubeSEQ>(input);
  if (!RunPipeline(*task)) {
    return 0;
  }
  return task->GetOutput();
}

OutType RunMpiTask(const InType& input) {
  auto task = std::make_shared<YushkovaPHypercubeMPI>(input);
  if (!RunPipeline(*task)) {
    return 0;
  }
  return task->GetOutput();
}

void CheckRouteForBothImplementations(const InType& input, int expected) {
  EXPECT_EQ(RunSeqTask(input), expected);
  if (ppc::util::IsUnderMpirun()) {
    EXPECT_EQ(RunMpiTask(input), expected);
  }
}

void CheckDirectNeighborRoute(int to, int payload) {
  EXPECT_TRUE(CanSendDirectlyInHypercube(0, to));
  EXPECT_EQ(RouteIntThroughHypercube(0, to, payload), payload);
  EXPECT_EQ(HammingDistance(0, to), 1);
}

}  // namespace

TEST(HypercubeFunctionalTopology, DirectNeighborZeroToOneIsOneHop) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }
  if (!IsValidHypercubeWorld(8)) {
    GTEST_SKIP();
  }
  CheckDirectNeighborRoute(1, 111);
}

TEST(HypercubeFunctionalTopology, DirectNeighborZeroToTwoIsOneHop) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }
  if (!IsValidHypercubeWorld(8)) {
    GTEST_SKIP();
  }
  CheckDirectNeighborRoute(2, 222);
}

TEST(HypercubeFunctionalTopology, DirectNeighborZeroToFourIsOneHop) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }
  if (!IsValidHypercubeWorld(8)) {
    GTEST_SKIP();
  }
  CheckDirectNeighborRoute(4, 333);
}

TEST(HypercubeFunctionalTopology, RoutingToFarNodeWorks) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }
  if (!IsValidHypercubeWorld(8)) {
    GTEST_SKIP();
  }

  EXPECT_EQ(RouteIntThroughHypercube(0, 7, 31415), 31415);
  EXPECT_EQ(HammingDistance(0, 7), 3);
}

TEST(HypercubeFunctionalTopology, NonNeighborDirectLinkIsRejectedAndRouted) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }
  if (!IsValidHypercubeWorld(8)) {
    GTEST_SKIP();
  }

  EXPECT_FALSE(CanSendDirectlyInHypercube(0, 7));
  EXPECT_EQ(RouteIntThroughHypercube(0, 7, -77), -77);
}

TEST(HypercubeFunctionalTopology, DataIntegrityForLargeVector) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }
  if (!IsValidHypercubeWorld(8)) {
    GTEST_SKIP();
  }

  std::vector<int> payload(1 << 15);
  std::iota(payload.begin(), payload.end(), -5000);
  const auto delivered = RouteVectorThroughHypercube(0, 7, payload);

  ASSERT_EQ(delivered.size(), payload.size());
  EXPECT_EQ(std::memcmp(delivered.data(), payload.data(), payload.size() * sizeof(int)), 0);
}

TEST(HypercubeFunctionalTopology, SelfSendKeepsPayload) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }
  if (!IsValidHypercubeWorld(1)) {
    GTEST_SKIP();
  }

  constexpr int kPayload = 2026;
  EXPECT_EQ(RouteIntThroughHypercube(0, 0, kPayload), kPayload);
}

TEST(HypercubeFunctionalTopology, RouteBetweenMiddleProcessesWorks) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }
  if (!IsValidHypercubeWorld(8)) {
    GTEST_SKIP();
  }

  constexpr int kSource = 3;
  constexpr int kDestination = 5;
  constexpr int kPayload = -1234;

  EXPECT_EQ(RouteIntThroughHypercube(kSource, kDestination, kPayload), kPayload);
  EXPECT_EQ(HammingDistance(kSource, kDestination), 2);
}

TEST(HypercubeFunctionalTopology, EmptyVectorIntegrity) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }
  if (!IsValidHypercubeWorld(2)) {
    GTEST_SKIP();
  }

  EXPECT_TRUE(RouteVectorThroughHypercube(0, 1, {}).empty());
}

TEST(HypercubeFunctionalTopology, ValidationDependsOnPowerOfTwoWorldSize) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }

  int world_size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  {
    YushkovaPHypercubeMPI task(InType{0, 0, 1});
    EXPECT_EQ(task.Validation(), IsPowerOfTwo(world_size));
  }
  ppc::util::DestructorFailureFlag::Unset();
}

TEST(YushkovaPHypercubeFunctional, SeqRouteCases) {
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (mpi_initialized == 0) {
    GTEST_SKIP();
  }
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }
  CheckRouteForBothImplementations(InType{0, 1, 42}, 42);
  CheckRouteForBothImplementations(InType{2, 7, -15}, -15);
  CheckRouteForBothImplementations(InType{5, 3, 2025}, 2025);
}

}  // namespace yushkova_p_hypercube
