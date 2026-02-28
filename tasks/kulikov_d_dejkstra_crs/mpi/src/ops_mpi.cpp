#include "kulikov_d_dejkstra_crs/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

#include "kulikov_d_dejkstra_crs/common/include/common.hpp"

namespace kulikov_d_dejkstra_crs {

namespace {

struct VertexRange {
  int start = 0;
  int end = 0;
  int count = 0;

  VertexRange() = default;
  VertexRange(int s, int e) : start(s), end(e), count(e - s) {}
};

VertexRange ComputeLocalRange(int rank, int size, int total_vertices) {
  if (size <= 0 || total_vertices <= 0) {
    return {0, 0};
  }

  const int base = total_vertices / size;
  const int extra = total_vertices % size;
  const int local_count = base + ((rank < extra) ? 1 : 0);

  const int start = (rank < extra) ? (rank * (base + 1)) : ((extra * (base + 1)) + ((rank - extra) * base));

  return {start, start + local_count};
}

struct LocalEdges {
  std::vector<int> columns;
  std::vector<double> weights;
  std::vector<int> offsets;
};

LocalEdges ExtractLocalEdges(const GraphData &graph, const VertexRange &range) {
  LocalEdges result;

  if (range.count <= 0 || range.start >= graph.num_vertices) {
    result.offsets.assign(1, 0);
    return result;
  }

  const int global_edge_start = graph.offsets[range.start];
  const int global_edge_end = graph.offsets[range.end];

  if (global_edge_end > global_edge_start) {
    result.columns.assign(graph.columns.begin() + global_edge_start, graph.columns.begin() + global_edge_end);
    result.weights.assign(graph.values.begin() + global_edge_start, graph.values.begin() + global_edge_end);
  }

  result.offsets.resize(range.count + 1);
  for (int i = 0; i <= range.count; ++i) {
    const int global_v = range.start + i;
    const int off = (global_v <= graph.num_vertices) ? graph.offsets[global_v] : graph.offsets[graph.num_vertices];
    result.offsets[i] = off - global_edge_start;
  }

  return result;
}

void InitializeDistances(std::vector<double> &dist, int total_vertices, int source, int source_owner) {
  dist.assign(total_vertices, std::numeric_limits<double>::infinity());

  if (source_owner >= 0 && source >= 0 && source < total_vertices) {
    dist[source] = 0.0;
  }
}

bool RelaxVertexEdges(double current_dist, const LocalEdges &edges, int local_idx, int total_vertices,
                      std::vector<double> &candidate) {
  if (local_idx + 1 >= static_cast<int>(edges.offsets.size())) {
    return false;
  }

  bool improved = false;
  const int begin = edges.offsets[local_idx];
  const int end = edges.offsets[local_idx + 1];

  for (int i = begin; i < end; ++i) {
    const int to = edges.columns[i];
    const double w = edges.weights[i];

    if (to < 0 || to >= total_vertices) {
      continue;
    }

    const double new_dist = current_dist + w;
    if (new_dist < candidate[to]) {
      candidate[to] = new_dist;
      improved = true;
    }
  }

  return improved;
}

bool PerformLocalRelaxation(const std::vector<double> &current, const LocalEdges &edges, const VertexRange &range,
                            int total_vertices, std::vector<double> &candidate) {
  bool any = false;

  for (int i = 0; i < range.count; ++i) {
    const int v = range.start + i;
    if (std::isinf(current[v])) {
      continue;
    }

    any = RelaxVertexEdges(current[v], edges, i, total_vertices, candidate) || any;
  }

  return any;
}

void SynchronizeDistances(std::vector<double> &global_dist, std::vector<double> &local_buf, MPI_Comm comm) {
  if (global_dist.empty()) {
    return;
  }

  if (local_buf.size() != global_dist.size()) {
    local_buf.resize(global_dist.size());
  }

  MPI_Allreduce(local_buf.data(), global_dist.data(), static_cast<int>(global_dist.size()), MPI_DOUBLE, MPI_MIN, comm);

  std::ranges::copy(global_dist, local_buf.begin());
}

}  // namespace

KulikovDDijkstraCRSMPI::KulikovDDijkstraCRSMPI(const InType &input) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = input;
  GetOutput().clear();
}

bool KulikovDDijkstraCRSMPI::ValidationImpl() {
  const GraphData &g = GetInput();

  if (g.num_vertices <= 0) {
    return false;
  }
  if (g.source_vertex < 0 || g.source_vertex >= g.num_vertices) {
    return false;
  }
  if (g.offsets.size() != static_cast<size_t>(g.num_vertices) + 1) {
    return false;
  }
  if (!g.offsets.empty() && static_cast<size_t>(g.offsets.back()) > g.columns.size()) {
    return false;
  }

  return GetOutput().empty();
}

bool KulikovDDijkstraCRSMPI::PreProcessingImpl() {
  return true;
}

bool KulikovDDijkstraCRSMPI::RunImpl() {
  const GraphData &graph = GetInput();
  const int n = graph.num_vertices;
  const int source = graph.source_vertex;

  if (n <= 0) {
    GetOutput().clear();
    MPI_Barrier(MPI_COMM_WORLD);
    return true;
  }

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const VertexRange range = ComputeLocalRange(rank, size, n);
  LocalEdges local_edges = ExtractLocalEdges(graph, range);

  const bool owns_source = (source >= range.start && source < range.end);

  int local_owner = owns_source ? rank : -1;
  int global_owner = -1;
  MPI_Allreduce(&local_owner, &global_owner, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

  std::vector<double> global_dist;
  InitializeDistances(global_dist, n, source, global_owner);

  if (global_owner >= 0) {
    MPI_Bcast(global_dist.data(), n, MPI_DOUBLE, global_owner, MPI_COMM_WORLD);
  }

  std::vector<double> local_dist(n);
  std::vector<double> candidate_dist(n);

  std::ranges::copy(global_dist, local_dist.begin());
  std::ranges::copy(global_dist, candidate_dist.begin());

  for (int iter = 0; iter < n - 1; ++iter) {
    const int local_improved = PerformLocalRelaxation(local_dist, local_edges, range, n, candidate_dist) ? 1 : 0;

    int global_improved = 0;
    MPI_Allreduce(&local_improved, &global_improved, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    if (global_improved == 0) {
      break;
    }

    SynchronizeDistances(global_dist, candidate_dist, MPI_COMM_WORLD);
    std::ranges::copy(global_dist, local_dist.begin());
  }

  GetOutput() = std::move(global_dist);
  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool KulikovDDijkstraCRSMPI::PostProcessingImpl() {
  return true;
}

}  // namespace kulikov_d_dejkstra_crs
