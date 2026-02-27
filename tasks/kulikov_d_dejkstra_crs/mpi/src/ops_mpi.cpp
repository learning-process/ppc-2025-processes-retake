#include "kulikov_d_dejkstra_crs/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <ranges>
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

  const int base_count = total_vertices / size;
  const int extra = total_vertices % size;
  const int local_count = base_count + (rank < extra ? 1 : 0);

  const int start =
      (rank < extra) ? rank * (base_count + 1) : (extra * (base_count + 1)) + ((rank - extra) * base_count);

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
  const int local_edge_count = global_edge_end - global_edge_start;

  if (local_edge_count > 0) {
    result.columns.assign(graph.columns.begin() + global_edge_start, graph.columns.begin() + global_edge_end);
    result.weights.assign(graph.values.begin() + global_edge_start, graph.values.begin() + global_edge_end);
  }

  result.offsets.resize(range.count + 1);
  for (int i = 0; i <= range.count; ++i) {
    const int global_idx = range.start + i;
    const int global_offset =
        (global_idx <= graph.num_vertices) ? graph.offsets[global_idx] : graph.offsets[graph.num_vertices];
    result.offsets[i] = global_offset - global_edge_start;
  }

  return result;
}

void InitializeDistances(std::vector<double> &distances, int total_vertices, int source, int source_owner_rank) {
  distances.assign(total_vertices, std::numeric_limits<double>::infinity());

  if (source_owner_rank >= 0 && source >= 0 && source < total_vertices) {
    distances[source] = 0.0;
  }
}

bool RelaxVertexEdges(double current_dist, const LocalEdges &local_edges, int local_index, int total_vertices,
                      std::vector<double> &candidate_dist) {
  if (local_index < 0 || local_index + 1 >= static_cast<int>(local_edges.offsets.size())) {
    return false;
  }

  bool improved = false;
  const int edge_start = local_edges.offsets[local_index];
  const int edge_end = local_edges.offsets[local_index + 1];

  for (int edge_idx = edge_start; edge_idx < edge_end; ++edge_idx) {
    const int neighbor = local_edges.columns[edge_idx];
    const double weight = local_edges.weights[edge_idx];

    if (neighbor < 0 || neighbor >= total_vertices) {
      continue;
    }

    const double new_dist = current_dist + weight;
    if (new_dist < candidate_dist[neighbor]) {
      candidate_dist[neighbor] = new_dist;
      improved = true;
    }
  }

  return improved;
}

bool PerformLocalRelaxation(const std::vector<double> &current_dist, const LocalEdges &local_edges,
                            const VertexRange &range, int total_vertices, std::vector<double> &candidate_dist) {
  bool any_improved = false;

  for (int local_idx = 0; local_idx < range.count; ++local_idx) {
    const int global_v = range.start + local_idx;
    const double dist = current_dist[global_v];

    if (std::isinf(dist)) {
      continue;
    }

    any_improved = RelaxVertexEdges(dist, local_edges, local_idx, total_vertices, candidate_dist) || any_improved;
  }

  return any_improved;
}

void SynchronizeDistances(std::vector<double> &global_dist, std::vector<double> &local_buffer, MPI_Comm comm) {
  if (global_dist.empty()) {
    return;
  }

  MPI_Allreduce(local_buffer.data(), global_dist.data(), static_cast<int>(global_dist.size()), MPI_DOUBLE, MPI_MIN,
                comm);
  local_buffer = global_dist;
}

}  // namespace

KulikovDDijkstraCRSMPI::KulikovDDijkstraCRSMPI(const InType &input) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = input;
  GetOutput().clear();
}

bool KulikovDDijkstraCRSMPI::ValidationImpl() {
  const GraphData &graph = GetInput();

  if (graph.num_vertices <= 0) {
    return false;
  }
  if (graph.source_vertex < 0 || graph.source_vertex >= graph.num_vertices) {
    return false;
  }
  if (graph.offsets.size() != static_cast<size_t>(graph.num_vertices) + 1) {
    return false;
  }
  if (!graph.offsets.empty() && std::cmp_greater(graph.offsets.back(), graph.columns.size())) {
    return false;
  }

  return GetOutput().empty();
}

bool KulikovDDijkstraCRSMPI::PreProcessingImpl() {
  return true;
}

bool KulikovDDijkstraCRSMPI::RunImpl() {
  const GraphData &graph = GetInput();
  const int total_vertices = graph.num_vertices;
  const int source = graph.source_vertex;

  if (total_vertices <= 0) {
    GetOutput().clear();
    MPI_Barrier(MPI_COMM_WORLD);
    return true;
  }

  int mpi_rank = 0;
  int mpi_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  const VertexRange local_range = ComputeLocalRange(mpi_rank, mpi_size, total_vertices);

  LocalEdges local_edges = ExtractLocalEdges(graph, local_range);

  const bool owns_source = (source >= local_range.start && source < local_range.end);

  int source_owner = owns_source ? mpi_rank : -1;

  int global_source_owner = -1;
  MPI_Allreduce(&source_owner, &global_source_owner, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

  std::vector<double> global_dist;
  InitializeDistances(global_dist, total_vertices, source, global_source_owner);

  if (global_source_owner >= 0) {
    MPI_Bcast(global_dist.data(), total_vertices, MPI_DOUBLE, global_source_owner, MPI_COMM_WORLD);
  }

  std::vector<double> local_dist = global_dist;
  std::vector<double> candidate_dist = global_dist;

  for (int iteration = 0; iteration < total_vertices - 1; ++iteration) {
    int local_improved_int =
        PerformLocalRelaxation(local_dist, local_edges, local_range, total_vertices, candidate_dist) ? 1 : 0;

    int global_improved = 0;
    MPI_Allreduce(&local_improved_int, &global_improved, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    if (global_improved == 0) {
      break;
    }

    SynchronizeDistances(global_dist, candidate_dist, MPI_COMM_WORLD);
    local_dist = global_dist;
  }

  GetOutput() = std::move(global_dist);
  MPI_Barrier(MPI_COMM_WORLD);

  return true;
}

bool KulikovDDijkstraCRSMPI::PostProcessingImpl() {
  return true;
}

}  // namespace kulikov_d_dejkstra_crs
