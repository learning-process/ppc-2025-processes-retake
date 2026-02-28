#include "kulikov_d_dejkstra_crs/seq/include/ops_seq.hpp"

#include <cstddef>
#include <functional>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

#include "kulikov_d_dejkstra_crs/common/include/common.hpp"

namespace kulikov_d_dejkstra_crs {

using Distance = double;
using VertexId = int;
using EdgeIndex = int;

struct QueueEntry {
  Distance distance;
  VertexId vertex;

  QueueEntry(Distance d, VertexId v) : distance(d), vertex(v) {}

  bool operator>(const QueueEntry &other) const {
    return distance > other.distance;
  }
};
namespace {
void RelaxOutgoingEdges(VertexId current_vertex, Distance current_distance, const GraphData &graph,
                        std::vector<Distance> &distances,
                        std::priority_queue<QueueEntry, std::vector<QueueEntry>, std::greater<>> &queue) {
  const EdgeIndex edge_start = graph.offsets[current_vertex];
  const EdgeIndex edge_end = graph.offsets[current_vertex + 1];

  for (EdgeIndex idx = edge_start; idx < edge_end; ++idx) {
    const VertexId neighbor = graph.columns[idx];
    const Distance weight = graph.values[idx];

    const Distance candidate_distance = current_distance + weight;
    if (candidate_distance < distances[neighbor]) {
      distances[neighbor] = candidate_distance;
      queue.emplace(candidate_distance, neighbor);
    }
  }
}
}  // namespace

KulikovDDijkstraCRSSEQ::KulikovDDijkstraCRSSEQ(const InType &input) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = input;
  GetOutput().clear();
}

bool KulikovDDijkstraCRSSEQ::ValidationImpl() {
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
  if (!graph.offsets.empty() && static_cast<std::size_t>(graph.offsets.back()) > graph.columns.size()) {
    return false;
  }

  return GetOutput().empty();
}

bool KulikovDDijkstraCRSSEQ::PreProcessingImpl() {
  return true;
}

bool KulikovDDijkstraCRSSEQ::RunImpl() {
  const GraphData &graph = GetInput();
  const VertexId num_vertices = graph.num_vertices;
  const VertexId source_vertex = graph.source_vertex;

  std::vector<Distance> distances(num_vertices, std::numeric_limits<Distance>::infinity());
  distances[source_vertex] = 0.0;

  std::priority_queue<QueueEntry, std::vector<QueueEntry>, std::greater<>> min_heap;
  min_heap.emplace(0.0, source_vertex);

  std::vector<bool> processed(num_vertices, false);

  while (!min_heap.empty()) {
    const QueueEntry current = min_heap.top();
    min_heap.pop();

    const VertexId current_vertex = current.vertex;
    const Distance current_distance = current.distance;

    if (processed[current_vertex]) {
      continue;
    }
    processed[current_vertex] = true;

    RelaxOutgoingEdges(current_vertex, current_distance, graph, distances, min_heap);
  }

  GetOutput() = std::move(distances);
  return true;
}

bool KulikovDDijkstraCRSSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace kulikov_d_dejkstra_crs
