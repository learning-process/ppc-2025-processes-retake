#include "tsarkov_k_hypercube/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "tsarkov_k_hypercube/common/include/common.hpp"

namespace tsarkov_k_hypercube {
namespace {

[[nodiscard]] bool IsPowerOfTwo(const int value) {
  if (value <= 0) {
    return false;
  }
  return (value & (value - 1)) == 0;
}

[[nodiscard]] int CalcDimensions(const int world_size) {
  int dimensions = 0;
  int tmp_size = world_size;
  while (tmp_size > 1) {
    tmp_size >>= 1;
    dimensions++;
  }
  return dimensions;
}

[[nodiscard]] bool HasValidInputShape(const InType &input_data) {
  return input_data.size() == 3U;
}

[[nodiscard]] std::vector<std::int32_t> CreatePayload(const int data_size) {
  std::vector<std::int32_t> payload(static_cast<std::size_t>(data_size));
  for (std::size_t index = 0; index < payload.size(); index++) {
    payload[index] = static_cast<std::int32_t>(index);
  }
  return payload;
}

[[nodiscard]] bool IsWorldAndRanksOk(const int world_size, const int source_rank, const int destination_rank) {
  if (!IsPowerOfTwo(world_size) || world_size <= 1) {
    return false;
  }
  if (source_rank < 0 || destination_rank < 0) {
    return false;
  }
  if (source_rank >= world_size || destination_rank >= world_size) {
    return false;
  }
  return true;
}

void SendrecvCounts(MPI_Comm comm, const int partner_rank, const int tag, const int send_count, int *recv_count) {
  MPI_Status status{};
  MPI_Sendrecv(&send_count, 1, MPI_INT, partner_rank, tag, recv_count, 1, MPI_INT, partner_rank, tag, comm, &status);
}

void SendrecvPayload(MPI_Comm comm, const int partner_rank, const int tag, const std::int32_t *send_ptr,
                     const int send_count, std::int32_t *recv_ptr, const int recv_count) {
  MPI_Status status{};
  MPI_Sendrecv(send_ptr, send_count, MPI_INT, partner_rank, tag, recv_ptr, recv_count, MPI_INT, partner_rank, tag, comm,
               &status);
}

void InitPayloadIfSource(const int world_rank, const int source_rank, const int data_size,
                         std::vector<std::int32_t> *payload_buffer, bool *has_payload) {
  if (world_rank != source_rank) {
    return;
  }
  *payload_buffer = CreatePayload(data_size);
  *has_payload = true;
}

[[nodiscard]] bool RouteOneDimension(const int world_rank, const int destination_rank, const int dim_index,
                                     std::vector<std::int32_t> *payload_buffer, bool *has_payload) {
  const int bit_mask = (1 << dim_index);
  const int color_value = world_rank & ~bit_mask;

  MPI_Comm dim_comm = MPI_COMM_NULL;
  MPI_Comm_split(MPI_COMM_WORLD, color_value, world_rank, &dim_comm);

  int dim_rank = 0;
  int dim_size = 0;
  MPI_Comm_rank(dim_comm, &dim_rank);
  MPI_Comm_size(dim_comm, &dim_size);

  if (dim_size != 2) {
    MPI_Comm_free(&dim_comm);
    return false;
  }

  const int partner_dim_rank = 1 - dim_rank;
  const bool should_send = (*has_payload) && (((world_rank ^ destination_rank) & bit_mask) != 0);

  const int send_count = should_send ? static_cast<int>(payload_buffer->size()) : 0;
  int recv_count = 0;

  SendrecvCounts(dim_comm, partner_dim_rank, 1000 + dim_index, send_count, &recv_count);

  std::vector<std::int32_t> recv_buffer{};
  if (recv_count > 0) {
    recv_buffer.resize(static_cast<std::size_t>(recv_count));
  }

  const std::int32_t *send_ptr = (send_count > 0) ? payload_buffer->data() : nullptr;
  std::int32_t *recv_ptr = (recv_count > 0) ? recv_buffer.data() : nullptr;

  SendrecvPayload(dim_comm, partner_dim_rank, 2000 + dim_index, send_ptr, send_count, recv_ptr, recv_count);

  if (should_send) {
    payload_buffer->clear();
    *has_payload = false;
  }

  if (recv_count > 0) {
    *payload_buffer = std::move(recv_buffer);
    *has_payload = true;
  }

  MPI_Comm_free(&dim_comm);
  return true;
}

[[nodiscard]] int FinalizeRoutedSize(const int world_rank, const int destination_rank, const bool has_payload,
                                     const std::vector<std::int32_t> &payload_buffer) {
  int routed_size = 0;
  if (world_rank == destination_rank) {
    routed_size = has_payload ? static_cast<int>(payload_buffer.size()) : 0;
  }
  MPI_Bcast(&routed_size, 1, MPI_INT, destination_rank, MPI_COMM_WORLD);
  return routed_size;
}

[[nodiscard]] int RouteHypercubeAndGetSize(const int world_rank, const int world_size, const int source_rank,
                                           const int destination_rank, const int data_size) {
  const int dimensions = CalcDimensions(world_size);

  std::vector<std::int32_t> payload_buffer{};
  bool has_payload = false;

  InitPayloadIfSource(world_rank, source_rank, data_size, &payload_buffer, &has_payload);

  for (int dim_index = 0; dim_index < dimensions; dim_index++) {
    const bool ok = RouteOneDimension(world_rank, destination_rank, dim_index, &payload_buffer, &has_payload);
    if (!ok) {
      return data_size;
    }
  }

  return FinalizeRoutedSize(world_rank, destination_rank, has_payload, payload_buffer);
}

}  // namespace

TsarkovKHypercubeMPI::TsarkovKHypercubeMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool TsarkovKHypercubeMPI::ValidationImpl() {
  const InType &input_data = GetInput();
  if (!HasValidInputShape(input_data)) {
    return false;
  }
  return input_data[2] >= 0;
}

bool TsarkovKHypercubeMPI::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool TsarkovKHypercubeMPI::RunImpl() {
  int world_rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const InType &input_data = GetInput();
  const int source_rank = input_data[0];
  const int destination_rank = input_data[1];
  const int data_size = input_data[2];

  if (!IsWorldAndRanksOk(world_size, source_rank, destination_rank)) {
    GetOutput() = data_size;
    return true;
  }

  GetOutput() = RouteHypercubeAndGetSize(world_rank, world_size, source_rank, destination_rank, data_size);
  return true;
}

bool TsarkovKHypercubeMPI::PostProcessingImpl() {
  return true;
}

}  // namespace tsarkov_k_hypercube
