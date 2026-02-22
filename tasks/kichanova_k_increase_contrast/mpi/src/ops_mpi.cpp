#include "kichanova_k_increase_contrast/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

#include "kichanova_k_increase_contrast/common/include/common.hpp"

namespace kichanova_k_increase_contrast {

KichanovaKIncreaseContrastMPI::KichanovaKIncreaseContrastMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().width = in.width;
  GetOutput().height = in.height;
  GetOutput().channels = in.channels;
  GetOutput().pixels.resize(in.pixels.size());
}

bool KichanovaKIncreaseContrastMPI::ValidationImpl() {
  return (GetInput().width > 0) && (GetInput().height > 0) && (GetInput().channels == 3) &&
         (GetInput().pixels.size() == static_cast<size_t>(GetInput().width) * GetInput().height * GetInput().channels);
}

bool KichanovaKIncreaseContrastMPI::PreProcessingImpl() {
  return true;
}

bool KichanovaKIncreaseContrastMPI::RunImpl() {
  const auto &input = GetInput();
  auto &output = GetOutput();

  const int width = input.width;
  const int height = input.height;
  const int channels = 3;
  const int row_size = width * channels;

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int rows_per_process = height / size;
  int remainder = height % size;
  int start_row = (rank * rows_per_process) + std::min(rank, remainder);
  int end_row = start_row + rows_per_process + (rank < remainder ? 1 : 0);
  int local_rows = end_row - start_row;

  std::array<uint8_t, 3> local_min = {255, 255, 255};
  std::array<uint8_t, 3> local_max = {0, 0, 0};

  for (int row = start_row; row < end_row; ++row) {
    for (int col = 0; col < width; ++col) {
      size_t idx = (static_cast<size_t>(row) * static_cast<size_t>(width)) + static_cast<size_t>(col);
      idx *= static_cast<size_t>(channels);
      for (int channel = 0; channel < 3; ++channel) {
        uint8_t val = input.pixels[idx + channel];
        local_min[channel] = std::min(val, local_min[channel]);
        local_max[channel] = std::max(val, local_max[channel]);
      }
    }
  }

  std::array<uint8_t, 3> global_min{};
  std::array<uint8_t, 3> global_max{};
  MPI_Allreduce(local_min.data(), global_min.data(), 3, MPI_UINT8_T, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(local_max.data(), global_max.data(), 3, MPI_UINT8_T, MPI_MAX, MPI_COMM_WORLD);

  std::array<float, 3> scale{};
  std::array<bool, 3> need_scale{};
  for (int channel = 0; channel < 3; ++channel) {
    if (global_max[channel] > global_min[channel]) {
      scale[channel] = 255.0F / static_cast<float>(global_max[channel] - global_min[channel]);
      need_scale[channel] = true;
    } else {
      scale[channel] = 0.0F;
      need_scale[channel] = false;
    }
  }

  std::vector<uint8_t> local_output(static_cast<size_t>(local_rows) * row_size);
  for (int i = 0; i < local_rows; ++i) {
    int global_row = start_row + i;
    for (int col = 0; col < width; ++col) {
      size_t in_idx = (static_cast<size_t>(global_row) * width + col) * channels;
      size_t out_idx = (static_cast<size_t>(i) * width + col) * channels;
      for (int channel = 0; channel < 3; ++channel) {
        uint8_t val = input.pixels[in_idx + channel];
        if (need_scale[channel]) {
          float new_val = (static_cast<float>(val) - static_cast<float>(global_min[channel])) * scale[channel];
          local_output[out_idx + channel] = static_cast<uint8_t>(std::clamp(new_val, 0.0F, 255.0F));
        } else {
          local_output[out_idx + channel] = val;
        }
      }
    }
  }

  std::vector<int> recv_counts(size);
  std::vector<int> displs(size);
  for (int i = 0; i < size; ++i) {
    int i_start_row = (i * rows_per_process) + std::min(i, remainder);
    int i_end_row = i_start_row + rows_per_process + (i < remainder ? 1 : 0);
    int i_rows = i_end_row - i_start_row;
    recv_counts[i] = i_rows * row_size;
    displs[i] = i_start_row * row_size;
  }

  MPI_Allgatherv(local_output.data(), local_rows * row_size, MPI_UINT8_T, output.pixels.data(), recv_counts.data(),
                 displs.data(), MPI_UINT8_T, MPI_COMM_WORLD);

  return true;
}

bool KichanovaKIncreaseContrastMPI::PostProcessingImpl() {
  return true;
}

}  // namespace kichanova_k_increase_contrast
