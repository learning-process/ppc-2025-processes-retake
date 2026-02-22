#include "kichanova_k_increase_contrast/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
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
         (GetInput().pixels.size() == static_cast<size_t>(GetInput().width * GetInput().height * GetInput().channels));
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

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int rows_per_process = height / size;
  int remainder = height % size;
  int start_row = rank * rows_per_process + std::min(rank, remainder);
  int end_row = start_row + rows_per_process + (rank < remainder ? 1 : 0);
  int local_rows = end_row - start_row;

  uint8_t local_min[3] = {255, 255, 255};
  uint8_t local_max[3] = {0, 0, 0};

  for (int row = start_row; row < end_row; ++row) {
    for (int col = 0; col < width; ++col) {
      size_t idx = (row * width + col) * channels;
      for (int c = 0; c < 3; ++c) {
        uint8_t val = input.pixels[idx + c];
        if (val < local_min[c]) {
          local_min[c] = val;
        }
        if (val > local_max[c]) {
          local_max[c] = val;
        }
      }
    }
  }

  uint8_t global_min[3], global_max[3];
  MPI_Allreduce(local_min, global_min, 3, MPI_UINT8_T, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(local_max, global_max, 3, MPI_UINT8_T, MPI_MAX, MPI_COMM_WORLD);

  float scale[3];
  bool need_scale[3];
  for (int c = 0; c < 3; ++c) {
    if (global_max[c] > global_min[c]) {
      scale[c] = 255.0f / (global_max[c] - global_min[c]);
      need_scale[c] = true;
    } else {
      scale[c] = 0.0f;
      need_scale[c] = false;
    }
  }

  std::vector<uint8_t> local_output(local_rows * row_size);
  for (int i = 0; i < local_rows; ++i) {
    int global_row = start_row + i;
    for (int col = 0; col < width; ++col) {
      size_t in_idx = (global_row * width + col) * channels;
      size_t out_idx = (i * width + col) * channels;
      for (int c = 0; c < 3; ++c) {
        uint8_t val = input.pixels[in_idx + c];
        if (need_scale[c]) {
          float new_val = (val - global_min[c]) * scale[c];
          local_output[out_idx + c] = static_cast<uint8_t>(std::clamp(new_val, 0.0f, 255.0f));
        } else {
          local_output[out_idx + c] = val;
        }
      }
    }
  }

  std::vector<int> recv_counts(size);
  std::vector<int> displs(size);
  for (int i = 0; i < size; ++i) {
    int i_start_row = i * rows_per_process + std::min(i, remainder);
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
