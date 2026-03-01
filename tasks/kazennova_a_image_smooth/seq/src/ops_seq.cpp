#include "kazennova_a_image_smooth/seq/include/ops_seq.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

#include "kazennova_a_image_smooth/common/include/common.hpp"

namespace kazennova_a_image_smooth {

const std::array<std::array<float, 3>, 3> kKernel = {
    {{{1.0F / 16, 2.0F / 16, 1.0F / 16}}, {{2.0F / 16, 4.0F / 16, 2.0F / 16}}, {{1.0F / 16, 2.0F / 16, 1.0F / 16}}}};

KazennovaAImageSmoothSEQ::KazennovaAImageSmoothSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = in;
}

bool KazennovaAImageSmoothSEQ::ValidationImpl() {
  const auto &in = GetInput();
  return in.width > 0 && in.height > 0 && !in.data.empty() && (in.channels == 1 || in.channels == 3);
}

bool KazennovaAImageSmoothSEQ::PreProcessingImpl() {
  GetOutput().data.resize(GetInput().data.size());
  return true;
}

uint8_t KazennovaAImageSmoothSEQ::ApplyKernelToPixel(int x, int y, int c) {
  const auto &in = GetInput();
  float sum = 0.0F;

  {
    int nx = std::clamp(x - 1, 0, in.width - 1);
    int ny = std::clamp(y - 1, 0, in.height - 1);
    int idx = ((ny * in.width + nx) * in.channels) + c;
    sum += static_cast<float>(in.data[idx]) * kKernel[0][0];
  }

  {
    int nx = std::clamp(x, 0, in.width - 1);
    int ny = std::clamp(y - 1, 0, in.height - 1);
    int idx = ((ny * in.width + nx) * in.channels) + c;
    sum += static_cast<float>(in.data[idx]) * kKernel[0][1];
  }

  {
    int nx = std::clamp(x + 1, 0, in.width - 1);
    int ny = std::clamp(y - 1, 0, in.height - 1);
    int idx = ((ny * in.width + nx) * in.channels) + c;
    sum += static_cast<float>(in.data[idx]) * kKernel[0][2];
  }

  {
    int nx = std::clamp(x - 1, 0, in.width - 1);
    int ny = std::clamp(y, 0, in.height - 1);
    int idx = ((ny * in.width + nx) * in.channels) + c;
    sum += static_cast<float>(in.data[idx]) * kKernel[1][0];
  }

  {
    int nx = std::clamp(x, 0, in.width - 1);
    int ny = std::clamp(y, 0, in.height - 1);
    int idx = ((ny * in.width + nx) * in.channels) + c;
    sum += static_cast<float>(in.data[idx]) * kKernel[1][1];
  }

  {
    int nx = std::clamp(x + 1, 0, in.width - 1);
    int ny = std::clamp(y, 0, in.height - 1);
    int idx = ((ny * in.width + nx) * in.channels) + c;
    sum += static_cast<float>(in.data[idx]) * kKernel[1][2];
  }

  {
    int nx = std::clamp(x - 1, 0, in.width - 1);
    int ny = std::clamp(y + 1, 0, in.height - 1);
    int idx = ((ny * in.width + nx) * in.channels) + c;
    sum += static_cast<float>(in.data[idx]) * kKernel[2][0];
  }

  {
    int nx = std::clamp(x, 0, in.width - 1);
    int ny = std::clamp(y + 1, 0, in.height - 1);
    int idx = ((ny * in.width + nx) * in.channels) + c;
    sum += static_cast<float>(in.data[idx]) * kKernel[2][1];
  }

  {
    int nx = std::clamp(x + 1, 0, in.width - 1);
    int ny = std::clamp(y + 1, 0, in.height - 1);
    int idx = ((ny * in.width + nx) * in.channels) + c;
    sum += static_cast<float>(in.data[idx]) * kKernel[2][2];
  }

  return static_cast<uint8_t>(std::round(sum));
}

bool KazennovaAImageSmoothSEQ::RunImpl() {
  const auto &in = GetInput();
  auto &out = GetOutput();

  out.width = in.width;
  out.height = in.height;
  out.channels = in.channels;
  out.data.resize(in.data.size());

  for (int row = 0; row < in.height; ++row) {
    for (int col = 0; col < in.width; ++col) {
      for (int ch = 0; ch < in.channels; ++ch) {
        int idx = ((row * in.width + col) * in.channels) + ch;
        out.data[idx] = ApplyKernelToPixel(col, row, ch);
      }
    }
  }

  return true;
}

bool KazennovaAImageSmoothSEQ::PostProcessingImpl() {
  return !GetOutput().data.empty();
}

}  // namespace kazennova_a_image_smooth
