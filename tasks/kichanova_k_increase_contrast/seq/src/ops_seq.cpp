#include "kichanova_k_increase_contrast/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>

#include "kichanova_k_increase_contrast/common/include/common.hpp"
#include "util/include/util.hpp"

namespace kichanova_k_increase_contrast {

KichanovaKIncreaseContrastSEQ::KichanovaKIncreaseContrastSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().width = in.width;
  GetOutput().height = in.height;
  GetOutput().channels = in.channels;
  GetOutput().pixels.resize(in.pixels.size());
}

bool KichanovaKIncreaseContrastSEQ::ValidationImpl() {
  return (GetInput().width > 0) && (GetInput().height > 0) && (GetInput().channels == 3) &&
         (GetInput().pixels.size() == static_cast<size_t>(GetInput().width * GetInput().height * GetInput().channels));
}

bool KichanovaKIncreaseContrastSEQ::PreProcessingImpl() {
  return true;
}

bool KichanovaKIncreaseContrastSEQ::RunImpl() {
  const auto &input = GetInput();
  auto &output = GetOutput();

  const int width = input.width;
  const int height = input.height;
  const int channels = 3;
  const size_t total_pixels = width * height;

  uint8_t min_r = 255, max_r = 0;
  uint8_t min_g = 255, max_g = 0;
  uint8_t min_b = 255, max_b = 0;

  for (size_t i = 0; i < total_pixels; ++i) {
    size_t idx = i * channels;

    uint8_t r = input.pixels[idx];
    uint8_t g = input.pixels[idx + 1];
    uint8_t b = input.pixels[idx + 2];

    if (r < min_r) {
      min_r = r;
    }
    if (r > max_r) {
      max_r = r;
    }
    if (g < min_g) {
      min_g = g;
    }
    if (g > max_g) {
      max_g = g;
    }
    if (b < min_b) {
      min_b = b;
    }
    if (b > max_b) {
      max_b = b;
    }
  }

  float scale_r = 0.0f, scale_g = 0.0f, scale_b = 0.0f;

  if (max_r > min_r) {
    scale_r = 255.0f / (max_r - min_r);
  }

  if (max_g > min_g) {
    scale_g = 255.0f / (max_g - min_g);
  }

  if (max_b > min_b) {
    scale_b = 255.0f / (max_b - min_b);
  }

  for (size_t i = 0; i < total_pixels; ++i) {
    size_t idx = i * channels;

    uint8_t r = input.pixels[idx];
    uint8_t g = input.pixels[idx + 1];
    uint8_t b = input.pixels[idx + 2];

    if (max_r > min_r) {
      float new_r = (r - min_r) * scale_r;
      output.pixels[idx] = static_cast<uint8_t>(std::clamp(new_r, 0.0f, 255.0f));
    } else {
      output.pixels[idx] = r;
    }

    if (max_g > min_g) {
      float new_g = (g - min_g) * scale_g;
      output.pixels[idx + 1] = static_cast<uint8_t>(std::clamp(new_g, 0.0f, 255.0f));
    } else {
      output.pixels[idx + 1] = g;
    }

    if (max_b > min_b) {
      float new_b = (b - min_b) * scale_b;
      output.pixels[idx + 2] = static_cast<uint8_t>(std::clamp(new_b, 0.0f, 255.0f));
    } else {
      output.pixels[idx + 2] = b;
    }
  }

  return true;
}

bool KichanovaKIncreaseContrastSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace kichanova_k_increase_contrast
