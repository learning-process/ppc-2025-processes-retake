#include "kichanova_k_increase_contrast/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "kichanova_k_increase_contrast/common/include/common.hpp"

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
         (GetInput().pixels.size() == static_cast<size_t>(GetInput().width) * GetInput().height * GetInput().channels);
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
  const size_t total_pixels = static_cast<size_t>(width) * static_cast<size_t>(height);

  uint8_t min_r = 255;
  uint8_t max_r = 0;
  uint8_t min_g = 255;
  uint8_t max_g = 0;
  uint8_t min_b = 255;
  uint8_t max_b = 0;

  for (size_t i = 0; i < total_pixels; ++i) {
    size_t idx = i * channels;

    uint8_t r = input.pixels[idx];
    uint8_t g = input.pixels[idx + 1];
    uint8_t b = input.pixels[idx + 2];

    min_r = std::min(r, min_r);
    max_r = std::max(r, max_r);
    min_g = std::min(g, min_g);
    max_g = std::max(g, max_g);
    min_b = std::min(b, min_b);
    max_b = std::max(b, max_b);
  }

  float scale_r = 0.0F;
  float scale_g = 0.0F;
  float scale_b = 0.0F;

  if (max_r > min_r) {
    scale_r = 255.0F / static_cast<float>(max_r - min_r);
  }

  if (max_g > min_g) {
    scale_g = 255.0F / static_cast<float>(max_g - min_g);
  }

  if (max_b > min_b) {
    scale_b = 255.0F / static_cast<float>(max_b - min_b);
  }

  for (size_t i = 0; i < total_pixels; ++i) {
    size_t idx = i * channels;

    uint8_t r = input.pixels[idx];
    uint8_t g = input.pixels[idx + 1];
    uint8_t b = input.pixels[idx + 2];

    if (max_r > min_r) {
      float new_r = (static_cast<float>(r) - static_cast<float>(min_r)) * scale_r;
      output.pixels[idx] = static_cast<uint8_t>(std::clamp(new_r, 0.0F, 255.0F));
    } else {
      output.pixels[idx] = r;
    }

    if (max_g > min_g) {
      float new_g = (static_cast<float>(g) - static_cast<float>(min_g)) * scale_g;
      output.pixels[idx + 1] = static_cast<uint8_t>(std::clamp(new_g, 0.0F, 255.0F));
    } else {
      output.pixels[idx + 1] = g;
    }

    if (max_b > min_b) {
      float new_b = (static_cast<float>(b) - static_cast<float>(min_b)) * scale_b;
      output.pixels[idx + 2] = static_cast<uint8_t>(std::clamp(new_b, 0.0F, 255.0F));
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
