#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace kichanova_k_increase_contrast {

struct Image {
  std::vector<uint8_t> pixels;
  int width{0};
  int height{0};
  int channels{0};
};

using InType = Image;
using OutType = Image;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace kichanova_k_increase_contrast
