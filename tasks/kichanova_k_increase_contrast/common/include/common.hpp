#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace kichanova_k_increase_contrast {

struct Image {
  std::vector<uint8_t> pixels;
  int width;
  int height;
  int channels;
};

using InType = Image;
using OutType = Image;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace kichanova_k_increase_contrast