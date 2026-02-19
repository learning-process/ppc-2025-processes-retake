#include "morozova_s_connected_components/seq/include/ops_seq.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <queue>
#include <utility>
#include <vector>

#include "morozova_s_connected_components/common/include/common.hpp"

namespace morozova_s_connected_components {

MorozovaSConnectedComponentsSEQ::MorozovaSConnectedComponentsSEQ(const InType &in) : BaseTask() {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

namespace {

constexpr std::array<std::pair<int, int>, 8> kShifts = {
    {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}}};
}  // namespace

bool MorozovaSConnectedComponentsSEQ::ValidationImpl() {
  const auto &input = GetInput();
  if (input.empty()) {
    return true;
  }
  const size_t cols = input.front().size();
  if (cols == 0) {
    return false;
  }
  for (const auto &row : input) {
    if (row.size() != cols) {
      return false;
    }
    for (int val : row) {
      if (val != 0 && val != 1) {
        return false;
      }
    }
  }
  return true;
}

bool MorozovaSConnectedComponentsSEQ::PreProcessingImpl() {
  const auto &input = GetInput();
  if (input.empty()) {
    GetOutput().clear();
    return true;
  }
  rows_ = static_cast<int>(input.size());
  cols_ = static_cast<int>(input.front().size());
  auto &output = GetOutput();
  output.resize(rows_);
  for (int i = 0; i < rows_; ++i) {
    output[i].resize(cols_, 0);
  }
  return true;
}

void MorozovaSConnectedComponentsSEQ::ProcessComponent(int start_i, int start_j, int current_label) {
  const auto &input = GetInput();
  auto &output = GetOutput();
  std::queue<std::pair<int, int>> q;
  q.emplace(start_i, start_j);
  output[start_i][start_j] = current_label;
  while (!q.empty()) {
    const auto [x, y] = q.front();
    q.pop();
    for (const auto &[dx, dy] : kShifts) {
      const int nx = x + dx;
      const int ny = y + dy;
      if (nx >= 0 && nx < rows_ && ny >= 0 && ny < cols_ && input[nx][ny] == 1 && output[nx][ny] == 0) {
        output[nx][ny] = current_label;
        q.emplace(nx, ny);
      }
    }
  }
}

bool MorozovaSConnectedComponentsSEQ::RunImpl() {
  const auto &input = GetInput();
  auto &output = GetOutput();
  if (input.empty() || rows_ == 0 || cols_ == 0) {
    return true;
  }
  int current_label = 1;
  for (int i = 0; i < rows_; ++i) {
    for (int j = 0; j < cols_; ++j) {
      if (input[i][j] == 1 && output[i][j] == 0) {
        ProcessComponent(i, j, current_label);
        ++current_label;
      }
    }
  }

  return true;
}

bool MorozovaSConnectedComponentsSEQ::PostProcessingImpl() {
  auto &output = GetOutput();
  if (output.empty()) {
    return true;
  }
  int max_label = 0;
  for (const auto &row : output) {
    for (const int v : row) {
      max_label = std::max(max_label, v);
    }
  }
  output.push_back({max_label});
  return true;
}

}  // namespace morozova_s_connected_components
