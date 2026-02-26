#include "dilshodov_a_convex_hull/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

#include "dilshodov_a_convex_hull/common/include/common.hpp"

namespace dilshodov_a_convex_hull {

namespace {

int Cross(const Point &o, const Point &a, const Point &b) {
  return ((a.x - o.x) * (b.y - o.y)) - ((a.y - o.y) * (b.x - o.x));
}

bool PointLess(const Point &a, const Point &b) {
  return a.x < b.x || (a.x == b.x && a.y < b.y);
}

std::vector<Point> ExtractPoints(const std::vector<int> &image, int width, int height) {
  std::vector<Point> pts;
  for (int py = 0; py < height; ++py) {
    for (int px = 0; px < width; ++px) {
      if (image[(static_cast<std::size_t>(py) * static_cast<std::size_t>(width)) + static_cast<std::size_t>(px)] != 0) {
        pts.push_back({px, py});
      }
    }
  }
  return pts;
}

std::vector<Point> GrahamScan(std::vector<Point> pts) {
  if (pts.size() < 3) {
    return {};
  }

  std::ranges::sort(pts, PointLess);
  pts.erase(std::ranges::unique(pts).begin(), pts.end());

  if (pts.size() < 3) {
    return {};
  }

  std::vector<Point> hull;

  for (const auto &p : pts) {
    while (hull.size() >= 2 && Cross(hull[hull.size() - 2], hull.back(), p) <= 0) {
      hull.pop_back();
    }
    hull.push_back(p);
  }

  auto lower_size = hull.size();
  for (int i = static_cast<int>(pts.size()) - 2; i >= 0; --i) {
    const auto &p = pts[static_cast<std::size_t>(i)];
    while (hull.size() > lower_size && Cross(hull[hull.size() - 2], hull.back(), p) <= 0) {
      hull.pop_back();
    }
    hull.push_back(p);
  }

  hull.pop_back();
  if (hull.size() < 3) {
    return {};
  }
  return hull;
}

}  // namespace

ConvexHullSEQ::ConvexHullSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  InType copy = in;
  GetInput() = std::move(copy);
  GetOutput().clear();
}

bool ConvexHullSEQ::ValidationImpl() {
  const auto &input = GetInput();
  if (input.size() < 3) {
    return false;
  }
  width_ = input[0];
  height_ = input[1];
  return width_ > 0 && height_ > 0 && static_cast<int>(input.size()) == (width_ * height_) + 2;
}

bool ConvexHullSEQ::PreProcessingImpl() {
  return true;
}

bool ConvexHullSEQ::RunImpl() {
  const auto &input = GetInput();
  std::vector<int> pixels(input.begin() + 2, input.end());

  auto pts = ExtractPoints(pixels, width_, height_);
  GetOutput() = GrahamScan(std::move(pts));
  return true;
}

bool ConvexHullSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace dilshodov_a_convex_hull
