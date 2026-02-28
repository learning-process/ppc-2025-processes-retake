#include "nazyrov_a_global_opt_2d/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

#include "nazyrov_a_global_opt_2d/common/include/common.hpp"

namespace nazyrov_a_global_opt_2d {

void GlobalOpt2dSEQ::PeanoMap(double t_val, int level, double &ux, double &uy) {
  ux = 0.0;
  uy = 0.0;
  double scale = 0.5;
  for (int i = 0; i < level; ++i) {
    int quad = static_cast<int>(t_val * 4.0);
    t_val = (t_val * 4.0) - quad;
    double dx = 0.0;
    double dy = 0.0;
    switch (quad) {
      case 0:
        break;
      case 1:
        dy = 1.0;
        break;
      case 2:
        dx = 1.0;
        dy = 1.0;
        break;
      case 3:
        dx = 1.0;
        break;
      default:
        break;
    }
    ux += dx * scale;
    uy += dy * scale;
    scale *= 0.5;
  }
}

double GlobalOpt2dSEQ::ToX(double t_val) {
  double ux = 0.0;
  double uy = 0.0;
  PeanoMap(t_val, kPeanoLevel, ux, uy);
  return GetInput().x_min + (ux * (GetInput().x_max - GetInput().x_min));
}

double GlobalOpt2dSEQ::ToY(double t_val) {
  double ux = 0.0;
  double uy = 0.0;
  PeanoMap(t_val, kPeanoLevel, ux, uy);
  return GetInput().y_min + (uy * (GetInput().y_max - GetInput().y_min));
}

void GlobalOpt2dSEQ::SortTrials() {
  std::vector<std::size_t> idx(t_points_.size());
  for (std::size_t i = 0; i < idx.size(); ++i) {
    idx[i] = i;
  }
  std::ranges::sort(idx, [this](std::size_t a, std::size_t b) { return t_points_[a] < t_points_[b]; });

  std::vector<double> sorted_t(t_points_.size());
  std::vector<TrialPoint> sorted_tr(trials_.size());
  for (std::size_t i = 0; i < idx.size(); ++i) {
    sorted_t[i] = t_points_[idx[i]];
    sorted_tr[i] = trials_[idx[i]];
  }
  t_points_ = std::move(sorted_t);
  trials_ = std::move(sorted_tr);
}

double GlobalOpt2dSEQ::ComputeLipschitz() {
  double max_lip = 0.0;
  for (std::size_t i = 1; i < t_points_.size(); ++i) {
    double dt = t_points_[i] - t_points_[i - 1];
    if (dt > 1e-15) {
      double lip = std::abs(trials_[i].z - trials_[i - 1].z) / dt;
      max_lip = std::max(max_lip, lip);
    }
  }
  return max_lip;
}

int GlobalOpt2dSEQ::FindBestInterval() {
  const auto &input = GetInput();
  double m_val = input.r_param * lip_est_;
  if (m_val < 1e-10) {
    m_val = 1.0;
  }

  double best_char = -std::numeric_limits<double>::max();
  int best_idx = 0;

  for (std::size_t i = 1; i < t_points_.size(); ++i) {
    double dt = t_points_[i] - t_points_[i - 1];
    double dz = trials_[i].z - trials_[i - 1].z;
    double characteristic = (m_val * dt) + ((dz * dz) / (m_val * dt)) - (2.0 * (trials_[i].z + trials_[i - 1].z));
    if (characteristic > best_char) {
      best_char = characteristic;
      best_idx = static_cast<int>(i) - 1;
    }
  }
  return best_idx;
}

GlobalOpt2dSEQ::GlobalOpt2dSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType();
}

bool GlobalOpt2dSEQ::ValidationImpl() {
  const auto &input = GetInput();
  return (input.func != nullptr) && (input.x_min < input.x_max) && (input.y_min < input.y_max) &&
         (input.epsilon > 0.0) && (input.r_param > 1.0) && (input.max_iterations > 0);
}

bool GlobalOpt2dSEQ::PreProcessingImpl() {
  t_points_.clear();
  trials_.clear();
  lip_est_ = 1.0;

  const auto &input = GetInput();
  double t0 = 0.0;
  double t1 = 1.0;

  t_points_.push_back(t0);
  t_points_.push_back(t1);

  double x0 = ToX(t0);
  double y0 = ToY(t0);
  double x1 = ToX(t1);
  double y1 = ToY(t1);

  trials_.push_back({x0, y0, input.func(x0, y0)});
  trials_.push_back({x1, y1, input.func(x1, y1)});

  return true;
}

bool GlobalOpt2dSEQ::RunImpl() {
  const auto &input = GetInput();
  auto &output = GetOutput();

  output.iterations = 0;
  output.converged = false;

  for (int iter = 0; iter < input.max_iterations; ++iter) {
    SortTrials();
    lip_est_ = ComputeLipschitz();
    if (lip_est_ < 1e-10) {
      lip_est_ = 1.0;
    }

    int best_idx = FindBestInterval();
    double t_left = t_points_[static_cast<std::size_t>(best_idx)];
    double t_right = t_points_[static_cast<std::size_t>(best_idx) + 1];

    if (t_right - t_left < input.epsilon) {
      output.converged = true;
      output.iterations = iter + 1;
      break;
    }

    double z_left = trials_[static_cast<std::size_t>(best_idx)].z;
    double z_right = trials_[static_cast<std::size_t>(best_idx) + 1].z;
    double m_val = input.r_param * lip_est_;

    double t_new = (0.5 * (t_left + t_right)) - ((z_right - z_left) / (2.0 * m_val));
    t_new = std::max(t_left + 1e-12, std::min(t_new, t_right - 1e-12));

    double xn = ToX(t_new);
    double yn = ToY(t_new);
    double zn = input.func(xn, yn);

    t_points_.push_back(t_new);
    trials_.push_back({xn, yn, zn});
    output.iterations = iter + 1;
  }

  double best_z = std::numeric_limits<double>::max();
  for (auto &trial : trials_) {
    if (trial.z < best_z) {
      best_z = trial.z;
      output.x_opt = trial.x;
      output.y_opt = trial.y;
      output.func_min = trial.z;
    }
  }

  return true;
}

bool GlobalOpt2dSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace nazyrov_a_global_opt_2d
