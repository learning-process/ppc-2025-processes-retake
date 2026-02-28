#include "nazyrov_a_global_opt_2d/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

#include "nazyrov_a_global_opt_2d/common/include/common.hpp"

namespace nazyrov_a_global_opt_2d {

namespace {

void ComputeDistribution(int num_items, int world_size, std::vector<int> &counts, std::vector<int> &displs) {
  counts.assign(static_cast<std::size_t>(world_size), 0);
  displs.assign(static_cast<std::size_t>(world_size), 0);
  int base = num_items / world_size;
  int rem = num_items % world_size;
  for (int i = 0; i < world_size; ++i) {
    counts[static_cast<std::size_t>(i)] = base + ((i < rem) ? 1 : 0);
    displs[static_cast<std::size_t>(i)] =
        (i == 0) ? 0 : (displs[static_cast<std::size_t>(i - 1)] + counts[static_cast<std::size_t>(i - 1)]);
  }
}

}  // namespace

void GlobalOpt2dMPI::PeanoMap(double t_val, int level, double &ux, double &uy) {
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

double GlobalOpt2dMPI::ToX(double t_val) {
  double ux = 0.0;
  double uy = 0.0;
  PeanoMap(t_val, kPeanoLevel, ux, uy);
  return GetInput().x_min + (ux * (GetInput().x_max - GetInput().x_min));
}

double GlobalOpt2dMPI::ToY(double t_val) {
  double ux = 0.0;
  double uy = 0.0;
  PeanoMap(t_val, kPeanoLevel, ux, uy);
  return GetInput().y_min + (uy * (GetInput().y_max - GetInput().y_min));
}

void GlobalOpt2dMPI::SortTrials() {
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

double GlobalOpt2dMPI::ComputeLipschitz() {
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

void GlobalOpt2dMPI::InitTrials() {
  std::vector<double> init_pack(8, 0.0);
  if (world_rank_ == 0) {
    double t0 = 0.0;
    double t1 = 1.0;
    double x0 = ToX(t0);
    double y0 = ToY(t0);
    double z0 = GetInput().func(x0, y0);
    double x1 = ToX(t1);
    double y1 = ToY(t1);
    double z1 = GetInput().func(x1, y1);
    init_pack = {t0, t1, x0, y0, z0, x1, y1, z1};
  }
  MPI_Bcast(init_pack.data(), 8, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  t_points_.push_back(init_pack[0]);
  t_points_.push_back(init_pack[1]);
  trials_.push_back({init_pack[2], init_pack[3], init_pack[4]});
  trials_.push_back({init_pack[5], init_pack[6], init_pack[7]});
}

void GlobalOpt2dMPI::DistributeAndCompute(double m_val, std::vector<double> &all_chars) {
  int interval_count = static_cast<int>(t_points_.size()) - 1;

  std::vector<double> packed(static_cast<std::size_t>(interval_count) * 4);
  for (int i = 0; i < interval_count; ++i) {
    auto idx = static_cast<std::size_t>(i);
    packed[(idx * 4) + 0] = t_points_[idx];
    packed[(idx * 4) + 1] = t_points_[idx + 1];
    packed[(idx * 4) + 2] = trials_[idx].z;
    packed[(idx * 4) + 3] = trials_[idx + 1].z;
  }

  std::vector<int> counts;
  std::vector<int> displs_items;
  ComputeDistribution(interval_count, world_size_, counts, displs_items);

  std::vector<int> send_counts(static_cast<std::size_t>(world_size_));
  std::vector<int> send_displs(static_cast<std::size_t>(world_size_));
  for (int i = 0; i < world_size_; ++i) {
    send_counts[static_cast<std::size_t>(i)] = counts[static_cast<std::size_t>(i)] * 4;
    send_displs[static_cast<std::size_t>(i)] = displs_items[static_cast<std::size_t>(i)] * 4;
  }

  int local_count = counts[static_cast<std::size_t>(world_rank_)];
  std::vector<double> local_data(static_cast<std::size_t>(local_count) * 4);

  MPI_Scatterv(packed.data(), send_counts.data(), send_displs.data(), MPI_DOUBLE, local_data.data(),
               static_cast<int>(local_data.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  std::vector<double> local_chars(static_cast<std::size_t>(local_count));
  for (int i = 0; i < local_count; ++i) {
    auto idx = static_cast<std::size_t>(i);
    double dt = local_data[(idx * 4) + 1] - local_data[(idx * 4) + 0];
    double dz = local_data[(idx * 4) + 3] - local_data[(idx * 4) + 2];
    local_chars[idx] =
        (m_val * dt) + ((dz * dz) / (m_val * dt)) - (2.0 * (local_data[(idx * 4) + 3] + local_data[(idx * 4) + 2]));
  }

  all_chars.resize(static_cast<std::size_t>(interval_count));
  MPI_Gatherv(local_chars.data(), local_count, MPI_DOUBLE, all_chars.data(), counts.data(), displs_items.data(),
              MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

std::array<double, 3> GlobalOpt2dMPI::SelectBestAndAddTrial(const std::vector<double> &all_chars, double m_val,
                                                            int &converge_flag) {
  std::array<double, 3> new_trial = {0.0, 0.0, 0.0};
  converge_flag = 0;

  int interval_count = static_cast<int>(t_points_.size()) - 1;
  int best_idx = 0;
  double best_char = -std::numeric_limits<double>::max();
  for (int i = 0; i < interval_count; ++i) {
    if (all_chars[static_cast<std::size_t>(i)] > best_char) {
      best_char = all_chars[static_cast<std::size_t>(i)];
      best_idx = i;
    }
  }

  double t_left = t_points_[static_cast<std::size_t>(best_idx)];
  double t_right = t_points_[static_cast<std::size_t>(best_idx) + 1];

  if (t_right - t_left < GetInput().epsilon) {
    converge_flag = 1;
    return new_trial;
  }

  double z_left = trials_[static_cast<std::size_t>(best_idx)].z;
  double z_right = trials_[static_cast<std::size_t>(best_idx) + 1].z;

  double t_new = (0.5 * (t_left + t_right)) - ((z_right - z_left) / (2.0 * m_val));
  t_new = std::max(t_left + 1e-12, std::min(t_new, t_right - 1e-12));

  double xn = ToX(t_new);
  double yn = ToY(t_new);
  double zn = GetInput().func(xn, yn);

  new_trial = {t_new, xn, yn};
  t_points_.push_back(t_new);
  trials_.push_back({xn, yn, zn});

  return new_trial;
}

void GlobalOpt2dMPI::FindBestResult() {
  auto &output = GetOutput();

  if (world_rank_ == 0) {
    double best_z = std::numeric_limits<double>::max();
    for (auto &trial : trials_) {
      if (trial.z < best_z) {
        best_z = trial.z;
        output.x_opt = trial.x;
        output.y_opt = trial.y;
        output.func_min = trial.z;
      }
    }
  }

  std::array<double, 3> result_pack = {output.x_opt, output.y_opt, output.func_min};
  MPI_Bcast(result_pack.data(), 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  output.x_opt = result_pack[0];
  output.y_opt = result_pack[1];
  output.func_min = result_pack[2];
}

GlobalOpt2dMPI::GlobalOpt2dMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType();
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
}

bool GlobalOpt2dMPI::ValidationImpl() {
  if (world_rank_ != 0) {
    return true;
  }
  const auto &input = GetInput();
  return (input.func != nullptr) && (input.x_min < input.x_max) && (input.y_min < input.y_max) &&
         (input.epsilon > 0.0) && (input.r_param > 1.0) && (input.max_iterations > 0);
}

bool GlobalOpt2dMPI::PreProcessingImpl() {
  t_points_.clear();
  trials_.clear();
  lip_est_ = 1.0;
  return true;
}

bool GlobalOpt2dMPI::RunImpl() {
  const auto &input = GetInput();
  auto &output = GetOutput();

  t_points_.clear();
  trials_.clear();
  output.iterations = 0;
  output.converged = false;

  InitTrials();

  for (int iter = 0; iter < input.max_iterations; ++iter) {
    SortTrials();
    lip_est_ = ComputeLipschitz();
    if (lip_est_ < 1e-10) {
      lip_est_ = 1.0;
    }

    double m_val = input.r_param * lip_est_;

    std::vector<double> all_chars;
    DistributeAndCompute(m_val, all_chars);

    std::array<double, 3> new_trial = {0.0, 0.0, 0.0};
    int converge_flag = 0;
    if (world_rank_ == 0) {
      new_trial = SelectBestAndAddTrial(all_chars, m_val, converge_flag);
    }

    MPI_Bcast(&converge_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (converge_flag != 0) {
      output.converged = true;
      output.iterations = iter + 1;
      break;
    }

    MPI_Bcast(new_trial.data(), 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (world_rank_ != 0) {
      double zn = input.func(new_trial[1], new_trial[2]);
      t_points_.push_back(new_trial[0]);
      trials_.push_back({new_trial[1], new_trial[2], zn});
    }

    output.iterations = iter + 1;
  }

  FindBestResult();
  return true;
}

bool GlobalOpt2dMPI::PostProcessingImpl() {
  return true;
}

}  // namespace nazyrov_a_global_opt_2d
