#pragma once

#include <algorithm>
#include <cstddef>
#include <random>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace nozdrin_a_scalar_mult_vectors {

struct Input {
	std::vector<double> a;
	std::vector<double> b;
};

using InType = Input;
using OutType = double;
using TestType = std::tuple<InType, OutType>;
using BaseTask = ppc::task::Task<InType, OutType>;

inline std::vector<double> GenerateRandomVector(std::size_t size, double min = -100.0, double max = 100.0) {
	std::mt19937 gen(std::random_device{}());
	std::uniform_real_distribution<double> dist(min, max);

	std::vector<double> v(size);
	for (auto &x : v) {
		x = dist(gen);
	}

	return v;
}

inline double AnalyticDotProduct(const std::vector<double> &a, const std::vector<double> &b) {
	const std::size_t n = std::min(a.size(), b.size());
	double sum = 0.0;
	for (std::size_t i = 0; i < n; ++i) {
		sum += a[i] * b[i];
	}
	return sum;
}

}  // namespace nozdrin_a_scalar_mult_vectors
