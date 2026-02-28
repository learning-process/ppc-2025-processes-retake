#include "nozdrin_a_scalar_mult_vectors/common/include/common.hpp"

#include <random>

namespace nozdrin_a_scalar_mult_vectors {

std::vector<double> GenerateRandomVector(std::size_t size, double min, double max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(min, max);

    std::vector<double> values(size);
    for (auto &x : values) {
        x = dis(gen);
    }

    return values;
}

double AnalyticDotProduct(const std::vector<double> &a, const std::vector<double> &b) {
    double result = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }
    return result;
}

}  // namespace nozdrin_a_scalar_mult_vectors
