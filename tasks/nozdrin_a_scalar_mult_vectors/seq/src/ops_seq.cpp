#include "nozdrin_a_scalar_mult_vectors/seq/include/ops_seq.hpp"

#include "nozdrin_a_scalar_mult_vectors/common/include/common.hpp"

namespace nozdrin_a_scalar_mult_vectors {

NozdrinAScalarMultVectorsSEQ::NozdrinAScalarMultVectorsSEQ(const InType &in) {
    SetTypeOfTask(GetStaticTypeOfTask());
    GetInput() = in;
    GetOutput() = 0.0;
}

bool NozdrinAScalarMultVectorsSEQ::ValidationImpl() {
    const auto &in = GetInput();
    return !in.a.empty() && (in.a.size() == in.b.size());
}

bool NozdrinAScalarMultVectorsSEQ::PreProcessingImpl() {
    GetOutput() = 0.0;
    return true;
}

bool NozdrinAScalarMultVectorsSEQ::RunImpl() {
    const auto &in = GetInput();

    double sum = 0.0;
    for (std::size_t i = 0; i < in.a.size(); ++i) {
        sum += in.a[i] * in.b[i];
    }

    GetOutput() = sum;
    return true;
}

bool NozdrinAScalarMultVectorsSEQ::PostProcessingImpl() {
    return true;
}

}  // namespace nozdrin_a_scalar_mult_vectors
