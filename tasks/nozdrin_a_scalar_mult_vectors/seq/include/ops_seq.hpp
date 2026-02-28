#pragma once

#include "nozdrin_a_scalar_mult_vectors/common/include/common.hpp"
#include "task/include/task.hpp"

namespace nozdrin_a_scalar_mult_vectors {

class NozdrinAScalarMultVectorsSEQ : public BaseTask {
 public:
	static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
		return ppc::task::TypeOfTask::kSEQ;
	}

	explicit NozdrinAScalarMultVectorsSEQ(const InType &in);

 private:
	bool ValidationImpl() override;
	bool PreProcessingImpl() override;
	bool RunImpl() override;
	bool PostProcessingImpl() override;
};

}  // namespace nozdrin_a_scalar_mult_vectors
