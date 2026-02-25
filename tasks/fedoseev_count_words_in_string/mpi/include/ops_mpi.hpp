#pragma once

#include "fedoseev_count_words_in_string/common/include/common.hpp"
#include "task/include/task.hpp"

namespace fedoseev_count_words_in_string {

class FedoseevCountWordsInStringMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit FedoseevCountWordsInStringMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static int count_words_mpi_impl(const char *text, int text_length, int process_id, int num_processes);
};

}  // namespace fedoseev_count_words_in_string
