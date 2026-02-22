#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "kichanova_k_increase_contrast/common/include/common.hpp"
#include "kichanova_k_increase_contrast/mpi/include/ops_mpi.hpp"
#include "kichanova_k_increase_contrast/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace kichanova_k_increase_contrast {

class KichanovaKIncreaseContrastFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    auto params = GetParam();
    auto test_params = std::get<static_cast<size_t>(ppc::util::GTestParamIndex::kTestParams)>(params);
    auto test_name = std::get<1>(test_params);

    if (test_name == "small") {
      input_data_.width = 4;
      input_data_.height = 4;
      input_data_.channels = 3;
      input_data_.pixels.resize(4 * 4 * 3);
      std::fill(input_data_.pixels.begin(), input_data_.pixels.end(), 128);

    } else if (test_name == "gradient") {
      input_data_.width = 8;
      input_data_.height = 8;
      input_data_.channels = 3;
      input_data_.pixels.resize(8 * 8 * 3);

      for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 8; ++x) {
          size_t idx = (y * 8 + x) * 3;
          input_data_.pixels[idx] = x * 32;
          input_data_.pixels[idx + 1] = y * 32;
          input_data_.pixels[idx + 2] = (x + y) * 16;
        }
      }

    } else if (test_name == "mpi_edge") {
      input_data_.width = 3;
      input_data_.height = 1;
      input_data_.channels = 3;
      input_data_.pixels = {50, 60, 70, 150, 160, 170, 200, 210, 220};

    } else {
      int width = -1;
      int height = -1;
      int channels = -1;

      std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_kichanova_k_increase_contrast, "pic.ppm");
      auto *data = stbi_load(abs_path.c_str(), &width, &height, &channels, STBI_rgb);

      input_data_.width = width;
      input_data_.height = height;
      input_data_.channels = channels;
      size_t total_pixels = width * height * channels;
      input_data_.pixels.assign(data, data + total_pixels);
      stbi_image_free(data);
    }

    KichanovaKIncreaseContrastSEQ seq_task(input_data_);

    if (seq_task.Validation()) {
      seq_task.PreProcessing();
      seq_task.Run();
      seq_task.PostProcessing();

      ref_output_ = seq_task.GetOutput();
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.pixels.empty()) {
      return false;
    }

    return output_data.pixels == ref_output_.pixels;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType ref_output_;
};

namespace {

TEST_P(KichanovaKIncreaseContrastFuncTests, IncreaseContrast) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 4> kTestParam = {std::make_tuple(1, "small"), std::make_tuple(2, "gradient"),
                                            std::make_tuple(3, "mpi_edge"), std::make_tuple(4, "real_image")};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<KichanovaKIncreaseContrastMPI, InType>(
                                               kTestParam, PPC_SETTINGS_kichanova_k_increase_contrast),
                                           ppc::util::AddFuncTask<KichanovaKIncreaseContrastSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_kichanova_k_increase_contrast));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = KichanovaKIncreaseContrastFuncTests::PrintFuncTestName<KichanovaKIncreaseContrastFuncTests>;

INSTANTIATE_TEST_SUITE_P(ContrastTests, KichanovaKIncreaseContrastFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace kichanova_k_increase_contrast
