#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <fstream>
#include <ios>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "boltenkov_s_broadcast/common/include/common.hpp"
#include "boltenkov_s_broadcast/mpi/include/ops_mpi.hpp"
#include "boltenkov_s_broadcast/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace boltenkov_s_broadcast {

class BoltenkovSBroadcastRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return test_param;
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    std::string file_name = params + ".bin";
    std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_boltenkov_s_broadcast, file_name);
    std::ifstream file_stream(abs_path, std::ios::in | std::ios::binary);
    if (!file_stream.is_open()) {
      throw std::runtime_error("Error opening file " + file_name + "!");
    }
    int root = -1;
    int data_type = -1;
    int cnt = -1;
    file_stream.read(reinterpret_cast<char *>(&root), sizeof(int));
    file_stream.read(reinterpret_cast<char *>(&data_type), sizeof(int));
    file_stream.read(reinterpret_cast<char *>(&cnt), sizeof(int));
    if (cnt < 0 || data_type < 0 || data_type > 2) {
      throw std::runtime_error("invalid input data!\n");
    }
    std::get<0>(input_data_) = root;
    std::get<1>(input_data_) = data_type;
    std::get<2>(input_data_) = cnt;
    int cnt_byte = 0;
    if (data_type == 0) {
      cnt_byte = cnt * sizeof(int);
    } else if (data_type == 1) {
      cnt_byte = cnt * sizeof(float);
    } else if (data_type == 2) {
      cnt_byte = cnt * sizeof(double);
    }
    char *arr = (char *)malloc(cnt_byte);
    file_stream.read(arr, static_cast<std::streamsize>(sizeof(int) * cnt));
    std::get<3>(input_data_) = (void *)arr;
    file_stream.close();
  }

  bool equalsDataInputData(int *data, int cnt) {
    for (int i = 0; i < cnt; i++) {
      if (data[i] != ((int *)std::get<3>(input_data_))[3]) {
        return false;
      }
    }
    return true;
  }

  bool equalsDataInputData(float *data, int cnt) {
    for (int i = 0; i < cnt; i++) {
      if (std::abs(data[i] - ((float *)std::get<3>(input_data_))[3]) < 1e-8) {
        return false;
      }
    }
    return true;
  }

  bool equalsDataInputData(double *data, int cnt) {
    for (int i = 0; i < cnt; i++) {
      if (std::abs(data[i] - ((double *)std::get<3>(input_data_))[3]) < 1e-14) {
        return false;
      }
    }
    return true;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (std::get<0>(output_data) == 0) {
      int *data = (int *)std::get<2>(output_data);
      return equalsDataInputData(data, std::get<1>(output_data));
    } else if (std::get<0>(output_data) == 1) {
      float *data = (float *)std::get<2>(output_data);
      return equalsDataInputData(data, std::get<1>(output_data));
    } else {
      double *data = (double *)std::get<2>(output_data);
      return equalsDataInputData(data, std::get<1>(output_data));
    }
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

TEST_P(BoltenkovSBroadcastRunFuncTestsProcesses, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {"test1", "test2", "test3"};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<BoltenkovSBroadcatskMPI, InType>(kTestParam, PPC_SETTINGS_boltenkov_s_broadcast),
    ppc::util::AddFuncTask<BoltenkovSBroadcastkSEQ, InType>(kTestParam, PPC_SETTINGS_boltenkov_s_broadcast));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    BoltenkovSBroadcastRunFuncTestsProcesses::PrintFuncTestName<BoltenkovSBroadcastRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, BoltenkovSBroadcastRunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace boltenkov_s_broadcast
