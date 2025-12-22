#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <ios>
#include <stdexcept>
#include <string>
#include <vector>

#include "boltenkov_s_broadcast/common/include/common.hpp"
#include "boltenkov_s_broadcast/mpi/include/ops_mpi.hpp"
#include "boltenkov_s_broadcast/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace boltenkov_s_broadcast {

class BoltenkovSBroadcastRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;

  void SetUp() override {
    std::string file_name = "test1.bin";
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
      if (data[i] != ((int *)std::get<3>(input_data_))[i]) {
        return false;
      }
    }
    return true;
  }

  bool equalsDataInputData(float *data, int cnt) {
    for (int i = 0; i < cnt; i++) {
      if (std::abs(data[i] - ((float *)std::get<3>(input_data_))[i]) < 1e-8) {
        return false;
      }
    }
    return true;
  }

  bool equalsDataInputData(double *data, int cnt) {
    for (int i = 0; i < cnt; i++) {
      if (std::abs(data[i] - ((double *)std::get<3>(input_data_))[i]) < 1e-14) {
        return false;
      }
    }
    return true;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    bool res = false;
    if (std::get<0>(output_data) == 0) {
      int *data = (int *)std::get<2>(output_data);
      res = equalsDataInputData(data, std::get<1>(output_data));
      free(data);
    } else if (std::get<0>(output_data) == 1) {
      float *data = (float *)std::get<2>(output_data);
      res = equalsDataInputData(data, std::get<1>(output_data));
      free(data);
    } else if (std::get<0>(output_data) == 2) {
      double *data = (double *)std::get<2>(output_data);
      res = equalsDataInputData(data, std::get<1>(output_data));
      free(data);
    }
    return res;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 public:
  ~BoltenkovSBroadcastRunPerfTestProcesses() {
    free(std::get<3>(input_data_));
  }
};

TEST_P(BoltenkovSBroadcastRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, BoltenkovSBroadcatskMPI, BoltenkovSBroadcastkSEQ>(
    PPC_SETTINGS_boltenkov_s_broadcast);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = BoltenkovSBroadcastRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, BoltenkovSBroadcastRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace boltenkov_s_broadcast
