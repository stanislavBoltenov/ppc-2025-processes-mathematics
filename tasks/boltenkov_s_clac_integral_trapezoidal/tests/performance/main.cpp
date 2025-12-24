#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

#include "boltenkov_s_clac_integral_trapezoidal/common/include/common.hpp"
#include "boltenkov_s_clac_integral_trapezoidal/mpi/include/ops_mpi.hpp"
#include "boltenkov_s_clac_integral_trapezoidal/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace boltenkov_s_clac_integral_trapezoidal {

class BoltenkovSCalcIntegralRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;
  OutType expected_output_;
  double eps_;

  static double F3(std::vector<double> args) {
    double res = 0.0;
    for (std::size_t i = 0; i < args.size(); i++) {
      res += args[i] * args[i];
    }
    return res;
  }

  void SetUp() override {
    TestType test_params = std::make_tuple(
        std::make_tuple(1 << 8, 3, std::vector<std::pair<double, double>>({{0., 1.}, {0., 1.}, {0., 1.}}), F3), 1.,
        1e-5, 0);
    input_data_ = std::get<0>(test_params);
    expected_output_ = std::get<1>(test_params);
    eps_ = std::get<2>(test_params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return std::abs(output_data - expected_output_) < eps_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 public:
  BoltenkovSCalcIntegralRunPerfTestProcesses() {
    input_data_ = InType{};
    expected_output_ = OutType{};
    eps_ = 0.0;
  }
};

TEST_P(BoltenkovSCalcIntegralRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, BoltenkovSCalcIntegralkMPI, BoltenkovSCalcIntegralkSEQ>(
    PPC_SETTINGS_boltenkov_s_clac_integral_trapezoidal);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = BoltenkovSCalcIntegralRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, BoltenkovSCalcIntegralRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace boltenkov_s_clac_integral_trapezoidal
