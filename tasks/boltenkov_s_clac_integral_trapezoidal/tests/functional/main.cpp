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

#include "boltenkov_s_clac_integral_trapezoidal/common/include/common.hpp"
#include "boltenkov_s_clac_integral_trapezoidal/mpi/include/ops_mpi.hpp"
#include "boltenkov_s_clac_integral_trapezoidal/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace boltenkov_s_clac_integral_trapezoidal {

class BoltenkovSCalcIntegralRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    int grid_dim =
        static_cast<int>(std::pow(std::get<0>(std::get<0>(test_param)), std::get<1>(std::get<0>(test_param))));
    int space_dim = std::get<1>(std::get<0>(test_param));
    int test_number = std::get<3>(test_param);
    std::string str = "_grid_dimension_" + std::to_string(grid_dim) + "_dimension_of_space_" +
                      std::to_string(space_dim) + "_test_number_" + std::to_string(test_number);
    return str;
  }

 protected:
  void SetUp() override {
    TestType test_params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
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

 private:
  InType input_data_;
  OutType expected_output_;
  double eps_;
};

namespace {

TEST_P(BoltenkovSCalcIntegralRunFuncTestsProcesses, MatmulFromPic) {
  ExecuteTest(GetParam());
}

double f1(std::vector<double> args) {
  double res = 1.0;
  for (std::size_t i = 0; i < args.size(); i++) {
    res += 0.0 * args[i];
  }
  return res;
}

double f2(std::vector<double> args) {
  double res = 0.0;
  for (std::size_t i = 0; i < args.size(); i++) {
    res += args[i];
  }
  return res;
}

double f3(std::vector<double> args) {
  double res = 0.0;
  for (std::size_t i = 0; i < args.size(); i++) {
    res += args[i] * args[i];
  }
  return res;
}

const std::array<TestType, 8> kTestParam = {
    std::make_tuple(std::make_tuple(1 << 3, 1, std::vector<std::pair<double, double>>({{0., 1.}}), f1), 1., 1e-14, 0),
    std::make_tuple(std::make_tuple(1 << 3, 2, std::vector<std::pair<double, double>>({{0., 1.}, {0., 1.}}), f1), 1.,
                    1e-14, 1),
    std::make_tuple(
        std::make_tuple(1 << 3, 3, std::vector<std::pair<double, double>>({{0., 1.}, {0., 1.}, {0., 1.}}), f1), 1.,
        1e-14, 2),
    std::make_tuple(std::make_tuple(1 << 3, 1, std::vector<std::pair<double, double>>({{0., 1.}}), f2), 0.5, 1e-14, 3),
    std::make_tuple(std::make_tuple(1 << 3, 2, std::vector<std::pair<double, double>>({{0., 1.}, {0., 1.}}), f2), 1.,
                    1e-14, 4),
    std::make_tuple(
        std::make_tuple(1 << 3, 3, std::vector<std::pair<double, double>>({{0., 1.}, {0., 1.}, {0., 1.}}), f2), 1.5,
        1e-14, 5),
    std::make_tuple(
        std::make_tuple(1 << 3, 4, std::vector<std::pair<double, double>>({{0., 1.}, {0., 1.}, {0., 1.}, {0., 1.}}),
                        f2),
        2., 1e-14, 6),
    std::make_tuple(std::make_tuple(1 << 9, 2, std::vector<std::pair<double, double>>({{0., 1.}, {0., 1.}}), f3),
                    2. / 3., 1e-5, 7),
};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<BoltenkovSCalcIntegralkMPI, InType>(
                                               kTestParam, PPC_SETTINGS_boltenkov_s_clac_integral_trapezoidal),
                                           ppc::util::AddFuncTask<BoltenkovSCalcIntegralkSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_boltenkov_s_clac_integral_trapezoidal));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    BoltenkovSCalcIntegralRunFuncTestsProcesses::PrintFuncTestName<BoltenkovSCalcIntegralRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, BoltenkovSCalcIntegralRunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace boltenkov_s_clac_integral_trapezoidal
