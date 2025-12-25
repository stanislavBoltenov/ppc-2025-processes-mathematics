#pragma once

#include <memory>
#include <queue>
#include <utility>
#include <vector>

#include "boltenkov_s_clac_integral_trapezoidal/common/include/common.hpp"
#include "task/include/task.hpp"

namespace boltenkov_s_clac_integral_trapezoidal {

class BoltenkovSCalcIntegralkMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit BoltenkovSCalcIntegralkMPI(const InType &in);

 private:
  int sign_integral_;
  static double CalcCoef(const std::vector<double> &args, const std::vector<std::pair<double, double>> &limits);
  static void CalcPoints(const int &n, const int &ind_cur_args, const std::vector<double> &h,
                         std::unique_ptr<std::queue<std::vector<double>>> &args);
  double CalcIntegral(const int &cnt_limits, const std::vector<std::pair<double, double>> &limits,
                      const std::vector<double> &h, double (*func)(std::vector<double>));
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace boltenkov_s_clac_integral_trapezoidal
