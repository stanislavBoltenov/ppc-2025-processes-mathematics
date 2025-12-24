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
  double CalcCoef(const std::vector<double> &args, const std::vector<std::pair<double, double>> &limits);
  void CalcPoints(const int &n, const int &ind_cur_args, const std::vector<double> &h,
                  std::unique_ptr<std::queue<std::vector<double>>> &args);
  double CalcIntegral(const int &n, const int &cnt_limits, const std::vector<std::pair<double, double>> &limits,
                      double (*func)(std::vector<double>));
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace boltenkov_s_clac_integral_trapezoidal
