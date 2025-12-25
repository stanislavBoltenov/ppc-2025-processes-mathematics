#include "boltenkov_s_clac_integral_trapezoidal/seq/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <memory>
#include <queue>
#include <utility>
#include <vector>

#include "boltenkov_s_clac_integral_trapezoidal/common/include/common.hpp"

namespace boltenkov_s_clac_integral_trapezoidal {

BoltenkovSCalcIntegralkSEQ::BoltenkovSCalcIntegralkSEQ(const InType &in) : sign_integral_(1) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool BoltenkovSCalcIntegralkSEQ::ValidationImpl() {
  return std::get<0>(GetInput()) > 1 && std::cmp_equal(std::get<1>(GetInput()), std::get<2>(GetInput()).size());
}

bool BoltenkovSCalcIntegralkSEQ::PreProcessingImpl() {
  for (int i = 0; i < std::get<1>(GetInput()); i++) {
    if (std::get<2>(GetInput())[i].first > std::get<2>(GetInput())[i].second) {
      sign_integral_ *= -1;
      std::swap(std::get<2>(GetInput())[i].first, std::get<2>(GetInput())[i].second);
    }
  }
  return true;
}

double BoltenkovSCalcIntegralkSEQ::CalcCoef(const std::vector<double> &args,
                                            const std::vector<std::pair<double, double>> &limits) {
  double coef = 1.0;
  for (std::size_t i = 0; i < args.size(); i++) {
    if (std::abs(args[i] - limits[i].first) > 1e-14 && std::abs(args[i] - limits[i].second) > 1e-14) {
      coef *= 2;
    }
  }
  return coef;
}

void BoltenkovSCalcIntegralkSEQ::CalcPoints(const int &n, const int &ind_cur_args, const std::vector<double> &h,
                                            std::unique_ptr<std::queue<std::vector<double>>> &args) {
  std::vector<double> cur_args;
  std::unique_ptr<std::queue<std::vector<double>>> args_tmp = std::make_unique<std::queue<std::vector<double>>>();
  while (!args->empty()) {
    cur_args = args->front();
    args->pop();
    args_tmp->push(cur_args);
    for (int step = 0; step < n; step++) {
      cur_args[ind_cur_args] += h[ind_cur_args];
      args_tmp->push(cur_args);
    }
    cur_args[ind_cur_args] += h[ind_cur_args];
  }
  std::swap(args, args_tmp);
}

double BoltenkovSCalcIntegralkSEQ::CalcIntegral(const int &n, const int &cnt_limits,
                                                const std::vector<std::pair<double, double>> &limits,
                                                double (*func)(std::vector<double>)) const {
  double per = 1.0;
  std::vector<double> h(cnt_limits);
  for (std::size_t i = 0; i < h.size(); i++) {
    h[i] = (limits[i].second - limits[i].first) / static_cast<double>(n);
    per *= h[i];
  }

  std::vector<double> args_init(cnt_limits);
  for (std::size_t i = 0; i < args_init.size(); i++) {
    args_init[i] = limits[i].first;
  }

  std::unique_ptr<std::queue<std::vector<double>>> args = std::make_unique<std::queue<std::vector<double>>>();
  args->push(args_init);

  double res = 0.0;
  int ind_cur_args = 0;

  while (ind_cur_args != cnt_limits) {
    CalcPoints(n, ind_cur_args, h, args);
    ind_cur_args++;
  }

  while (!args->empty()) {
    args_init = args->front();
    args->pop();
    double coef = CalcCoef(args_init, limits);
    res += func(args_init) * coef * per / std::pow(2., static_cast<double>(cnt_limits));
  }

  res *= sign_integral_;
  return res;
}

bool BoltenkovSCalcIntegralkSEQ::RunImpl() {
  GetOutput() =
      CalcIntegral(std::get<0>(GetInput()), std::get<1>(GetInput()), std::get<2>(GetInput()), std::get<3>(GetInput()));
  return true;
}

bool BoltenkovSCalcIntegralkSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace boltenkov_s_clac_integral_trapezoidal
