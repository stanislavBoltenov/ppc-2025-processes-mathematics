#include "boltenkov_s_clac_integral_trapezoidal/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <memory>
#include <queue>
#include <utility>
#include <vector>

#include "boltenkov_s_clac_integral_trapezoidal/common/include/common.hpp"

namespace boltenkov_s_clac_integral_trapezoidal {

BoltenkovSCalcIntegralkMPI::BoltenkovSCalcIntegralkMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    GetInput() = in;
  } else {
    GetInput() = InType{};
    std::get<3>(GetInput()) = std::get<3>(in);
  }
  GetOutput() = 0.0;
  sign_integral_ = 1;
}

bool BoltenkovSCalcIntegralkMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank != 0) {
    return true;
  }
  return std::get<0>(GetInput()) > 1 && std::cmp_equal(std::get<1>(GetInput()), std::get<2>(GetInput()).size());
}

bool BoltenkovSCalcIntegralkMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank != 0) {
    return true;
  }
  for (int i = 0; i < std::get<1>(GetInput()); i++) {
    if (std::get<2>(GetInput())[i].first > std::get<2>(GetInput())[i].second) {
      sign_integral_ *= -1;
      std::swap(std::get<2>(GetInput())[i].first, std::get<2>(GetInput())[i].second);
    }
  }
  return true;
}

double BoltenkovSCalcIntegralkMPI::CalcCoef(const std::vector<double> &args,
                                            const std::vector<std::pair<double, double>> &limits) {
  double coef = 1.0;
  for (std::size_t i = 0; i < args.size(); i++) {
    if (std::abs(args[i] - limits[i].first) > 1e-14 && std::abs(args[i] - limits[i].second) > 1e-14) {
      coef *= 2;
    }
  }
  return coef;
}

void BoltenkovSCalcIntegralkMPI::CalcPoints(const int &n, const int &ind_cur_args, const std::vector<double> &h,
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

double BoltenkovSCalcIntegralkMPI::CalcIntegral(const int &cnt_limits,
                                                const std::vector<std::pair<double, double>> &limits,
                                                const std::vector<double> &h, double (*func)(std::vector<double>)) {
  double per = 1.0;
  for (std::size_t i = 0; i < h.size(); i++) {
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
    CalcPoints((limits[ind_cur_args].second - limits[ind_cur_args].first) / h[ind_cur_args], ind_cur_args, h, args);
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

bool BoltenkovSCalcIntegralkMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  MPI_Bcast(&sign_integral_, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int n = 0;
  int cnt_limits = 0;
  std::vector<std::pair<double, double>> limits;

  if (rank == 0) {
    n = std::get<0>(GetInput());
    cnt_limits = std::get<1>(GetInput());
    limits = std::get<2>(GetInput());
  }

  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cnt_limits, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    limits.resize(cnt_limits);
  }

  for (int i = 0; i < cnt_limits; i++) {
    double first = 0.0;
    double second = 0.0;
    if (rank == 0) {
      first = limits[i].first;
      second = limits[i].second;
    }
    MPI_Bcast(&first, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&second, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank != 0) {
      limits[i] = {first, second};
    }
  }

  double local_a = 0.0;
  double local_b = 0.0;
  double global_a = limits[0].first;
  double global_b = limits[0].second;
  double interval_length = (global_b - global_a) / size;

  local_a = global_a + (rank * interval_length);
  local_b = global_a + ((rank + 1) * interval_length);

  std::vector<std::pair<double, double>> local_limits = limits;
  local_limits[0] = {local_a, local_b};

  std::vector<double> h(cnt_limits);
  h[0] = (local_b - local_a) * (static_cast<double>(size) / (static_cast<double>(n)));
  for (std::size_t i = 1; i < h.size(); i++) {
    h[i] = (limits[i].second - limits[i].first) / static_cast<double>(n);
  }

  double local_integral = CalcIntegral(cnt_limits, local_limits, h, std::get<3>(GetInput()));

  double global_integral = 0.0;
  MPI_Reduce(&local_integral, &global_integral, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    GetOutput() = global_integral;
  }

  MPI_Bcast(&GetOutput(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  return true;
}

bool BoltenkovSCalcIntegralkMPI::PostProcessingImpl() {
  return true;
}

}  // namespace boltenkov_s_clac_integral_trapezoidal
