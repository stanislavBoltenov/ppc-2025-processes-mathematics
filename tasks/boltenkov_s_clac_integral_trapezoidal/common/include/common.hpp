#pragma once

#include <tuple>
#include <utility>
#include <vector>

#include "task/include/task.hpp"

namespace boltenkov_s_clac_integral_trapezoidal {

using InType = std::tuple<int, int, std::vector<std::pair<double, double>>,
                          double (*)(std::vector<double>)>;  // n, cnt intervals, limits, func
using OutType = double;                                      // res calc integral
using TestType =
    std::tuple<std::tuple<int, int, std::vector<std::pair<double, double>>, double (*)(std::vector<double>)>, double,
               double, int>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace boltenkov_s_clac_integral_trapezoidal
