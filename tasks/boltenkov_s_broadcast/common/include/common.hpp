#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace boltenkov_s_broadcast {

using InType = std::tuple<int, int, int, std::vector<char>>;  // root, data type, cnt item, data
using OutType = std::tuple<int, int, std::vector<char>>;      // data type, cnt item, data
using TestType = std::string;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace boltenkov_s_broadcast
