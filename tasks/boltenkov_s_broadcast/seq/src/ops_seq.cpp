#include "boltenkov_s_broadcast/seq/include/ops_seq.hpp"

#include <cmath>
#include <limits>
#include <vector>

#include "boltenkov_s_broadcast/common/include/common.hpp"

namespace boltenkov_s_broadcast {

BoltenkovSBroadcastkSEQ::BoltenkovSBroadcastkSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == std::get<0>(in)) {
    GetInput() = in;
  } else {
    GetInput() = InType{};
  }
  GetOutput() = std::make_tuple(-1, -1, nullptr);
}

MPI_Datatype BoltenkovSBroadcastkSEQ::getTypeData(const int &ind_data_type) {
  if (ind_data_type == 0) {
    return MPI_INT;
  } else if (ind_data_type == 1) {
    return MPI_FLOAT;
  }
  return MPI_DOUBLE;
}

bool BoltenkovSBroadcastkSEQ::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == std::get<0>(GetInput())) {
    int size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    return std::get<0>(GetInput()) >= 0 && std::get<0>(GetInput()) < size && std::get<2>(GetInput()) >= 0 &&
           std::get<3>(GetInput()) != nullptr && std::get<1>(GetInput()) >= 0 && std::get<1>(GetInput()) < 3;
  }
  return true;
}

bool BoltenkovSBroadcastkSEQ::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == std::get<0>(GetInput())) {
    int ind_data_type = std::get<1>(GetInput());
    mpi_type = getTypeData(ind_data_type);
  }
  return true;
}

bool BoltenkovSBroadcastkSEQ::RunImpl() {
  int res_mpi =
      MPI_Bcast(std::get<3>(GetInput()), std::get<2>(GetInput()), mpi_type, std::get<0>(GetInput()), MPI_COMM_WORLD);
  return res_mpi == MPI_SUCCESS;
}

bool BoltenkovSBroadcastkSEQ::PostProcessingImpl() {
  return std::get<1>(GetOutput()) >= 0 && std::get<2>(GetOutput()) != nullptr && std::get<0>(GetOutput()) >= 0 &&
         std::get<0>(GetOutput()) <= 2;
}

}  // namespace boltenkov_s_broadcast
