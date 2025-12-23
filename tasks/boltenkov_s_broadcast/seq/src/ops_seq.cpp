#include "boltenkov_s_broadcast/seq/include/ops_seq.hpp"

#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <tuple>
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
    int cnt_byte = 0;
    if (std::get<1>(in) == static_cast<size_t>(0)) {
      cnt_byte = std::get<2>(in) * static_cast<int>(sizeof(int));
    } else if (std::get<1>(in) == static_cast<size_t>(1)) {
      cnt_byte = std::get<2>(in) * static_cast<int>(sizeof(float));
    } else if (std::get<1>(in) == static_cast<size_t>(2)) {
      cnt_byte = std::get<2>(in) * static_cast<int>(sizeof(double));
    }

    GetInput() = std::make_tuple(std::get<0>(in), std::get<1>(in), std::get<2>(in), std::vector<char>(cnt_byte));
  }
  GetOutput() = std::make_tuple(-1, -1, std::vector<char>(0));
  mpi_type_ = MPI_DOUBLE;
}

MPI_Datatype BoltenkovSBroadcastkSEQ::GetTypeData(const int &ind_data_type) {
  if (ind_data_type == 0) {
    return MPI_INT;
  }
  if (ind_data_type == 1) {
    return MPI_FLOAT;
  }
  return MPI_DOUBLE;
}

int BoltenkovSBroadcastkSEQ::GetIndTypeData(MPI_Datatype datatype) {
  if (datatype == MPI_INT) {
    return 0;
  }
  if (datatype == MPI_FLOAT) {
    return 1;
  }
  if (datatype == MPI_DOUBLE) {
    return 2;
  }
  return -1;
}

bool BoltenkovSBroadcastkSEQ::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == std::get<0>(GetInput())) {
    int size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    return std::get<0>(GetInput()) >= 0 && std::get<0>(GetInput()) < size && std::get<2>(GetInput()) >= 0 &&
           !std::get<3>(GetInput()).empty() && std::get<1>(GetInput()) >= 0 && std::get<1>(GetInput()) < 3;
  }
  return true;
}

bool BoltenkovSBroadcastkSEQ::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == std::get<0>(GetInput())) {
    int ind_data_type = std::get<1>(GetInput());
    mpi_type_ = GetTypeData(ind_data_type);
  }
  return true;
}

bool BoltenkovSBroadcastkSEQ::RunImpl() {
  int res_mpi = MPI_Bcast(static_cast<void *>(std::get<3>(GetInput()).data()), std::get<2>(GetInput()), mpi_type_,
                          std::get<0>(GetInput()), MPI_COMM_WORLD);
  GetOutput() = std::make_tuple(GetIndTypeData(mpi_type_), std::get<2>(GetInput()), std::get<3>(GetInput()));
  return res_mpi == MPI_SUCCESS;
}

bool BoltenkovSBroadcastkSEQ::PostProcessingImpl() {
  return std::get<1>(GetOutput()) >= 0 && !std::get<2>(GetOutput()).empty() && std::get<0>(GetOutput()) >= 0 &&
         std::get<0>(GetOutput()) <= 2;
}

}  // namespace boltenkov_s_broadcast
