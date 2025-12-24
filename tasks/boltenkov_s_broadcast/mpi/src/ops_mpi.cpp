#include "boltenkov_s_broadcast/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <tuple>
#include <vector>

#include "boltenkov_s_broadcast/common/include/common.hpp"

namespace boltenkov_s_broadcast {

BoltenkovSBroadcastkMPI::BoltenkovSBroadcastkMPI(const InType &in) {
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

    std::vector<char> vec;
    if (cnt_byte > 0) {
      vec.resize(cnt_byte);
    }

    GetInput() = std::make_tuple(std::get<0>(in), std::get<1>(in), std::get<2>(in), std::move(vec));
  }
  GetOutput() = std::make_tuple(-1, -1, std::vector<char>());
  mpi_type_ = MPI_DOUBLE;
}

MPI_Datatype BoltenkovSBroadcastkMPI::GetTypeData(const int &ind_data_type) {
  if (ind_data_type == 0) {
    return MPI_INT;
  }
  if (ind_data_type == 1) {
    return MPI_FLOAT;
  }
  return MPI_DOUBLE;
}

bool BoltenkovSBroadcastkMPI::ValidationImpl() {
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

bool BoltenkovSBroadcastkMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == std::get<0>(GetInput())) {
    int ind_data_type = std::get<1>(GetInput());
    mpi_type_ = GetTypeData(ind_data_type);
  }
  int type_int = GetIndTypeData(mpi_type_);
  MPI_Bcast(&type_int, 1, MPI_INT, std::get<0>(GetInput()), MPI_COMM_WORLD);
  mpi_type_ = GetTypeData(type_int);
  return true;
}

int BoltenkovSBroadcastkMPI::GetIndTypeData(MPI_Datatype datatype) {
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

bool BoltenkovSBroadcastkMPI::CheckLeftChild(int left_child, int shift_left_child, MPI_Comm comm) {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  return shift_left_child < size && left_child != rank;
}

bool BoltenkovSBroadcastkMPI::CheckRightChild(int right_child, int shift_right_child, MPI_Comm comm) {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  return shift_right_child < size && right_child != rank;
}

int BoltenkovSBroadcastkMPI::MyBcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm) {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  if (size == 1) {
    return MPI_SUCCESS;
  }

  // for root = 0
  int shift_rank = (rank - root + size) % size;
  int shift_parent = (shift_rank == 0) ? -1 : (shift_rank - 1) / 2;
  int shift_left_child = (2 * shift_rank) + 1;
  int shift_right_child = (2 * shift_rank) + 2;

  // for cur root
  int parent = (shift_parent + root) % size;
  int left_child = (shift_left_child + root) % size;
  int right_child = (shift_right_child + root) % size;

  if (rank != root) {
    MPI_Recv(buffer, count, datatype, parent, 0, comm, MPI_STATUS_IGNORE);
  }

  if (CheckLeftChild(left_child, shift_left_child, comm)) {
    MPI_Send(buffer, count, datatype, left_child, 0, comm);
  }

  if (CheckRightChild(right_child, shift_right_child, comm)) {
    MPI_Send(buffer, count, datatype, right_child, 0, comm);
  }

  return MPI_SUCCESS;
}

bool BoltenkovSBroadcastkMPI::RunImpl() {
  int res_mpi = MyBcast(static_cast<void *>(std::get<3>(GetInput()).data()), std::get<2>(GetInput()), mpi_type_,
                        std::get<0>(GetInput()), MPI_COMM_WORLD);
  GetOutput() = std::make_tuple(GetIndTypeData(mpi_type_), std::get<2>(GetInput()), std::get<3>(GetInput()));
  return res_mpi == MPI_SUCCESS;
}

bool BoltenkovSBroadcastkMPI::PostProcessingImpl() {
  return std::get<1>(GetOutput()) >= 0 && !std::get<2>(GetOutput()).empty() && std::get<0>(GetOutput()) >= 0 &&
         std::get<0>(GetOutput()) <= 2;
}

}  // namespace boltenkov_s_broadcast
