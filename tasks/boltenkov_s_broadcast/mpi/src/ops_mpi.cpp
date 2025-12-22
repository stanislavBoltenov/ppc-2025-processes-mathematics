#include "boltenkov_s_broadcast/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <limits>
#include <vector>

#include "boltenkov_s_broadcast/common/include/common.hpp"

namespace boltenkov_s_broadcast {

BoltenkovSBroadcatskMPI::BoltenkovSBroadcatskMPI(const InType &in) {
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

MPI_Datatype BoltenkovSBroadcatskMPI::getTypeData(const int &ind_data_type) {
  if (ind_data_type == 0) {
    return MPI_INT;
  } else if (ind_data_type == 1) {
    return MPI_FLOAT;
  }
  return MPI_DOUBLE;
}

bool BoltenkovSBroadcatskMPI::ValidationImpl() {
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

bool BoltenkovSBroadcatskMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == std::get<0>(GetInput())) {
    int ind_data_type = std::get<1>(GetInput());
    mpi_type = getTypeData(ind_data_type);
  }
  return true;
}

int BoltenkovSBroadcatskMPI::my_bcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm) {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size == 1) {
    return MPI_SUCCESS;
  }

  // for root = 0
  int shift_rank = (rank - root + size) % size;
  int shift_parent = (shift_rank == 0) ? -1 : (shift_rank - 1) / 2;
  int shift_left_child = 2 * shift_rank + 1;
  int shift_right_child = 2 * shift_rank + 2;

  // for cur root
  int parent = (shift_parent + root) % size;
  int left_child = (shift_left_child + root) % size;
  int right_child = (shift_right_child + root) % size;

  if (rank == root) {
    if (left_child < size && left_child != rank) {
      MPI_Send(buffer, count, datatype, left_child, 0, comm);
    }

    if (right_child < size && right_child != rank) {
      MPI_Send(buffer, count, datatype, right_child, 0, comm);
    }

  } else if (shift_parent >= 0 && parent < size) {
    MPI_Recv(buffer, count, datatype, parent, 0, comm, MPI_STATUS_IGNORE);

    if (shift_left_child < size && left_child != rank) {
      MPI_Send(buffer, count, datatype, left_child, 0, comm);
    }

    if (shift_right_child < size && right_child != rank) {
      MPI_Send(buffer, count, datatype, right_child, 0, comm);
    }
  }

  return MPI_SUCCESS;
}

bool BoltenkovSBroadcatskMPI::RunImpl() {
  int res_mpi =
      my_bcast(std::get<3>(GetInput()), std::get<2>(GetInput()), mpi_type, std::get<0>(GetInput()), MPI_COMM_WORLD);
  return res_mpi == MPI_SUCCESS;
}

bool BoltenkovSBroadcatskMPI::PostProcessingImpl() {
  return std::get<1>(GetOutput()) >= 0 && std::get<2>(GetOutput()) != nullptr && std::get<0>(GetOutput()) >= 0 &&
         std::get<0>(GetOutput()) <= 2;
}

}  // namespace boltenkov_s_broadcast
