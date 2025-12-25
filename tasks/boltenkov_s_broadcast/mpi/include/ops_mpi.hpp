#pragma once

#include <mpi.h>

#include "boltenkov_s_broadcast/common/include/common.hpp"
#include "task/include/task.hpp"

namespace boltenkov_s_broadcast {

class BoltenkovSBroadcastkMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit BoltenkovSBroadcastkMPI(const InType &in);

 private:
  MPI_Datatype mpi_type_;
  static MPI_Datatype GetTypeData(const int &ind_data_type);
  static int GetIndTypeData(MPI_Datatype datatype);
  static bool CheckLeftChild(int left_child, int shift_left_child, MPI_Comm comm);
  static bool CheckRightChild(int right_child, int shift_right_child, MPI_Comm comm);
  static int MyBcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm);
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace boltenkov_s_broadcast
