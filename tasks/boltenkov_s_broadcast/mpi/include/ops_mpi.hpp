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
  MPI_Datatype GetTypeData(const int &);
  static int GetIndTypeData(MPI_Datatype);
  static bool CheckLeftChild(int, int, MPI_Comm);
  static bool CheckRightChild(int, int, MPI_Comm);
  int MyBcast(void *, int, MPI_Datatype, int, MPI_Comm);
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace boltenkov_s_broadcast
