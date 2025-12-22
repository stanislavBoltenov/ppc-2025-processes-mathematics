#pragma once

#include <mpi.h>

#include <string>

#include "boltenkov_s_broadcast/common/include/common.hpp"
#include "task/include/task.hpp"

namespace boltenkov_s_broadcast {

class BoltenkovSBroadcatskMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit BoltenkovSBroadcatskMPI(const InType &in);

 private:
  MPI_Datatype mpi_type;
  MPI_Datatype getTypeData(const int &);
  int my_bcast(void *, int, MPI_Datatype, int, MPI_Comm);
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace boltenkov_s_broadcast
