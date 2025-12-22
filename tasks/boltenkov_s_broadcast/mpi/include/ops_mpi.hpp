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
  ~BoltenkovSBroadcatskMPI();
  explicit BoltenkovSBroadcatskMPI(const InType &in);

 private:
  MPI_Datatype mpi_type;
  MPI_Datatype getTypeData(const int &);
  int getIndTypeData(MPI_Datatype);
  int my_bcast(void *, int, MPI_Datatype, int, MPI_Comm);
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace boltenkov_s_broadcast
