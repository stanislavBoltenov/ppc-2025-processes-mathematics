#pragma once

#include <mpi.h>

#include "boltenkov_s_broadcast/common/include/common.hpp"
#include "task/include/task.hpp"

namespace boltenkov_s_broadcast {

class BoltenkovSBroadcastkSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  ~BoltenkovSBroadcastkSEQ();
  explicit BoltenkovSBroadcastkSEQ(const InType &in);

 private:
  MPI_Datatype mpi_type;
  MPI_Datatype getTypeData(const int &);
  int getIndTypeData(MPI_Datatype);
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace boltenkov_s_broadcast
