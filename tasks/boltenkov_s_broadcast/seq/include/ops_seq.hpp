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
  explicit BoltenkovSBroadcastkSEQ(const InType &in);

 private:
  MPI_Datatype mpi_type_;
  static MPI_Datatype GetTypeData(const int &ind_data_type);
  static int GetIndTypeData(MPI_Datatype datatype);
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace boltenkov_s_broadcast
