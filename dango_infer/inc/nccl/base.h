#ifndef DANGO_INCLUDE_NCCL_BASE_H_
#define DANGO_INCLUDE_NCCL_BASE_H_
#include "cuda_runtime.h"
#include "base/base.h"
#include "nccl.h"
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <glog/logging.h>


//nccl 一些全局ID
//初始化配置


#define NCCLCHECK(cmd)                                                         \
  do {                                                                         \
    ncclResult_t res = cmd;                                                    \
    if (res != ncclSuccess) {                                                  \
      fprintf(stderr, "Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__,   \
              ncclGetErrorString(res));                                        \
      fprintf(stderr, "Failed NCCL operation: %s\n", #cmd);                    \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)



namespace nccl
{
  
  extern int G_MPI_RANK,G_MPI_SIZE,G_LOCAL_RANK;

  extern ncclUniqueId G_NCCL_ID;

  extern ncclComm_t COMM;




  void InitNcclWithMpi(int argc, char *argv[]);

  void FinalizeNccl();

  int GetLocalRank(MPI_Comm comm);




}



#endif  