#include <nccl/base.h>
namespace nccl
{
  
    int G_MPI_RANK,G_MPI_SIZE,G_LOCAL_RANK;

    ncclUniqueId G_NCCL_ID;
    ncclComm_t COMM = nullptr;

    void InitNcclWithMpi(int argc, char *argv[])
    {
        int num_gpus = 0;

        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &G_MPI_RANK);
        MPI_Comm_size(MPI_COMM_WORLD, &G_MPI_SIZE);

        LOG(INFO) << "MPI initialized - Process " << G_MPI_RANK << " of "
                  << G_MPI_SIZE << " total processes";

        if (G_MPI_RANK == 0)
            LOG(INFO) << "Starting NCCL communicator lifecycle example with "
                      << G_MPI_SIZE << " processes";

        G_LOCAL_RANK = GetLocalRank(MPI_COMM_WORLD);

        LOG(INFO)<<"MPI initialized - Process"<<G_MPI_RANK<<"of "<<G_MPI_RANK<< "total processes\n";

        CUDACHECK(cudaGetDeviceCount(&num_gpus));
        LOG(INFO)<<"Found "<<num_gpus<<" CUDA devices on this node\n";


        CHECK_NE(num_gpus, 0) << "No CUDA devices found on this node";

        CHECK_LT(G_LOCAL_RANK, num_gpus)
            << "Process " << G_MPI_RANK << " needs GPU " << G_LOCAL_RANK
            << " but only " << num_gpus << " device(s) available";


        //确实,一个进程一个cudaDevice我就不用总是切换我的Device了
        CUDACHECK(cudaSetDevice(G_LOCAL_RANK));

        if (G_MPI_RANK == 0) 
        {
            NCCLCHECK(ncclGetUniqueId(&G_NCCL_ID));
            printf("Rank 0 generated NCCL unique ID for all processes\n");
        }

        // 通过 MPI 广播将唯一 ID 分发给所有进程
        // 参数结构:  1. 将要广播的数据类型  2. 广播的数据长度 3.广播的数据类型 4.线程所在的线程组
        MPI_Bcast(&G_NCCL_ID, NCCL_UNIQUE_ID_BYTES, MPI_CHAR, 0, MPI_COMM_WORLD);
        LOG(INFO)<<"INFO: Rank "<<G_MPI_RANK<<"received NCCL unique ID\n";

        // 为当前进程创建 NCCL 通信器
        // 此处每个进程加入分布式 NCCL 通信域
        NCCLCHECK(ncclCommInitRank(&COMM, G_MPI_SIZE, G_NCCL_ID, G_MPI_RANK));
        LOG(INFO)<<"Rank "<<G_MPI_RANK<<"created NCCL communicator\n";

        // 查询通信器属性，确认配置正确
        int comm_rank, comm_size, comm_device;
        NCCLCHECK(ncclCommUserRank(COMM, &comm_rank));
        NCCLCHECK(ncclCommCount(COMM, &comm_size));
        NCCLCHECK(ncclCommCuDevice(COMM, &comm_device));

        LOG(INFO) << "MPI rank " << G_MPI_RANK << " \u2192 NCCL rank "
                  << comm_rank << "/" << comm_size
                  << " on GPU device " << comm_device;

        MPI_Barrier(MPI_COMM_WORLD);
    }


    void FinalizeNccl()
    {
        if (COMM != NULL) 
        {
            NCCLCHECK(ncclCommFinalize(COMM));
            NCCLCHECK(ncclCommDestroy(COMM));
            LOG(INFO)<<"  Rank "<<G_MPI_RANK<<" destroyed NCCL communicator\n";
        }

        //然后才销毁cudaStream

        //结束
        if (G_MPI_RANK == 0) 

            LOG(INFO)<<"\nAll NCCL communicators created and cleaned up properly!\n";
    }



    int GetLocalRank(MPI_Comm comm) 
    {
        int world_size;
        MPI_Comm_size(comm, &world_size);

        int world_rank;
        MPI_Comm_rank(comm, &world_rank);

        // 基于共享内存（节点）拆分通信器
        MPI_Comm node_comm;
        MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, world_rank, MPI_INFO_NULL,&node_comm);

        // 获取节点内的 rank 和进程数
        int node_rank, node_size;
        MPI_Comm_rank(node_comm, &node_rank);
        MPI_Comm_size(node_comm, &node_size);

        // 释放节点通信器
        MPI_Comm_free(&node_comm);

        return node_rank;
    }


}
