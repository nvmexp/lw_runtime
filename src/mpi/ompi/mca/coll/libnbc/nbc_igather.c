/* -*- Mode: C; c-basic-offset:2 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2006      The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2006      The Technical University of Chemnitz. All
 *                         rights reserved.
 * Copyright (c) 2013      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2014-2018 Research Organization for Information Science
 *                         and Technology (RIST).  All rights reserved.
 * Copyright (c) 2015      Los Alamos National Security, LLC. All rights
 *                         reserved.
 * Copyright (c) 2017      IBM Corporation.  All rights reserved.
 * Copyright (c) 2018      FUJITSU LIMITED.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * Author(s): Torsten Hoefler <htor@cs.indiana.edu>
 *
 */
#include "nbc_internal.h"

#ifdef NBC_CACHE_SCHEDULE
/* tree comparison function for schedule cache */
int NBC_Gather_args_compare(NBC_Gather_args *a, NBC_Gather_args *b, void *param) {
  if ((a->sendbuf == b->sendbuf) &&
      (a->sendcount == b->sendcount) &&
      (a->sendtype == b->sendtype) &&
      (a->recvbuf == b->recvbuf) &&
      (a->recvcount == b->recvcount) &&
      (a->recvtype == b->recvtype) &&
      (a->root == b->root)) {
    return 0;
  }

  if( a->sendbuf < b->sendbuf ) {
    return -1;
  }

  return 1;
}
#endif

static int nbc_gather_init(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf,
                           int recvcount, MPI_Datatype recvtype, int root,
                           struct ompi_communicator_t *comm, ompi_request_t ** request,
                           struct mca_coll_base_module_2_3_0_t *module, bool persistent) {
  int rank, p, res;
  MPI_Aint rcvext = 0;
  NBC_Schedule *schedule;
  char *rbuf, inplace = 0;
  ompi_coll_libnbc_module_t *libnbc_module = (ompi_coll_libnbc_module_t*) module;

  rank = ompi_comm_rank (comm);
  if (root == rank) {
    NBC_IN_PLACE(sendbuf, recvbuf, inplace);
  }
  p = ompi_comm_size (comm);

  if (rank == root) {
      res = ompi_datatype_type_extent (recvtype, &rcvext);
      if (MPI_SUCCESS != res) {
        NBC_Error("MPI Error in ompi_datatype_type_extent() (%i)", res);
        return res;
      }
  }

  if (inplace) {
    sendcount = recvcount;
    sendtype = recvtype;
  }

#ifdef NBC_CACHE_SCHEDULE
  NBC_Gather_args *args, *found, search;

  /* search schedule in communicator specific tree */
  search.sendbuf = sendbuf;
  search.sendcount = sendcount;
  search.sendtype = sendtype;
  search.recvbuf = recvbuf;
  search.recvcount = recvcount;
  search.recvtype = recvtype;
  search.root = root;
  found = (NBC_Gather_args *) hb_tree_search ((hb_tree *) libnbc_module->NBC_Dict[NBC_GATHER],
                                              &search);
  if (NULL == found) {
#endif
    schedule = OBJ_NEW(NBC_Schedule);
    if (OPAL_UNLIKELY(NULL == schedule)) {
      return OMPI_ERR_OUT_OF_RESOURCE;
    }

    /* send to root */
    if (rank != root) {
      /* send msg to root */
      res = NBC_Sched_send(sendbuf, false, sendcount, sendtype, root, schedule, false);
      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        OBJ_RELEASE(schedule);
        return res;
      }
    } else {
      for (int i = 0 ; i < p ; ++i) {
        rbuf = (char *)recvbuf + i * recvcount * rcvext;
        if (i == root) {
          if (!inplace) {
            /* if I am the root - just copy the message */
            res = NBC_Sched_copy ((void *)sendbuf, false, sendcount, sendtype,
                                  rbuf, false, recvcount, recvtype, schedule, false);
            if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
              OBJ_RELEASE(schedule);
              return res;
            }
          }
        } else {
          /* root receives message to the right buffer */
          res = NBC_Sched_recv (rbuf, false, recvcount, recvtype, i, schedule, false);
          if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
            OBJ_RELEASE(schedule);
            return res;
          }
        }
      }
    }

    res = NBC_Sched_commit (schedule);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      OBJ_RELEASE(schedule);
      return res;
    }

#ifdef NBC_CACHE_SCHEDULE
    /* save schedule to tree */
    args = (NBC_Gather_args *) malloc (sizeof (args));
    if (NULL != args) {
      args->sendbuf = sendbuf;
      args->sendcount = sendcount;
      args->sendtype = sendtype;
      args->recvbuf = recvbuf;
      args->recvcount = recvcount;
      args->recvtype = recvtype;
      args->root = root;
      args->schedule = schedule;
      res = hb_tree_insert ((hb_tree *) libnbc_module->NBC_Dict[NBC_GATHER], args, args, 0);
      if (0 == res) {
        OBJ_RETAIN(schedule);

        /* increase number of elements for A2A */
        if (++libnbc_module->NBC_Dict_size[NBC_GATHER] > NBC_SCHED_DICT_UPPER) {
          NBC_SchedCache_dictwipe ((hb_tree *) libnbc_module->NBC_Dict[NBC_GATHER],
                                   &libnbc_module->NBC_Dict_size[NBC_GATHER]);
        }
      } else {
        NBC_Error("error in dict_insert() (%i)", res);
        free (args);
      }
    }
  } else {
    /* found schedule */
    schedule = found->schedule;
    OBJ_RETAIN(schedule);
  }
#endif

  res = NBC_Schedule_request(schedule, comm, libnbc_module, persistent, request, NULL);
  if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
    OBJ_RELEASE(schedule);
    return res;
  }

  return OMPI_SUCCESS;
}

int ompi_coll_libnbc_igather(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf,
                             int recvcount, MPI_Datatype recvtype, int root,
                             struct ompi_communicator_t *comm, ompi_request_t ** request,
                             struct mca_coll_base_module_2_3_0_t *module) {
    int res = nbc_gather_init(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root,
                              comm, request, module, false);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
    }
  
    res = NBC_Start(*(ompi_coll_libnbc_request_t **)request);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        NBC_Return_handle (*(ompi_coll_libnbc_request_t **)request);
        *request = &ompi_request_null.request;
        return res;
    }

    return OMPI_SUCCESS;
}

static int nbc_gather_inter_init (const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf,
                                  int recvcount, MPI_Datatype recvtype, int root,
                                  struct ompi_communicator_t *comm, ompi_request_t ** request,
                                  struct mca_coll_base_module_2_3_0_t *module, bool persistent) {
    int res, rsize;
    MPI_Aint rcvext = 0;
    NBC_Schedule *schedule;
    char *rbuf;
    ompi_coll_libnbc_module_t *libnbc_module = (ompi_coll_libnbc_module_t*) module;

    rsize = ompi_comm_remote_size (comm);

    if (root == MPI_ROOT) {
        res = ompi_datatype_type_extent(recvtype, &rcvext);
        if (MPI_SUCCESS != res) {
          NBC_Error("MPI Error in ompi_datatype_type_extent() (%i)", res);
          return res;
        }
    }

    schedule = OBJ_NEW(NBC_Schedule);
    if (OPAL_UNLIKELY(NULL == schedule)) {
      return OMPI_ERR_OUT_OF_RESOURCE;
    }

    /* send to root */
    if (root != MPI_ROOT && root != MPI_PROC_NULL) {
        /* send msg to root */
        res = NBC_Sched_send (sendbuf, false, sendcount, sendtype, root, schedule, false);
        if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
          OBJ_RELEASE(schedule);
          return res;
        }
    } else if (MPI_ROOT == root) {
        for (int i = 0 ; i < rsize ; ++i) {
            rbuf = ((char *)recvbuf) + (i * recvcount * rcvext);
            /* root receives message to the right buffer */
            res = NBC_Sched_recv (rbuf, false, recvcount, recvtype, i, schedule, false);
            if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
              OBJ_RELEASE(schedule);
              return res;
            }
        }
    }

    res = NBC_Sched_commit (schedule);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      OBJ_RELEASE(schedule);
      return res;
    }

    res = NBC_Schedule_request(schedule, comm, libnbc_module, persistent, request, NULL);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      OBJ_RELEASE(schedule);
      return res;
    }

    return OMPI_SUCCESS;
}

int ompi_coll_libnbc_igather_inter(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf,
                                   int recvcount, MPI_Datatype recvtype, int root,
                                   struct ompi_communicator_t *comm, ompi_request_t ** request,
                                   struct mca_coll_base_module_2_3_0_t *module) {
    int res = nbc_gather_inter_init(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root,
                                    comm, request, module, false);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
    }
  
    res = NBC_Start(*(ompi_coll_libnbc_request_t **)request);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        NBC_Return_handle (*(ompi_coll_libnbc_request_t **)request);
        *request = &ompi_request_null.request;
        return res;
    }

    return OMPI_SUCCESS;
}

int ompi_coll_libnbc_gather_init(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf,
                                 int recvcount, MPI_Datatype recvtype, int root,
                                 struct ompi_communicator_t *comm, MPI_Info info, ompi_request_t ** request,
                                 struct mca_coll_base_module_2_3_0_t *module) {
    int res = nbc_gather_init(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root,
                              comm, request, module, true);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
    }

    return OMPI_SUCCESS;
}

int ompi_coll_libnbc_gather_inter_init(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf,
                                       int recvcount, MPI_Datatype recvtype, int root,
                                       struct ompi_communicator_t *comm, MPI_Info info, ompi_request_t ** request,
                                       struct mca_coll_base_module_2_3_0_t *module) {
    int res = nbc_gather_inter_init(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root,
                                    comm, request, module, true);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
    }

    return OMPI_SUCCESS;
}
