! -*- f90 -*-
!
! Copyright (c) 2009-2012 Cisco Systems, Inc.  All rights reserved.
! Copyright (c) 2009-2012 Los Alamos National Security, LLC.
!                         All rights reserved.
! Copyright (c) 2018      Research Organization for Information Science
!                         and Technology (RIST).  All rights reserved.
! $COPYRIGHT$

#include "ompi/mpi/fortran/configure-fortran-output.h"

subroutine PMPI_Iscatterv_f08(sendbuf,sendcounts,displs,sendtype,recvbuf,&
                            recvcount,recvtype,root,comm,request,ierror)
   use :: mpi_f08_types, only : MPI_Datatype, MPI_Comm, MPI_Request
   use :: ompi_mpifh_bindings, only : ompi_iscatterv_f
   implicit none
   OMPI_FORTRAN_IGNORE_TKR_TYPE, INTENT(IN), ASYNCHRONOUS :: sendbuf
   OMPI_FORTRAN_IGNORE_TKR_TYPE, ASYNCHRONOUS :: recvbuf
   INTEGER, INTENT(IN) :: recvcount, root
   INTEGER, INTENT(IN), ASYNCHRONOUS :: sendcounts(*), displs(*)
   TYPE(MPI_Datatype), INTENT(IN) :: sendtype
   TYPE(MPI_Datatype), INTENT(IN) :: recvtype
   TYPE(MPI_Comm), INTENT(IN) :: comm
   TYPE(MPI_Request), INTENT(OUT) :: request
   INTEGER, OPTIONAL, INTENT(OUT) :: ierror
   integer :: c_ierror

   call ompi_iscatterv_f(sendbuf,sendcounts,displs,sendtype%MPI_VAL,recvbuf,&
                        recvcount,recvtype%MPI_VAL,root,comm%MPI_VAL,request%MPI_VAL,c_ierror)
   if (present(ierror)) ierror = c_ierror

end subroutine PMPI_Iscatterv_f08
