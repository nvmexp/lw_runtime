! -*- f90 -*-
!
! Copyright (c) 2009-2012 Cisco Systems, Inc.  All rights reserved.
! Copyright (c) 2009-2013 Los Alamos National Security, LLC.
!                         All rights reserved.
! Copyright (c) 2018      Research Organization for Information Science
!                         and Technology (RIST).  All rights reserved.
! Copyright (c) 2018      FUJITSU LIMITED.  All rights reserved.
! $COPYRIGHT$

#include "ompi/mpi/fortran/configure-fortran-output.h"

subroutine MPI_Neighbor_allgatherv_f08(sendbuf,sendcount,sendtype,recvbuf,recvcounts,&
                                       displs,recvtype,comm,ierror)
   use :: mpi_f08_types, only : MPI_Datatype, MPI_Comm
   use :: ompi_mpifh_bindings, only : ompi_neighbor_allgatherv_f
   implicit none
   OMPI_FORTRAN_IGNORE_TKR_TYPE, INTENT(IN) :: sendbuf
   OMPI_FORTRAN_IGNORE_TKR_TYPE :: recvbuf
   INTEGER, INTENT(IN) :: sendcount
   INTEGER, INTENT(IN) :: recvcounts(*), displs(*)
   TYPE(MPI_Datatype), INTENT(IN) :: sendtype
   TYPE(MPI_Datatype), INTENT(IN) :: recvtype
   TYPE(MPI_Comm), INTENT(IN) :: comm
   INTEGER, OPTIONAL, INTENT(OUT) :: ierror
   integer :: c_ierror

   call ompi_neighbor_allgatherv_f(sendbuf,sendcount,sendtype%MPI_VAL,recvbuf,recvcounts,&
                                   displs,recvtype%MPI_VAL,comm%MPI_VAL,c_ierror)
   if (present(ierror)) ierror = c_ierror

end subroutine MPI_Neighbor_allgatherv_f08
