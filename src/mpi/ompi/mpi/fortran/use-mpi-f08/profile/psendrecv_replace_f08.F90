! -*- f90 -*-
!
! Copyright (c) 2009-2012 Cisco Systems, Inc.  All rights reserved.
! Copyright (c) 2009-2012 Los Alamos National Security, LLC.
!                         All rights reserved.
! Copyright (c) 2018      Research Organization for Information Science
!                         and Technology (RIST).  All rights reserved.
! Copyright (c) 2018      FUJITSU LIMITED.  All rights reserved.
! $COPYRIGHT$

#include "ompi/mpi/fortran/configure-fortran-output.h"

subroutine PMPI_Sendrecv_replace_f08(buf,count,datatype,dest,sendtag,source, &
                                    recvtag,comm,status,ierror)
   use :: mpi_f08_types, only : MPI_Datatype, MPI_Comm, MPI_Status
   use :: ompi_mpifh_bindings, only : ompi_sendrecv_replace_f
   implicit none
   OMPI_FORTRAN_IGNORE_TKR_TYPE :: buf
   INTEGER, INTENT(IN) :: count, dest, sendtag, source, recvtag
   TYPE(MPI_Datatype), INTENT(IN) :: datatype
   TYPE(MPI_Comm), INTENT(IN) :: comm
   TYPE(MPI_Status), INTENT(OUT) :: status
   INTEGER, OPTIONAL, INTENT(OUT) :: ierror
   integer :: c_ierror

   call ompi_sendrecv_replace_f(buf,count,datatype%MPI_VAL,dest,sendtag,source, &
                                recvtag,comm%MPI_VAL,status,c_ierror)
   if (present(ierror)) ierror = c_ierror

end subroutine PMPI_Sendrecv_replace_f08
