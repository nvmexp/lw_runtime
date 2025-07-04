! -*- f90 -*-
!
! Copyright (c) 2009-2012 Cisco Systems, Inc.  All rights reserved.
! Copyright (c) 2009-2012 Los Alamos National Security, LLC.
!                         All rights reserved.
! Copyright (c) 2018      Research Organization for Information Science
!                         and Technology (RIST).  All rights reserved.
! $COPYRIGHT$

#include "ompi/mpi/fortran/configure-fortran-output.h"

subroutine MPI_Reduce_local_f08(inbuf,inoutbuf,count,datatype,op,ierror)
   use :: mpi_f08_types, only : MPI_Datatype, MPI_Op
   use :: ompi_mpifh_bindings, only : ompi_reduce_local_f
   implicit none
   OMPI_FORTRAN_IGNORE_TKR_TYPE, INTENT(IN) :: inbuf
   OMPI_FORTRAN_IGNORE_TKR_TYPE :: inoutbuf
   INTEGER, INTENT(IN) :: count
   TYPE(MPI_Datatype), INTENT(IN) :: datatype
   TYPE(MPI_Op), INTENT(IN) :: op
   INTEGER, OPTIONAL, INTENT(OUT) :: ierror
   integer :: c_ierror

   call ompi_reduce_local_f(inbuf,inoutbuf,count,datatype%MPI_VAL,op%MPI_VAL,c_ierror)
   if (present(ierror)) ierror = c_ierror

end subroutine MPI_Reduce_local_f08
