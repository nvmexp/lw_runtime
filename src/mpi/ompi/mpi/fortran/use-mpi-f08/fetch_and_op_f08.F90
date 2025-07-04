! -*- f90 -*-
!
! Copyright (c) 2010-2014 Cisco Systems, Inc.  All rights reserved.
! Copyright (c) 2009-2014 Los Alamos National Security, LLC.
!               All Rights reserved.
! Copyright (c) 2018      Research Organization for Information Science
!                         and Technology (RIST).  All rights reserved.
! $COPYRIGHT$

#include "ompi/mpi/fortran/configure-fortran-output.h"

subroutine MPI_Fetch_and_op_f08(origin_addr,result_addr,datatype,target_rank, &
                                target_disp,op,win,ierror)
   use :: mpi_f08_types, only : MPI_Datatype, MPI_Op, MPI_Win, MPI_ADDRESS_KIND
   use :: ompi_mpifh_bindings, only : ompi_fetch_and_op_f
   implicit none
   OMPI_FORTRAN_IGNORE_TKR_TYPE, INTENT(IN), ASYNCHRONOUS :: origin_addr
   OMPI_FORTRAN_IGNORE_TKR_TYPE, ASYNCHRONOUS :: result_addr
   TYPE(MPI_Datatype), INTENT(IN) :: datatype
   INTEGER, INTENT(IN) :: target_rank
   INTEGER(MPI_ADDRESS_KIND), INTENT(IN) :: target_disp
   TYPE(MPI_Op), INTENT(IN) :: op
   TYPE(MPI_Win), INTENT(IN) :: win
   INTEGER, OPTIONAL, INTENT(OUT) :: ierror
   integer :: c_ierror

   call ompi_fetch_and_op_f(origin_addr,result_addr,datatype%MPI_VAL,target_rank,&
                            target_disp,op%MPI_VAL,win%MPI_VAL,c_ierror)
   if (present(ierror)) ierror = c_ierror

end subroutine MPI_Fetch_and_op_f08
