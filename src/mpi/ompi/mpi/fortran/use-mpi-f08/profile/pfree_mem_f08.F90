! -*- f90 -*-
!
! Copyright (c) 2010-2012 Cisco Systems, Inc.  All rights reserved.
! Copyright (c) 2009-2012 Los Alamos National Security, LLC.
!                         All Rights reserved.
! Copyright (c) 2015-2018 Research Organization for Information Science
!                         and Technology (RIST). All rights reserved.
! $COPYRIGHT$

#include "ompi/mpi/fortran/configure-fortran-output.h"

subroutine PMPI_Free_mem_f08(base,ierror)
   use :: ompi_mpifh_bindings, only : ompi_free_mem_f
   implicit none
   OMPI_FORTRAN_IGNORE_TKR_TYPE, INTENT(IN) :: base
   INTEGER, OPTIONAL, INTENT(OUT) :: ierror
   integer :: c_ierror

   call ompi_free_mem_f(base,c_ierror)
   if (present(ierror)) ierror = c_ierror

end subroutine PMPI_Free_mem_f08
