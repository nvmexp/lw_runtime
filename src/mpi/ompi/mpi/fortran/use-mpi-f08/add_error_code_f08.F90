! -*- f90 -*-
!
! Copyright (c) 2010-2012 Cisco Systems, Inc.  All rights reserved.
! Copyright (c) 2009-2012 Los Alamos National Security, LLC.
!                         All rights reserved.
! Copyright (c) 2018      Research Organization for Information Science
!                         and Technology (RIST).  All rights reserved.
! $COPYRIGHT$

subroutine MPI_Add_error_code_f08(errorclass,errorcode,ierror)
   use :: ompi_mpifh_bindings, only : ompi_add_error_code_f
   implicit none
   INTEGER, INTENT(IN) :: errorclass
   INTEGER, INTENT(OUT) :: errorcode
   INTEGER, OPTIONAL, INTENT(OUT) :: ierror
   integer :: c_ierror

   call ompi_add_error_code_f(errorclass,errorcode,c_ierror)
   if (present(ierror)) ierror = c_ierror

end subroutine MPI_Add_error_code_f08
