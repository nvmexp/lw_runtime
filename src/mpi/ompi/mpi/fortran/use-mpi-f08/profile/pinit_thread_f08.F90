! -*- f90 -*-
!
! Copyright (c) 2010-2012 Cisco Systems, Inc.  All rights reserved.
! Copyright (c) 2009-2012 Los Alamos National Security, LLC.
!               All Rights reserved.
! Copyright (c) 2018      Research Organization for Information Science
!                         and Technology (RIST).  All rights reserved.
! $COPYRIGHT$

subroutine PMPI_Init_thread_f08(required,provided,ierror)
   use :: ompi_mpifh_bindings, only : ompi_init_thread_f
   implicit none
   INTEGER, INTENT(IN) :: required
   INTEGER, INTENT(OUT) :: provided
   INTEGER, OPTIONAL, INTENT(OUT) :: ierror
   integer :: c_ierror

   call ompi_init_thread_f(required,provided,c_ierror)
   if (present(ierror)) ierror = c_ierror

end subroutine PMPI_Init_thread_f08
