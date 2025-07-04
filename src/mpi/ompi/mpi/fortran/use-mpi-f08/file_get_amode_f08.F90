! -*- f90 -*-
!
! Copyright (c) 2010-2012 Cisco Systems, Inc.  All rights reserved.
! Copyright (c) 2009-2012 Los Alamos National Security, LLC.
!               All Rights reserved.
! Copyright (c) 2018      Research Organization for Information Science
!                         and Technology (RIST).  All rights reserved.
! $COPYRIGHT$

subroutine MPI_File_get_amode_f08(fh,amode,ierror)
   use :: mpi_f08_types, only : MPI_File
   use :: ompi_mpifh_bindings, only : ompi_file_get_amode_f
   implicit none
   TYPE(MPI_File), INTENT(IN) :: fh
   INTEGER, INTENT(OUT) :: amode
   INTEGER, OPTIONAL, INTENT(OUT) :: ierror
   integer :: c_ierror

   call ompi_file_get_amode_f(fh%MPI_VAL,amode,c_ierror)
   if (present(ierror)) ierror = c_ierror

end subroutine MPI_File_get_amode_f08
