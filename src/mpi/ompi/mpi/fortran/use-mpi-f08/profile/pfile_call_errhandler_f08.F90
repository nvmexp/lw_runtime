! -*- f90 -*-
!
! Copyright (c) 2010-2012 Cisco Systems, Inc.  All rights reserved.
! Copyright (c) 2009-2012 Los Alamos National Security, LLC.
!               All Rights reserved.
! Copyright (c) 2018      Research Organization for Information Science
!                         and Technology (RIST).  All rights reserved.
! $COPYRIGHT$

subroutine PMPI_File_call_errhandler_f08(fh,errorcode,ierror)
   use :: mpi_f08_types, only : MPI_File
   use :: ompi_mpifh_bindings, only : ompi_file_call_errhandler_f
   implicit none
   TYPE(MPI_File), INTENT(IN) :: fh
   INTEGER, INTENT(IN) :: errorcode
   INTEGER, OPTIONAL, INTENT(OUT) :: ierror
   integer :: c_ierror

   call ompi_file_call_errhandler_f(fh%MPI_VAL,errorcode,c_ierror)
   if (present(ierror)) ierror = c_ierror

end subroutine PMPI_File_call_errhandler_f08
