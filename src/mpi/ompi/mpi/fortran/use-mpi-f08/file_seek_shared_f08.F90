! -*- f90 -*-
!
! Copyright (c) 2009-2012 Cisco Systems, Inc.  All rights reserved.
! Copyright (c) 2009-2012 Los Alamos National Security, LLC.
!               All Rights reserved.
! Copyright (c) 2018      Research Organization for Information Science
!                         and Technology (RIST).  All rights reserved.
! $COPYRIGHT$

subroutine MPI_File_seek_shared_f08(fh,offset,whence,ierror)
   use :: mpi_f08_types, only : MPI_File, MPI_OFFSET_KIND
   use :: ompi_mpifh_bindings, only : ompi_file_seek_shared_f
   implicit none
   TYPE(MPI_File), INTENT(IN) :: fh
   INTEGER(MPI_OFFSET_KIND), INTENT(IN) :: offset
   INTEGER, INTENT(IN) :: whence
   INTEGER, OPTIONAL, INTENT(OUT) :: ierror
   integer :: c_ierror

   call ompi_file_seek_shared_f(fh%MPI_VAL,offset,whence,c_ierror)
   if (present(ierror)) ierror = c_ierror

end subroutine MPI_File_seek_shared_f08
