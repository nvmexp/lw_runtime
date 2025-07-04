! -*- f90 -*-
!
! Copyright (c) 2010-2012 Cisco Systems, Inc.  All rights reserved.
! Copyright (c) 2009-2012 Los Alamos National Security, LLC.
!               All Rights reserved.
! Copyright (c) 2018      Research Organization for Information Science
!                         and Technology (RIST).  All rights reserved.
! $COPYRIGHT$

subroutine PMPI_File_get_group_f08(fh,group,ierror)
   use :: mpi_f08_types, only : MPI_File, MPI_Group
   use :: ompi_mpifh_bindings, only : ompi_file_get_group_f
   implicit none
   TYPE(MPI_File), INTENT(IN) :: fh
   TYPE(MPI_Group), INTENT(OUT) :: group
   INTEGER, OPTIONAL, INTENT(OUT) :: ierror
   integer :: c_ierror

   call ompi_file_get_group_f(fh%MPI_VAL,group%MPI_VAL,c_ierror)
   if (present(ierror)) ierror = c_ierror

end subroutine PMPI_File_get_group_f08
