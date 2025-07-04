! -*- f90 -*-
!
! Copyright (c) 2009-2012 Cisco Systems, Inc.  All rights reserved.
! Copyright (c) 2009-2012 Los Alamos National Security, LLC.
!                         All rights reserved.
! Copyright (c) 2018      Research Organization for Information Science
!                         and Technology (RIST).  All rights reserved.
! $COPYRIGHT$

subroutine MPI_Group_size_f08(group,size,ierror)
   use :: mpi_f08_types, only : MPI_Group
   use :: ompi_mpifh_bindings, only : ompi_group_size_f
   implicit none
   TYPE(MPI_Group), INTENT(IN) :: group
   INTEGER, INTENT(OUT) :: size
   INTEGER, OPTIONAL, INTENT(OUT) :: ierror
   integer :: c_ierror

   call ompi_group_size_f(group%MPI_VAL,size,c_ierror)
   if (present(ierror)) ierror = c_ierror

end subroutine MPI_Group_size_f08
