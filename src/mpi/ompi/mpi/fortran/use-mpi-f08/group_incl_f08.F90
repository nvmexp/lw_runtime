! -*- f90 -*-
!
! Copyright (c) 2009-2012 Cisco Systems, Inc.  All rights reserved.
! Copyright (c) 2009-2012 Los Alamos National Security, LLC.
!                         All rights reserved.
! Copyright (c) 2018      Research Organization for Information Science
!                         and Technology (RIST).  All rights reserved.
! $COPYRIGHT$

subroutine MPI_Group_incl_f08(group,n,ranks,newgroup,ierror)
   use :: mpi_f08_types, only : MPI_Group
   use :: ompi_mpifh_bindings, only : ompi_group_incl_f
   implicit none
   INTEGER, INTENT(IN) :: n
   INTEGER, INTENT(IN) :: ranks(*)
   TYPE(MPI_Group), INTENT(IN) :: group
   TYPE(MPI_Group), INTENT(OUT) :: newgroup
   INTEGER, OPTIONAL, INTENT(OUT) :: ierror
   integer :: c_ierror

   call ompi_group_incl_f(group%MPI_VAL,n,ranks,newgroup%MPI_VAL,c_ierror)
   if (present(ierror)) ierror = c_ierror

end subroutine MPI_Group_incl_f08
