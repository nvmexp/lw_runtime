! -*- f90 -*-
!
! Copyright (c) 2009-2012 Cisco Systems, Inc.  All rights reserved.
! Copyright (c) 2009-2012 Los Alamos National Security, LLC.
!                         All rights reserved.
! Copyright (c) 2018      Research Organization for Information Science
!                         and Technology (RIST).  All rights reserved.
! $COPYRIGHT$

subroutine PMPI_Waitany_f08(count,array_of_requests,index,status,ierror)
   use :: mpi_f08_types, only : MPI_Request, MPI_Status
   use :: ompi_mpifh_bindings, only : ompi_waitany_f
   implicit none
   INTEGER, INTENT(IN) :: count
   TYPE(MPI_Request), INTENT(INOUT) :: array_of_requests(count)
   INTEGER, INTENT(OUT) :: index
   TYPE(MPI_Status), INTENT(OUT) :: status
   INTEGER, OPTIONAL, INTENT(OUT) :: ierror
   integer :: c_ierror

   call ompi_waitany_f(count,array_of_requests(:)%MPI_VAL,index,status,c_ierror)
   if (present(ierror)) ierror = c_ierror

end subroutine PMPI_Waitany_f08
