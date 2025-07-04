! -*- f90 -*-
!
! Copyright (c) 2010-2012 Cisco Systems, Inc.  All rights reserved.
! Copyright (c) 2009-2012 Los Alamos National Security, LLC.
!               All Rights reserved.
! Copyright (c) 2018      Research Organization for Information Science
!                         and Technology (RIST).  All rights reserved.
! $COPYRIGHT$

subroutine MPI_Status_set_elements_f08(status,datatype,count,ierror)
   use :: mpi_f08_types, only : MPI_Status, MPI_Datatype
   use :: ompi_mpifh_bindings, only : ompi_status_set_elements_f
   implicit none
   TYPE(MPI_Status), INTENT(INOUT) :: status
   TYPE(MPI_Datatype), INTENT(IN) :: datatype
   INTEGER, INTENT(IN) :: count
   INTEGER, OPTIONAL, INTENT(OUT) :: ierror
   integer :: c_ierror

   call ompi_status_set_elements_f(status,datatype%MPI_VAL,count,c_ierror)
   if (present(ierror)) ierror = c_ierror

end subroutine MPI_Status_set_elements_f08
