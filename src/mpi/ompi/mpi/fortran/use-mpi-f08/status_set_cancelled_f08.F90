! -*- f90 -*-
!
! Copyright (c) 2010-2018 Cisco Systems, Inc.  All rights reserved
! Copyright (c) 2009-2012 Los Alamos National Security, LLC.
!               All Rights reserved.
! Copyright (c) 2018      FUJITSU LIMITED.  All rights reserved.
! $COPYRIGHT$

subroutine MPI_Status_set_cancelled_f08(status,flag,ierror)
   use :: mpi_f08_types, only : MPI_Status
   implicit none
   TYPE(MPI_Status), INTENT(INOUT) :: status
   LOGICAL, INTENT(IN) :: flag
   INTEGER, OPTIONAL, INTENT(OUT) :: ierror
   integer :: c_ierror

   ! See note in mpi-f-interfaces-bind.h for why we include an
   ! interface here and call a PMPI_* subroutine below.
   interface
      subroutine PMPI_Status_set_cancelled(status, flag, ierror)
        use :: mpi_f08_types, only : MPI_Status
        type(MPI_Status), intent(inout) :: status
        logical, intent(in) :: flag
        integer, intent(out) :: ierror
      end subroutine PMPI_Status_set_cancelled
   end interface

   call PMPI_Status_set_cancelled(status,flag,c_ierror)
   if (present(ierror)) ierror = c_ierror
end subroutine MPI_Status_set_cancelled_f08
