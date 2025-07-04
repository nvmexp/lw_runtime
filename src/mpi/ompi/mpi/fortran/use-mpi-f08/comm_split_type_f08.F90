! -*- f90 -*-
!
! Copyright (c) 2009-2012 Cisco Systems, Inc.  All rights reserved.
! Copyright (c) 2009-2012 Los Alamos National Security, LLC.
!               All Rights reserved.
! Copyright (c) 2018      Research Organization for Information Science
!                         and Technology (RIST).  All rights reserved.
! $COPYRIGHT$

subroutine MPI_Comm_split_type_f08(comm,split_type,key,info,newcomm,ierror)
   use :: mpi_f08_types, only : MPI_Comm, MPI_Info
   use :: ompi_mpifh_bindings, only : ompi_comm_split_type_f
   implicit none
   TYPE(MPI_Comm), INTENT(IN) :: comm
   INTEGER, INTENT(IN) :: split_type
   INTEGER, INTENT(IN) :: key
   TYPE(MPI_Info), INTENT(IN) :: info
   TYPE(MPI_Comm), INTENT(OUT) :: newcomm
   INTEGER, OPTIONAL, INTENT(OUT) :: ierror
   integer :: c_ierror

   call ompi_comm_split_type_f(comm%MPI_VAL,split_type,key,info%MPI_VAL,newcomm%MPI_VAL,c_ierror)
   if (present(ierror)) ierror = c_ierror

end subroutine MPI_Comm_split_type_f08
