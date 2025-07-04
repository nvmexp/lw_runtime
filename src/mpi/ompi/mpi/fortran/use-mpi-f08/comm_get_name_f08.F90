! -*- f90 -*-
!
! Copyright (c) 2009-2012 Cisco Systems, Inc.  All rights reserved.
! Copyright (c) 2009-2012 Los Alamos National Security, LLC.
!                         All rights reserved.
! Copyright (c) 2018      Research Organization for Information Science
!                         and Technology (RIST).  All rights reserved.
! $COPYRIGHT$

subroutine MPI_Comm_get_name_f08(comm,comm_name,resultlen,ierror)
   use :: mpi_f08_types, only : MPI_Comm, MPI_MAX_OBJECT_NAME
   use :: ompi_mpifh_bindings, only : ompi_comm_get_name_f
   implicit none
   TYPE(MPI_Comm), INTENT(IN) :: comm
   CHARACTER(LEN=*), INTENT(OUT) :: comm_name
   INTEGER, INTENT(OUT) :: resultlen
   INTEGER, OPTIONAL, INTENT(OUT) :: ierror
   integer :: c_ierror

   call ompi_comm_get_name_f(comm%MPI_VAL,comm_name,resultlen,c_ierror,len(comm_name))
   if (present(ierror)) ierror = c_ierror

end subroutine MPI_Comm_get_name_f08
