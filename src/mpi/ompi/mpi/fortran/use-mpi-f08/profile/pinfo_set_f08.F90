! -*- f90 -*-
!
! Copyright (c) 2010-2012 Cisco Systems, Inc.  All rights reserved.
! Copyright (c) 2009-2012 Los Alamos National Security, LLC.
!               All Rights reserved.
! Copyright (c) 2018      Research Organization for Information Science
!                         and Technology (RIST).  All rights reserved.
! $COPYRIGHT$

subroutine PMPI_Info_set_f08(info,key,value,ierror)
   use :: mpi_f08_types, only : MPI_Info
   use :: ompi_mpifh_bindings, only : ompi_info_set_f
   implicit none
   TYPE(MPI_Info), INTENT(IN) :: info
   CHARACTER(LEN=*), INTENT(IN) :: key, value
   INTEGER, OPTIONAL, INTENT(OUT) :: ierror
   integer :: c_ierror

   call ompi_info_set_f(info%MPI_VAL,key,value,c_ierror,len(key),len(value))
   if (present(ierror)) ierror = c_ierror

end subroutine PMPI_Info_set_f08
