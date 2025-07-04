! -*- f90 -*-
!
! Copyright (c) 2009-2012 Cisco Systems, Inc.  All rights reserved.
! Copyright (c) 2009-2012 Los Alamos National Security, LLC.
!                         All rights reserved.
! Copyright (c) 2018      Research Organization for Information Science
!                         and Technology (RIST).  All rights reserved.
! $COPYRIGHT$

subroutine MPI_Type_get_extent_f08(datatype,lb,extent,ierror)
   use :: mpi_f08_types, only : MPI_Datatype, MPI_ADDRESS_KIND
   use :: ompi_mpifh_bindings, only : ompi_type_get_extent_f
   implicit none
   TYPE(MPI_Datatype), INTENT(IN) :: datatype
   INTEGER(MPI_ADDRESS_KIND), INTENT(OUT) :: lb, extent
   INTEGER, OPTIONAL, INTENT(OUT) :: ierror
   integer :: c_ierror

   call ompi_type_get_extent_f(datatype%MPI_VAL,lb,extent,c_ierror)
   if (present(ierror)) ierror = c_ierror

end subroutine MPI_Type_get_extent_f08
