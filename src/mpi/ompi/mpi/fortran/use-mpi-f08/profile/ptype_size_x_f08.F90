! -*- f90 -*-
!
! Copyright (c) 2009-2012 Cisco Systems, Inc.  All rights reserved.
! Copyright (c) 2009-2013 Los Alamos National Security, LLC.
!                         All rights reserved.
! Copyright (c) 2015-2018 Research Organization for Information Science
!                         and Technology (RIST). All rights reserved.
! $COPYRIGHT$

subroutine PMPI_Type_size_x_f08(datatype,size,ierror)
   use :: mpi_f08_types, only : MPI_Datatype, MPI_COUNT_KIND
   use :: ompi_mpifh_bindings, only : ompi_type_size_x_f
   implicit none
   TYPE(MPI_Datatype), INTENT(IN) :: datatype
   INTEGER(MPI_COUNT_KIND), INTENT(OUT) :: size
   INTEGER, OPTIONAL, INTENT(OUT) :: ierror
   integer :: c_ierror

   call ompi_type_size_x_f(datatype%MPI_VAL,size,c_ierror)
   if (present(ierror)) ierror = c_ierror

end subroutine PMPI_Type_size_x_f08
