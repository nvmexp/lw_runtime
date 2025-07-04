! -*- f90 -*-
!
! Copyright (c) 2009-2012 Cisco Systems, Inc.  All rights reserved.
! Copyright (c) 2009-2012 Los Alamos National Security, LLC.
!                         All rights reserved.
! Copyright (c) 2018      Research Organization for Information Science
!                         and Technology (RIST).  All rights reserved.
! $COPYRIGHT$

subroutine MPI_Type_create_subarray_f08(ndims,array_of_sizes,array_of_subsizes, &
                                        array_of_starts,order,oldtype,newtype,ierror)
   use :: mpi_f08_types, only : MPI_Datatype
   use :: ompi_mpifh_bindings, only : ompi_type_create_subarray_f
   implicit none
   INTEGER, INTENT(IN) :: ndims, order
   INTEGER, INTENT(IN) :: array_of_sizes(*), array_of_subsizes(*), array_of_starts(*)
   TYPE(MPI_Datatype), INTENT(IN) :: oldtype
   TYPE(MPI_Datatype), INTENT(OUT) :: newtype
   INTEGER, OPTIONAL, INTENT(OUT) :: ierror
   integer :: c_ierror

   call ompi_type_create_subarray_f(ndims,array_of_sizes, &
                                    array_of_subsizes,array_of_starts, &
                                    order,oldtype%MPI_VAL,newtype%MPI_VAL,c_ierror)
   if (present(ierror)) ierror = c_ierror

end subroutine MPI_Type_create_subarray_f08
