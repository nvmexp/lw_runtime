! -*- f90 -*-
!
! Copyright (c) 2012      The University of Tennessee and The University
!                         of Tennessee Research Foundation.  All rights
!                         reserved.
! Copyright (c) 2012      Inria.  All rights reserved.
! Copyright (c) 2012 Cisco Systems, Inc.  All rights reserved.
! Copyright (c) 2018      Research Organization for Information Science
!                         and Technology (RIST).  All rights reserved.
! $COPYRIGHT$

subroutine PMPI_Type_create_hindexed_block_f08(count,blocklength, &
                           array_of_displacements,oldtype,newtype,ierror)
   use :: mpi_f08_types, only : MPI_Datatype, MPI_ADDRESS_KIND
   use :: ompi_mpifh_bindings, only : ompi_type_create_hindexed_block_f
   implicit none
   INTEGER, INTENT(IN) :: count, blocklength
   INTEGER(MPI_ADDRESS_KIND), INTENT(IN) :: array_of_displacements(count)
   TYPE(MPI_Datatype), INTENT(IN) :: oldtype
   TYPE(MPI_Datatype), INTENT(OUT) :: newtype
   INTEGER, OPTIONAL, INTENT(OUT) :: ierror
   integer :: c_ierror

   call ompi_type_create_hindexed_block_f(count,blocklength,array_of_displacements, &
                                         oldtype%MPI_VAL,newtype%MPI_VAL,c_ierror)
   if (present(ierror)) ierror = c_ierror

end subroutine PMPI_Type_create_hindexed_block_f08
