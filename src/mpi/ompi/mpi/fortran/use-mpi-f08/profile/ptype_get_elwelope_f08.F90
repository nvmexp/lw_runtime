! -*- f90 -*-
!
! Copyright (c) 2009-2012 Cisco Systems, Inc.  All rights reserved.
! Copyright (c) 2009-2012 Los Alamos National Security, LLC.
!                         All rights reserved.
! Copyright (c) 2018      Research Organization for Information Science
!                         and Technology (RIST).  All rights reserved.
! $COPYRIGHT$

subroutine PMPI_Type_get_elwelope_f08(datatype,num_integers,num_addresses, &
                                      num_datatypes,combiner,ierror)
   use :: mpi_f08_types, only : MPI_Datatype, MPI_ADDRESS_KIND
   use :: ompi_mpifh_bindings, only : ompi_type_get_elwelope_f
   implicit none
   TYPE(MPI_Datatype), INTENT(IN) :: datatype
   INTEGER, INTENT(OUT) :: num_integers, num_addresses, num_datatypes, combiner
   INTEGER, OPTIONAL, INTENT(OUT) :: ierror
   integer :: c_ierror

   call ompi_type_get_elwelope_f(datatype%MPI_VAL,num_integers,num_addresses, &
                                 num_datatypes,combiner,c_ierror)
   if (present(ierror)) ierror = c_ierror

end subroutine PMPI_Type_get_elwelope_f08
