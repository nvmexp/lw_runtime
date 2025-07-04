! -*- f90 -*-
!
! Copyright (c) 2010-2012 Cisco Systems, Inc.  All rights reserved.
! Copyright (c) 2009-2012 Los Alamos National Security, LLC.
!               All Rights reserved.
! Copyright (c) 2018      Research Organization for Information Science
!                         and Technology (RIST).  All rights reserved.
! $COPYRIGHT$

subroutine PMPI_Unpublish_name_f08(service_name,info,port_name,ierror)
   use :: mpi_f08_types, only : MPI_Info
   use :: ompi_mpifh_bindings, only : ompi_unpublish_name_f
   implicit none
   CHARACTER(LEN=*), INTENT(IN) :: service_name, port_name
   TYPE(MPI_Info), INTENT(IN) :: info
   INTEGER, OPTIONAL, INTENT(OUT) :: ierror
   integer :: c_ierror

   call ompi_unpublish_name_f(service_name,info%MPI_VAL,port_name,c_ierror, &
                              len(service_name), len(port_name))
   if (present(ierror)) ierror = c_ierror

end subroutine PMPI_Unpublish_name_f08
