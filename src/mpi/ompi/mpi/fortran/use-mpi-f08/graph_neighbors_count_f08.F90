! -*- f90 -*-
!
! Copyright (c) 2010-2012 Cisco Systems, Inc.  All rights reserved.
! Copyright (c) 2009-2012 Los Alamos National Security, LLC.
!                         All rights reserved.
! Copyright (c) 2018      Research Organization for Information Science
!                         and Technology (RIST).  All rights reserved.
! $COPYRIGHT$

subroutine MPI_Graph_neighbors_count_f08(comm,rank,nneighbors,ierror)
   use :: mpi_f08_types, only : MPI_Comm
   use :: ompi_mpifh_bindings, only : ompi_graph_neighbors_count_f
   implicit none
   TYPE(MPI_Comm), INTENT(IN) :: comm
   INTEGER, INTENT(IN) :: rank
   INTEGER, INTENT(OUT) :: nneighbors
   INTEGER, OPTIONAL, INTENT(OUT) :: ierror
   integer :: c_ierror

   call ompi_graph_neighbors_count_f(comm%MPI_VAL,rank,nneighbors,c_ierror)
   if (present(ierror)) ierror = c_ierror

end subroutine MPI_Graph_neighbors_count_f08
