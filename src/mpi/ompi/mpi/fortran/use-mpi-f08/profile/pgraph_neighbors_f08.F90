! -*- f90 -*-
!
! Copyright (c) 2010-2012 Cisco Systems, Inc.  All rights reserved.
! Copyright (c) 2009-2012 Los Alamos National Security, LLC.
!                         All rights reserved.
! Copyright (c) 2018      Research Organization for Information Science
!                         and Technology (RIST).  All rights reserved.
! $COPYRIGHT$

subroutine PMPI_Graph_neighbors_f08(comm,rank,maxneighbors,neighbors,ierror)
   use :: mpi_f08_types, only : MPI_Comm
   use :: ompi_mpifh_bindings, only : ompi_graph_neighbors_f
   implicit none
   TYPE(MPI_Comm), INTENT(IN) :: comm
   INTEGER, INTENT(IN) :: rank, maxneighbors
   INTEGER, INTENT(OUT) :: neighbors(*)
   INTEGER, OPTIONAL, INTENT(OUT) :: ierror
   integer :: c_ierror

   call ompi_graph_neighbors_f(comm%MPI_VAL,rank,maxneighbors,neighbors,c_ierror)
   if (present(ierror)) ierror = c_ierror

end subroutine PMPI_Graph_neighbors_f08
