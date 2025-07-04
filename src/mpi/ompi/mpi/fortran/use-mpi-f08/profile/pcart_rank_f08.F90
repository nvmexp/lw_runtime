! -*- f90 -*-
!
! Copyright (c) 2010-2012 Cisco Systems, Inc.  All rights reserved.
! Copyright (c) 2009-2012 Los Alamos National Security, LLC.
!                         All rights reserved.
! Copyright (c) 2018      Research Organization for Information Science
!                         and Technology (RIST).  All rights reserved.
! $COPYRIGHT$

subroutine PMPI_Cart_rank_f08(comm,coords,rank,ierror)
   use :: mpi_f08_types, only : MPI_Comm
   use :: ompi_mpifh_bindings, only : ompi_cart_rank_f
   implicit none
   TYPE(MPI_Comm), INTENT(IN) :: comm
   INTEGER, INTENT(IN) :: coords(*)
   INTEGER, INTENT(OUT) :: rank
   INTEGER, OPTIONAL, INTENT(OUT) :: ierror
   integer :: c_ierror

   call ompi_cart_rank_f(comm%MPI_VAL,coords,rank,c_ierror)
   if (present(ierror)) ierror = c_ierror

end subroutine PMPI_Cart_rank_f08
