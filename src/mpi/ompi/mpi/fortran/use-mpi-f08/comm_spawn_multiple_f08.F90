! -*- f90 -*-
!
! Copyright (c) 2010-2012 Cisco Systems, Inc.  All rights reserved.
! Copyright (c) 2009-2012 Los Alamos National Security, LLC.
!               All Rights reserved.
! Copyright (c) 2015-2018 Research Organization for Information Science
!                         and Technology (RIST). All rights reserved.
! Copyright (c) 2018      FUJITSU LIMITED.  All rights reserved.
! $COPYRIGHT$

subroutine MPI_Comm_spawn_multiple_f08(count,array_of_commands,array_of_argv, &
                                       array_of_maxprocs,array_of_info,root,  &
                                       comm,intercomm,array_of_errcodes,ierror)
   use :: mpi_f08_types, only : MPI_Info, MPI_Comm
   use :: ompi_mpifh_bindings, only : ompi_comm_spawn_multiple_f
   implicit none
   INTEGER, INTENT(IN) :: count, root
   INTEGER, INTENT(IN) :: array_of_maxprocs(count)
   CHARACTER(LEN=*), INTENT(IN) :: array_of_commands(count), array_of_argv(count, *)
   TYPE(MPI_Info), INTENT(IN) :: array_of_info(count)
   TYPE(MPI_Comm), INTENT(IN) :: comm
   TYPE(MPI_Comm), INTENT(OUT) :: intercomm
   INTEGER :: array_of_errcodes(*)
   INTEGER, OPTIONAL, INTENT(OUT) :: ierror
   integer :: c_ierror

!
! WARNING: array handling needs to be tested
!

   call ompi_comm_spawn_multiple_f(count,array_of_commands,array_of_argv, &
                                   array_of_maxprocs,array_of_info(:)%MPI_VAL,root, &
                                   comm%MPI_VAL,intercomm%MPI_VAL,array_of_errcodes,c_ierror, &
                                   len(array_of_commands), len(array_of_argv))
   if (present(ierror)) ierror = c_ierror

end subroutine MPI_Comm_spawn_multiple_f08
