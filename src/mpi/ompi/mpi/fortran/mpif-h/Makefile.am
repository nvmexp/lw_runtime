#
# Copyright (c) 2004-2007 The Trustees of Indiana University and Indiana
#                         University Research and Technology
#                         Corporation.  All rights reserved.
# Copyright (c) 2004-2013 The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.
# Copyright (c) 2004-2009 High Performance Computing Center Stuttgart,
#                         University of Stuttgart.  All rights reserved.
# Copyright (c) 2004-2005 The Regents of the University of California.
#                         All rights reserved.
# Copyright (c) 2006-2015 Cisco Systems, Inc.  All rights reserved.
# Copyright (c) 2011-2013 Inria.  All rights reserved.
# Copyright (c) 2011-2013 Universite Bordeaux 1
# Copyright (c) 2013-2018 Los Alamos National Security, LLC. All rights
#                         reserved.
# Copyright (c) 2015-2017 Research Organization for Information Science
#                         and Technology (RIST). All rights reserved.
# Copyright (c) 2016      IBM Corporation.  All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

include $(top_srcdir)/Makefile.ompi-rules

SUBDIRS = profile

AM_CPPFLAGS = -DOMPI_BUILD_MPI_PROFILING=0 -DOMPI_COMPILING_FORTRAN_WRAPPERS=1

# The top directory only builds MPI_* bindings and some support
# glue. The bottom directory only builds PMPI_* bindings.  Each
# directory is built when certain conditions are met.  If the mpif.h
# bindings are disabled, nothing in the top or bottom directory is
# built.
#
# If the mpif.h MPI API bindings are enabled, the glue functions in
# the top directory are always built.
#
# The cases where the MPI API functions are build top directory needs
# to be built are:
#
# 1. When profiling is disabled
# 2. When profiling is enabled but weak symbol support is absent
#

lib_LTLIBRARIES =
CLEANFILES =
# Note that we ilwoke some OPAL functions directly in libmpi_mpifh.la,
# so we need to link in the OPAL library directly (pulling it in
# indirectly via libmpi.la does not work on all platforms).
lib@OMPI_LIBMPI_NAME@_mpifh_la_LIBADD = \
        $(top_builddir)/ompi/lib@OMPI_LIBMPI_NAME@.la \
        $(OMPI_MPIEXT_MPIFH_LIBS) \
        $(OMPI_TOP_BUILDDIR)/opal/lib@OPAL_LIB_PREFIX@open-pal.la

lib@OMPI_LIBMPI_NAME@_mpifh_la_LDFLAGS = -version-info $(libmpi_mpifh_so_version)

# Are we building the mpif.h bindings at all?
if OMPI_BUILD_FORTRAN_MPIFH_BINDINGS
# If yes, then we need to build the installable library and the glue
# colwenience library that will be sucked up into the main libmpi.
lib_LTLIBRARIES += lib@OMPI_LIBMPI_NAME@_mpifh.la
# Do we need to suck in the colwenience library from the lower
# directory?
if BUILD_PMPI_FORTRAN_MPIFH_BINDINGS_LAYER
lib@OMPI_LIBMPI_NAME@_mpifh_la_LIBADD += profile/libmpi_mpifh_pmpi.la
endif
endif

headers = \
    bindings.h \
    prototypes_mpi.h \
    status-colwersion.h

#
# These files are only built and added to libmpi_mpifh.la in certain cases.
#
lib@OMPI_LIBMPI_NAME@_mpifh_la_SOURCES =

# sizeof_f.f90 is generated based on some results from configure tests.
CLEANFILES += sizeof_f.f90

# If we're building the MPI_SIZEOF interfaces, generate the source
# code in sizeof_f.f90 (via a perl script).  Normally, we'd just add
# sizeof_f.f90 to the libmpi_mpif_la_SOURCES and be done with it.
# However, since this is the first *actual Fortran* source that we've
# added to libmpi_mpif_la_SOURCEs, this causes Libtool to use the
# *Fortran* linker to create the final libmpi_mpif.la library
# (vs. using the C linker, as it has done for years).  This exposes an
# unfortunate bug in the Absoft linker: it does not honor the order of
# -Wl arguments, which basically causes the resulting libmpi_mpif.so
# to be borked.
#
# To get around this, we compile sizeof_f.f90 into its own LT
# colwenience library, and then we LIBADD that colwenience library
# into the main libmpi_mpifh.la.  In this way, we basically trick
# Libtool into still using the C linker to create libmpi_mpif.la, and
# our -Wl flags therefore don't get re-ordered.
#
# All this being said, we can't rely on Automake/Libtool not figuring
# out this workaround indefinitely.  So Jeff advised Absoft to fix
# this bug in Sep 2014.
noinst_LTLIBRARIES=
if BUILD_FORTRAN_SIZEOF
noinst_LTLIBRARIES += libmpi_mpifh_sizeof.la
# Do not dist this file; it is generated
nodist_libmpi_mpifh_sizeof_la_SOURCES = sizeof_f.f90
lib@OMPI_LIBMPI_NAME@_mpifh_la_LIBADD += libmpi_mpifh_sizeof.la
endif

sizeof_pl = $(top_srcdir)/ompi/mpi/fortran/base/gen-mpi-sizeof.pl

sizeof_f.f90: $(top_builddir)/config.status
sizeof_f.f90: $(sizeof_pl)
sizeof_f.f90:
	$(OMPI_V_GEN) $(sizeof_pl) \
	    --impl=$@ --ierror=mandatory --mpi \
	    --maxrank=$(OMPI_FORTRAN_MAX_ARRAY_RANK) \
	    --generate=$(OMPI_FORTRAN_BUILD_SIZEOF) \
	    --real16=$(OMPI_HAVE_FORTRAN_REAL16) \
	    --complex32=$(OMPI_HAVE_FORTRAN_COMPLEX32)

if BUILD_MPI_FORTRAN_MPIFH_BINDINGS_LAYER
lib@OMPI_LIBMPI_NAME@_mpifh_la_SOURCES += \
        abort_f.c \
        add_error_class_f.c \
        add_error_code_f.c \
        add_error_string_f.c \
        aint_add_f.c \
        aint_diff_f.c \
        allgather_f.c \
        allgatherv_f.c \
        alloc_mem_f.c \
        allreduce_f.c \
        alltoall_f.c \
        alltoallv_f.c \
        alltoallw_f.c \
        attr_delete_f.c \
        attr_get_f.c \
        attr_put_f.c \
        barrier_f.c \
        bcast_f.c \
        bsend_f.c \
        bsend_init_f.c \
        buffer_attach_f.c \
        buffer_detach_f.c \
        cancel_f.c \
        cart_coords_f.c \
        cart_create_f.c \
        cartdim_get_f.c \
        cart_get_f.c \
        cart_map_f.c \
        cart_rank_f.c \
        cart_shift_f.c \
        cart_sub_f.c \
        close_port_f.c \
        comm_accept_f.c \
        comm_call_errhandler_f.c \
        comm_compare_f.c \
        comm_connect_f.c \
        comm_create_errhandler_f.c \
        comm_create_f.c \
        comm_create_group_f.c \
        comm_create_keyval_f.c \
        comm_delete_attr_f.c \
        comm_disconnect_f.c \
        comm_dup_f.c \
        comm_dup_with_info_f.c \
        comm_idup_f.c \
        comm_free_f.c \
        comm_free_keyval_f.c \
        comm_get_attr_f.c \
        comm_get_errhandler_f.c \
        comm_get_info_f.c \
        comm_get_name_f.c \
        comm_get_parent_f.c \
        comm_group_f.c \
        comm_join_f.c \
        comm_rank_f.c \
        comm_remote_group_f.c \
        comm_remote_size_f.c \
        comm_set_attr_f.c \
        comm_set_errhandler_f.c \
        comm_set_info_f.c \
        comm_set_name_f.c \
        comm_size_f.c \
        comm_spawn_f.c \
        comm_spawn_multiple_f.c \
        comm_split_f.c \
        comm_split_type_f.c \
        comm_test_inter_f.c \
        dist_graph_create_adjacent_f.c \
        dist_graph_create_f.c \
        dist_graph_neighbors_f.c \
        dist_graph_neighbors_count_f.c \
        dims_create_f.c \
        errhandler_free_f.c \
        error_class_f.c \
        error_string_f.c \
        exscan_f.c \
        f_sync_reg_f.c \
        file_call_errhandler_f.c \
        file_close_f.c \
        file_create_errhandler_f.c \
        file_delete_f.c \
        file_get_amode_f.c \
        file_get_atomicity_f.c \
        file_get_byte_offset_f.c \
        file_get_errhandler_f.c \
        file_get_group_f.c \
        file_get_info_f.c \
        file_get_position_f.c \
        file_get_position_shared_f.c \
        file_get_size_f.c \
        file_get_type_extent_f.c \
        file_get_view_f.c \
        file_iread_at_f.c \
        file_iread_f.c \
        file_iread_at_all_f.c \
        file_iread_all_f.c \
        file_iread_shared_f.c \
        file_iwrite_at_f.c \
        file_iwrite_f.c \
        file_iwrite_at_all_f.c \
        file_iwrite_all_f.c \
        file_iwrite_shared_f.c \
        file_open_f.c \
        file_preallocate_f.c \
        file_read_all_begin_f.c \
        file_read_all_end_f.c \
        file_read_all_f.c \
        file_read_at_all_begin_f.c \
        file_read_at_all_end_f.c \
        file_read_at_all_f.c \
        file_read_at_f.c \
        file_read_f.c \
        file_read_ordered_begin_f.c \
        file_read_ordered_end_f.c \
        file_read_ordered_f.c \
        file_read_shared_f.c \
        file_seek_f.c \
        file_seek_shared_f.c \
        file_set_atomicity_f.c \
        file_set_errhandler_f.c \
        file_set_info_f.c \
        file_set_size_f.c \
        file_set_view_f.c \
        file_sync_f.c \
        file_write_all_begin_f.c \
        file_write_all_end_f.c \
        file_write_all_f.c \
        file_write_at_all_begin_f.c \
        file_write_at_all_end_f.c \
        file_write_at_all_f.c \
        file_write_at_f.c \
        file_write_f.c \
        file_write_ordered_begin_f.c \
        file_write_ordered_end_f.c \
        file_write_ordered_f.c \
        file_write_shared_f.c \
        finalized_f.c \
        finalize_f.c \
        free_mem_f.c \
        gather_f.c \
        gatherv_f.c \
        get_address_f.c \
        get_count_f.c \
        get_elements_f.c \
        get_elements_x_f.c \
        get_library_version_f.c \
        get_processor_name_f.c \
        get_version_f.c \
        graph_create_f.c \
        graphdims_get_f.c \
        graph_get_f.c \
        graph_map_f.c \
        graph_neighbors_count_f.c \
        graph_neighbors_f.c \
        grequest_complete_f.c \
        grequest_start_f.c \
        group_compare_f.c \
        group_difference_f.c \
        group_excl_f.c \
        group_free_f.c \
        group_incl_f.c \
        group_intersection_f.c \
        group_range_excl_f.c \
        group_range_incl_f.c \
        group_rank_f.c \
        group_size_f.c \
        group_translate_ranks_f.c \
        group_union_f.c \
        iallgather_f.c \
        iallgatherv_f.c \
        iallreduce_f.c \
        ialltoall_f.c \
        ialltoallv_f.c \
        ialltoallw_f.c \
        ibarrier_f.c \
        ibcast_f.c \
        ibsend_f.c \
        iexscan_f.c \
        igather_f.c \
        igatherv_f.c \
        improbe_f.c \
        imrecv_f.c \
        ineighbor_allgather_f.c \
        ineighbor_allgatherv_f.c \
        ineighbor_alltoall_f.c \
        ineighbor_alltoallv_f.c \
        ineighbor_alltoallw_f.c \
        info_create_f.c \
        info_delete_f.c \
        info_dup_f.c \
        info_free_f.c \
        info_get_f.c \
        info_get_nkeys_f.c \
        info_get_nthkey_f.c \
        info_get_valuelen_f.c \
        info_set_f.c \
        init_f.c \
        initialized_f.c \
        init_thread_f.c \
        intercomm_create_f.c \
        intercomm_merge_f.c \
        iprobe_f.c \
        irecv_f.c \
        ireduce_f.c \
        ireduce_scatter_f.c \
        ireduce_scatter_block_f.c \
        irsend_f.c \
        isend_f.c \
        iscan_f.c \
        iscatter_f.c \
        iscatterv_f.c \
        issend_f.c \
        is_thread_main_f.c \
        keyval_create_f.c \
        keyval_free_f.c \
        lookup_name_f.c \
        mprobe_f.c \
        mrecv_f.c \
        neighbor_allgather_f.c \
        neighbor_allgatherv_f.c \
        neighbor_alltoall_f.c \
        neighbor_alltoallv_f.c \
        neighbor_alltoallw_f.c \
        op_commutative_f.c \
        op_create_f.c \
        open_port_f.c \
        op_free_f.c \
        pack_external_f.c \
        pack_external_size_f.c \
        pack_f.c \
        pack_size_f.c \
        pcontrol_f.c \
        probe_f.c \
        publish_name_f.c \
        query_thread_f.c \
        recv_f.c \
        recv_init_f.c \
        reduce_f.c \
        reduce_local_f.c \
        reduce_scatter_f.c \
        reduce_scatter_block_f.c \
        register_datarep_f.c \
        request_free_f.c \
        request_get_status_f.c \
        rsend_f.c \
        rsend_init_f.c \
        scan_f.c \
        scatter_f.c \
        scatterv_f.c \
        send_f.c \
        send_init_f.c \
        sendrecv_f.c \
        sendrecv_replace_f.c \
        ssend_f.c \
        ssend_init_f.c \
        startall_f.c \
        start_f.c \
        status_set_cancelled_f.c \
        status_set_elements_f.c \
        status_set_elements_x_f.c \
        testall_f.c \
        testany_f.c \
        test_cancelled_f.c \
        test_f.c \
        testsome_f.c \
        topo_test_f.c \
        type_commit_f.c \
        type_contiguous_f.c \
        type_create_darray_f.c \
        type_create_f90_complex_f.c \
        type_create_f90_integer_f.c \
        type_create_f90_real_f.c \
        type_create_hindexed_f.c \
        type_create_hvector_f.c \
        type_create_indexed_block_f.c \
        type_create_hindexed_block_f.c \
        type_create_keyval_f.c \
        type_create_resized_f.c \
        type_create_struct_f.c \
        type_create_subarray_f.c \
        type_delete_attr_f.c \
        type_dup_f.c \
        type_free_f.c \
        type_free_keyval_f.c \
        type_get_attr_f.c \
        type_get_contents_f.c \
        type_get_elwelope_f.c \
        type_get_extent_f.c \
        type_get_extent_x_f.c \
        type_get_name_f.c \
        type_get_true_extent_f.c \
        type_get_true_extent_x_f.c \
        type_indexed_f.c \
        type_match_size_f.c \
        type_set_attr_f.c \
        type_set_name_f.c \
        type_size_f.c \
        type_size_x_f.c \
        type_vector_f.c \
        unpack_external_f.c \
        unpack_f.c \
        unpublish_name_f.c \
        waitall_f.c \
        waitany_f.c \
        wait_f.c \
        waitsome_f.c \
        wtick_f.c \
        wtime_f.c \
        aclwmulate_f.c \
        raclwmulate_f.c \
        get_f.c \
        rget_f.c \
        get_aclwmulate_f.c \
        rget_aclwmulate_f.c \
        put_f.c \
        rput_f.c \
        compare_and_swap_f.c \
        fetch_and_op_f.c \
        win_allocate_f.c \
        win_allocate_shared_f.c \
        win_attach_f.c \
        win_call_errhandler_f.c \
        win_complete_f.c \
        win_create_dynamic_f.c \
        win_create_errhandler_f.c \
        win_create_f.c \
        win_create_keyval_f.c \
        win_delete_attr_f.c \
        win_detach_f.c \
        win_fence_f.c \
        win_free_f.c \
        win_free_keyval_f.c \
        win_get_attr_f.c \
        win_get_errhandler_f.c \
        win_get_group_f.c \
        win_get_info_f.c \
        win_get_name_f.c \
        win_lock_f.c \
        win_lock_all_f.c \
        win_post_f.c \
        win_set_attr_f.c \
        win_set_errhandler_f.c \
        win_set_info_f.c \
        win_set_name_f.c \
        win_shared_query_f.c \
        win_start_f.c \
        win_sync_f.c \
        win_test_f.c \
        win_unlock_f.c \
        win_unlock_all_f.c \
        win_wait_f.c \
        win_flush_f.c \
        win_flush_all_f.c \
        win_flush_local_f.c \
        win_flush_local_all_f.c

# include all of the removed MPI functions in library
# for v4.0.x regardless of state of OMPI_ENABLE_MPI1_COMPAT
lib@OMPI_LIBMPI_NAME@_mpifh_la_SOURCES += \
        address_f.c \
        errhandler_create_f.c \
        errhandler_get_f.c \
        errhandler_set_f.c \
        type_extent_f.c \
        type_hindexed_f.c \
        type_hvector_f.c \
        type_lb_f.c \
        type_struct_f.c \
        type_ub_f.c
endif

#
# Conditionally install the header files
#

if WANT_INSTALL_HEADERS
ompidir = $(ompiincludedir)/$(subdir)
ompi_HEADERS = $(headers)
endif
