# -*- makefile -*-
#
# Copyright (c) 2004-2005 The Trustees of Indiana University and Indiana
#                         University Research and Technology
#                         Corporation.  All rights reserved.
# Copyright (c) 2004-2013 The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.
# Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
#                         University of Stuttgart.  All rights reserved.
# Copyright (c) 2004-2005 The Regents of the University of California.
#                         All rights reserved.
# Copyright (c) 2009-2015 Cisco Systems, Inc.  All rights reserved.
# Copyright (c) 2011-2013 Inria.  All rights reserved.
# Copyright (c) 2011-2013 Universite Bordeaux 1
# Copyright (c) 2013-2014 Los Alamos National Security, LLC. All rights
#                         reserved.
# Copyright (c) 2015-2018 Research Organization for Information Science
#                         and Technology (RIST). All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

include $(top_srcdir)/Makefile.ompi-rules

CLEANFILES=
libmpi_mpifh_pmpi_la_LIBADD =

AM_CPPFLAGS = -DOMPI_BUILD_MPI_PROFILING=1 -DOMPI_COMPILING_FORTRAN_WRAPPERS=1
#
# This build needs to go through only if profiling is required.
# Further, this build HAS to go through if profiling is required.
#

noinst_LTLIBRARIES =
if BUILD_PMPI_FORTRAN_MPIFH_BINDINGS_LAYER
noinst_LTLIBRARIES += libmpi_mpifh_pmpi.la
endif

linked_files = \
        pabort_f.c \
        padd_error_class_f.c \
        padd_error_code_f.c \
        padd_error_string_f.c \
        paint_add_f.c \
        paint_diff_f.c \
        pallgather_f.c \
        pallgatherv_f.c \
        palloc_mem_f.c \
        pallreduce_f.c \
        palltoall_f.c \
        palltoallv_f.c \
        palltoallw_f.c \
        pattr_delete_f.c \
        pattr_get_f.c \
        pattr_put_f.c \
        pbarrier_f.c \
        pbcast_f.c \
        pbsend_f.c \
        pbsend_init_f.c \
        pbuffer_attach_f.c \
        pbuffer_detach_f.c \
        pcancel_f.c \
        pcart_coords_f.c \
        pcart_create_f.c \
        pcartdim_get_f.c \
        pcart_get_f.c \
        pcart_map_f.c \
        pcart_rank_f.c \
        pcart_shift_f.c \
        pcart_sub_f.c \
        pclose_port_f.c \
        pcomm_accept_f.c \
        pcomm_call_errhandler_f.c \
        pcomm_compare_f.c \
        pcomm_connect_f.c \
        pcomm_create_errhandler_f.c \
        pcomm_create_f.c \
        pcomm_create_group_f.c \
        pcomm_create_keyval_f.c \
        pcomm_delete_attr_f.c \
        pcomm_disconnect_f.c \
        pcomm_dup_f.c \
        pcomm_dup_with_info_f.c \
        pcomm_idup_f.c \
        pcomm_free_f.c \
        pcomm_free_keyval_f.c \
        pcomm_get_attr_f.c \
        pcomm_get_errhandler_f.c \
        pcomm_get_info_f.c \
        pcomm_get_name_f.c \
        pcomm_get_parent_f.c \
        pcomm_group_f.c \
        pcomm_join_f.c \
        pcomm_rank_f.c \
        pcomm_remote_group_f.c \
        pcomm_remote_size_f.c \
        pcomm_set_attr_f.c \
        pcomm_set_errhandler_f.c \
        pcomm_set_info_f.c \
        pcomm_set_name_f.c \
        pcomm_size_f.c \
        pcomm_spawn_f.c \
        pcomm_spawn_multiple_f.c \
        pcomm_split_f.c \
        pcomm_split_type_f.c \
        pcomm_test_inter_f.c \
        pdims_create_f.c \
        pdist_graph_create_adjacent_f.c \
        pdist_graph_create_f.c \
        pdist_graph_neighbors_f.c \
        pdist_graph_neighbors_count_f.c \
        perrhandler_free_f.c \
        perror_class_f.c \
        perror_string_f.c \
        pexscan_f.c \
        pf_sync_reg_f.c \
        pfile_call_errhandler_f.c \
        pfile_close_f.c \
        pfile_create_errhandler_f.c \
        pfile_delete_f.c \
        pfile_get_amode_f.c \
        pfile_get_atomicity_f.c \
        pfile_get_byte_offset_f.c \
        pfile_get_errhandler_f.c \
        pfile_get_group_f.c \
        pfile_get_info_f.c \
        pfile_get_position_f.c \
        pfile_get_position_shared_f.c \
        pfile_get_size_f.c \
        pfile_get_type_extent_f.c \
        pfile_get_view_f.c \
        pfile_iread_at_f.c \
        pfile_iread_f.c \
        pfile_iread_at_all_f.c \
        pfile_iread_all_f.c \
        pfile_iread_shared_f.c \
        pfile_iwrite_at_f.c \
        pfile_iwrite_f.c \
        pfile_iwrite_at_all_f.c \
        pfile_iwrite_all_f.c \
        pfile_iwrite_shared_f.c \
        pfile_open_f.c \
        pfile_preallocate_f.c \
        pfile_read_all_begin_f.c \
        pfile_read_all_end_f.c \
        pfile_read_all_f.c \
        pfile_read_at_all_begin_f.c \
        pfile_read_at_all_end_f.c \
        pfile_read_at_all_f.c \
        pfile_read_at_f.c \
        pfile_read_f.c \
        pfile_read_ordered_begin_f.c \
        pfile_read_ordered_end_f.c \
        pfile_read_ordered_f.c \
        pfile_read_shared_f.c \
        pfile_seek_f.c \
        pfile_seek_shared_f.c \
        pfile_set_atomicity_f.c \
        pfile_set_errhandler_f.c \
        pfile_set_info_f.c \
        pfile_set_size_f.c \
        pfile_set_view_f.c \
        pfile_sync_f.c \
        pfile_write_all_begin_f.c \
        pfile_write_all_end_f.c \
        pfile_write_all_f.c \
        pfile_write_at_all_begin_f.c \
        pfile_write_at_all_end_f.c \
        pfile_write_at_all_f.c \
        pfile_write_at_f.c \
        pfile_write_f.c \
        pfile_write_ordered_begin_f.c \
        pfile_write_ordered_end_f.c \
        pfile_write_ordered_f.c \
        pfile_write_shared_f.c \
        pfinalized_f.c \
        pfinalize_f.c \
        pfree_mem_f.c \
        pgather_f.c \
        pgatherv_f.c \
        pget_address_f.c \
        pget_count_f.c \
        pget_elements_f.c \
        pget_elements_x_f.c \
        pget_library_version_f.c \
        pget_processor_name_f.c \
        pget_version_f.c \
        pgraph_create_f.c \
        pgraphdims_get_f.c \
        pgraph_get_f.c \
        pgraph_map_f.c \
        pgraph_neighbors_count_f.c \
        pgraph_neighbors_f.c \
        pgrequest_complete_f.c \
        pgrequest_start_f.c \
        pgroup_compare_f.c \
        pgroup_difference_f.c \
        pgroup_excl_f.c \
        pgroup_free_f.c \
        pgroup_incl_f.c \
        pgroup_intersection_f.c \
        pgroup_range_excl_f.c \
        pgroup_range_incl_f.c \
        pgroup_rank_f.c \
        pgroup_size_f.c \
        pgroup_translate_ranks_f.c \
        pgroup_union_f.c \
        piallgather_f.c \
        piallgatherv_f.c \
        piallreduce_f.c \
        pialltoall_f.c \
        pialltoallv_f.c \
        pialltoallw_f.c \
        pibarrier_f.c \
        pibcast_f.c \
        pibsend_f.c \
        piexscan_f.c \
        pigather_f.c \
        pigatherv_f.c \
        pimprobe_f.c \
        pimrecv_f.c \
        pineighbor_allgather_f.c \
        pineighbor_allgatherv_f.c \
        pineighbor_alltoall_f.c \
        pineighbor_alltoallv_f.c \
        pineighbor_alltoallw_f.c \
        pinfo_create_f.c \
        pinfo_delete_f.c \
        pinfo_dup_f.c \
        pinfo_free_f.c \
        pinfo_get_f.c \
        pinfo_get_nkeys_f.c \
        pinfo_get_nthkey_f.c \
        pinfo_get_valuelen_f.c \
        pinfo_set_f.c \
        pinit_f.c \
        pinitialized_f.c \
        pinit_thread_f.c \
        pintercomm_create_f.c \
        pintercomm_merge_f.c \
        piprobe_f.c \
        pirecv_f.c \
        pireduce_f.c \
        pireduce_scatter_f.c \
        pireduce_scatter_block_f.c \
        pirsend_f.c \
        piscan_f.c \
        piscatter_f.c \
        piscatterv_f.c \
        pisend_f.c \
        pissend_f.c \
        pis_thread_main_f.c \
        pkeyval_create_f.c \
        pkeyval_free_f.c \
        plookup_name_f.c \
        pmprobe_f.c \
        pmrecv_f.c \
        pneighbor_allgather_f.c \
        pneighbor_allgatherv_f.c \
        pneighbor_alltoall_f.c \
        pneighbor_alltoallv_f.c \
        pneighbor_alltoallw_f.c \
        pop_commutative_f.c \
        pop_create_f.c \
        popen_port_f.c \
        pop_free_f.c \
        ppack_external_f.c \
        ppack_external_size_f.c \
        ppack_f.c \
        ppack_size_f.c \
        ppcontrol_f.c \
        pprobe_f.c \
        ppublish_name_f.c \
        pquery_thread_f.c \
        precv_f.c \
        precv_init_f.c \
        preduce_f.c \
        preduce_local_f.c \
        preduce_scatter_f.c \
        preduce_scatter_block_f.c \
        prequest_free_f.c \
        prequest_get_status_f.c \
        prsend_f.c \
        prsend_init_f.c \
        pscan_f.c \
        pscatter_f.c \
        pscatterv_f.c \
        psend_f.c \
        psend_init_f.c \
        psendrecv_f.c \
        psendrecv_replace_f.c \
        pssend_f.c \
        pssend_init_f.c \
        pstartall_f.c \
        pstart_f.c \
        pstatus_set_cancelled_f.c \
        pstatus_set_elements_f.c \
        pstatus_set_elements_x_f.c \
        ptestall_f.c \
        ptestany_f.c \
        ptest_cancelled_f.c \
        ptest_f.c \
        ptestsome_f.c \
        ptopo_test_f.c \
        ptype_commit_f.c \
        ptype_contiguous_f.c \
        ptype_create_darray_f.c \
        ptype_create_f90_complex_f.c \
        ptype_create_f90_integer_f.c \
        ptype_create_f90_real_f.c \
        ptype_create_hindexed_f.c \
        ptype_create_hvector_f.c \
        ptype_create_indexed_block_f.c \
        ptype_create_hindexed_block_f.c \
        ptype_create_keyval_f.c \
        ptype_create_resized_f.c \
        ptype_create_struct_f.c \
        ptype_create_subarray_f.c \
        ptype_delete_attr_f.c \
        ptype_dup_f.c \
        ptype_free_f.c \
        ptype_free_keyval_f.c \
        ptype_get_attr_f.c \
        ptype_get_contents_f.c \
        ptype_get_elwelope_f.c \
        ptype_get_extent_f.c \
        ptype_get_extent_x_f.c \
        ptype_get_name_f.c \
        ptype_get_true_extent_f.c \
        ptype_get_true_extent_x_f.c \
        ptype_indexed_f.c \
        ptype_match_size_f.c \
        ptype_set_attr_f.c \
        ptype_set_name_f.c \
        ptype_size_f.c \
        ptype_size_x_f.c \
        ptype_vector_f.c \
        punpack_external_f.c \
        punpack_f.c \
        punpublish_name_f.c \
        pwaitall_f.c \
        pwaitany_f.c \
        pwait_f.c \
        pwaitsome_f.c \
	pwtick_f.c \
	pwtime_f.c \
        paclwmulate_f.c \
        praclwmulate_f.c \
        pregister_datarep_f.c \
        pget_f.c \
        prget_f.c \
        pget_aclwmulate_f.c \
        prget_aclwmulate_f.c \
        pput_f.c \
        prput_f.c \
        pcompare_and_swap_f.c \
        pfetch_and_op_f.c \
        pwin_allocate_f.c \
        pwin_allocate_shared_f.c \
        pwin_attach_f.c \
        pwin_call_errhandler_f.c \
        pwin_complete_f.c \
        pwin_create_dynamic_f.c \
        pwin_create_errhandler_f.c \
        pwin_create_f.c \
        pwin_create_keyval_f.c \
        pwin_delete_attr_f.c \
        pwin_detach_f.c \
        pwin_fence_f.c \
        pwin_free_f.c \
        pwin_free_keyval_f.c \
        pwin_get_attr_f.c \
        pwin_get_errhandler_f.c \
        pwin_get_group_f.c \
        pwin_get_info_f.c \
        pwin_get_name_f.c \
        pwin_lock_f.c \
        pwin_lock_all_f.c \
        pwin_post_f.c \
        pwin_set_attr_f.c \
        pwin_set_errhandler_f.c \
        pwin_set_info_f.c \
        pwin_set_name_f.c \
        pwin_shared_query_f.c \
        pwin_start_f.c \
        pwin_sync_f.c \
        pwin_test_f.c \
        pwin_unlock_f.c \
        pwin_unlock_all_f.c \
        pwin_wait_f.c \
        pwin_flush_f.c \
        pwin_flush_all_f.c \
        pwin_flush_local_f.c \
        pwin_flush_local_all_f.c

# include all of the removed MPI functions in library
# for v4.0.x regardless of state of OMPI_ENABLE_MPI1_COMPAT
linked_files += \
        paddress_f.c \
        perrhandler_create_f.c \
        perrhandler_get_f.c \
        perrhandler_set_f.c \
        ptype_extent_f.c \
        ptype_hindexed_f.c \
        ptype_hvector_f.c \
        ptype_lb_f.c \
        ptype_struct_f.c \
        ptype_ub_f.c

#
# Sym link in the sources from the real MPI directory
#
$(linked_files):
	$(OMPI_V_LN_S) if test ! -r $@ ; then \
	    pname=`echo $@ | cut -b '2-'` ; \
	    $(LN_S) $(top_srcdir)/ompi/mpi/fortran/mpif-h/$$pname $@ ; \
	fi

# psizeof_f.f90 is generated based on some results from configure tests.
CLEANFILES += psizeof_f.f90

# Build the MPI_SIZEOF code in a separate colwenience library (see
# lengthy comment in ompi/mpi/fortran/mpif-h/Makefile.am for an
# explanation why).
if BUILD_FORTRAN_SIZEOF
noinst_LTLIBRARIES += libmpi_mpifh_psizeof.la
# Do not dist this file; it is generated
nodist_libmpi_mpifh_psizeof_la_SOURCES = psizeof_f.f90
libmpi_mpifh_pmpi_la_LIBADD += libmpi_mpifh_psizeof.la
endif

sizeof_pl=$(top_srcdir)/ompi/mpi/fortran/base/gen-mpi-sizeof.pl

psizeof_f.f90: $(top_builddir)/config.status
psizeof_f.f90: $(sizeof_pl)
psizeof_f.f90:
	$(OMPI_V_GEN) $(sizeof_pl) \
	    --impl=$@ --ierror=mandatory --pmpi \
	    --maxrank=$(OMPI_FORTRAN_MAX_ARRAY_RANK) \
	    --generate=$(OMPI_FORTRAN_BUILD_SIZEOF) \
	    --real16=$(OMPI_HAVE_FORTRAN_REAL16) \
	    --complex32=$(OMPI_HAVE_FORTRAN_COMPLEX32)

#
# The library itself
#
nodist_libmpi_mpifh_pmpi_la_SOURCES = \
        $(linked_files)

# Conditionally install the header files

if WANT_INSTALL_HEADERS
ompidir = $(ompiincludedir)/$(subdir)
endif

# These files were created by targets above

MAINTAINERCLEANFILES = $(nodist_libmpi_mpifh_pmpi_la_SOURCES)

# Don't want these targets in here

tags-relwrsive:
tags:
TAGS:
GTAGS:
ID:
