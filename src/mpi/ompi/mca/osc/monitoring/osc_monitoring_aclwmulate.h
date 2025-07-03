/*
 * Copyright (c) 2016-2018 Inria.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#ifndef MCA_OSC_MONITORING_ACLWMULATE_H
#define MCA_OSC_MONITORING_ACLWMULATE_H

#include <ompi/datatype/ompi_datatype.h>
#include <ompi/op/op.h>
#include <ompi/win/win.h>

#define OSC_MONITORING_GENERATE_TEMPLATE_ACLWMULATE(template)           \
                                                                        \
    static int ompi_osc_monitoring_## template ##_compare_and_swap (const void *origin_addr, \
                                                                    const void *compare_addr, \
                                                                    void *result_addr, \
                                                                    ompi_datatype_t *dt, \
                                                                    int target_rank, \
                                                                    ptrdiff_t target_disp, \
                                                                    ompi_win_t *win) \
    {                                                                   \
        int world_rank;                                                 \
        /**                                                             \
         * If this fails the destination is not part of my MPI_COM_WORLD \
         * Lookup its name in the rank hastable to get its MPI_COMM_WORLD rank \
         */                                                             \
        if(OPAL_SUCCESS == mca_common_monitoring_get_world_rank(target_rank, win->w_group, &world_rank)) { \
            size_t type_size;                                           \
            ompi_datatype_type_size(dt, &type_size);                    \
            mca_common_monitoring_record_osc(world_rank, type_size, SEND); \
            mca_common_monitoring_record_osc(world_rank, type_size, RECV); \
            OPAL_MONITORING_PRINT_INFO("MPI_Compare_and_swap to %d intercepted", world_rank); \
        }                                                               \
        return OMPI_OSC_MONITORING_MODULE_VARIABLE(template).osc_compare_and_swap(origin_addr, compare_addr, result_addr, dt, target_rank, target_disp, win); \
    }                                                                   \
                                                                        \
    static int ompi_osc_monitoring_## template ##_get_aclwmulate (const void *origin_addr, \
                                                                  int origin_count, \
                                                                  ompi_datatype_t*origin_datatype, \
                                                                  void *result_addr, \
                                                                  int result_count, \
                                                                  ompi_datatype_t*result_datatype, \
                                                                  int target_rank, \
                                                                  MPI_Aint target_disp, \
                                                                  int target_count, \
                                                                  ompi_datatype_t*target_datatype, \
                                                                  ompi_op_t *op, ompi_win_t*win) \
    {                                                                   \
        int world_rank;                                                 \
        /**                                                             \
         * If this fails the destination is not part of my MPI_COM_WORLD \
         * Lookup its name in the rank hastable to get its MPI_COMM_WORLD rank \
         */                                                             \
        if(OPAL_SUCCESS == mca_common_monitoring_get_world_rank(target_rank, win->w_group, &world_rank)) { \
            size_t type_size, data_size;                                \
            ompi_datatype_type_size(origin_datatype, &type_size);       \
            data_size = origin_count*type_size;                         \
            mca_common_monitoring_record_osc(world_rank, data_size, SEND); \
            ompi_datatype_type_size(result_datatype, &type_size);       \
            data_size = result_count*type_size;                         \
            mca_common_monitoring_record_osc(world_rank, data_size, RECV); \
            OPAL_MONITORING_PRINT_INFO("MPI_Get_aclwmulate to %d intercepted", world_rank); \
        }                                                               \
        return OMPI_OSC_MONITORING_MODULE_VARIABLE(template).osc_get_aclwmulate(origin_addr, origin_count, origin_datatype, result_addr, result_count, result_datatype, target_rank, target_disp, target_count, target_datatype, op, win); \
    }                                                                   \
                                                                        \
    static int ompi_osc_monitoring_## template ##_rget_aclwmulate (const void *origin_addr, \
                                                                   int origin_count, \
                                                                   ompi_datatype_t *origin_datatype, \
                                                                   void *result_addr, \
                                                                   int result_count, \
                                                                   ompi_datatype_t *result_datatype, \
                                                                   int target_rank, \
                                                                   MPI_Aint target_disp, \
                                                                   int target_count, \
                                                                   ompi_datatype_t*target_datatype, \
                                                                   ompi_op_t *op, \
                                                                   ompi_win_t *win, \
                                                                   ompi_request_t **request) \
    {                                                                   \
        int world_rank;                                                 \
        /**                                                             \
         * If this fails the destination is not part of my MPI_COM_WORLD \
         * Lookup its name in the rank hastable to get its MPI_COMM_WORLD rank \
         */                                                             \
        if(OPAL_SUCCESS == mca_common_monitoring_get_world_rank(target_rank, win->w_group, &world_rank)) { \
            size_t type_size, data_size;                                \
            ompi_datatype_type_size(origin_datatype, &type_size);       \
            data_size = origin_count*type_size;                         \
            mca_common_monitoring_record_osc(world_rank, data_size, SEND); \
            ompi_datatype_type_size(result_datatype, &type_size);       \
            data_size = result_count*type_size;                         \
            mca_common_monitoring_record_osc(world_rank, data_size, RECV); \
            OPAL_MONITORING_PRINT_INFO("MPI_Rget_aclwmulate to %d intercepted", world_rank); \
        }                                                               \
        return OMPI_OSC_MONITORING_MODULE_VARIABLE(template).osc_rget_aclwmulate(origin_addr, origin_count, origin_datatype, result_addr, result_count, result_datatype, target_rank, target_disp, target_count, target_datatype, op, win, request); \
    }                                                                   \
                                                                        \
    static int ompi_osc_monitoring_## template ##_raclwmulate (const void *origin_addr, \
                                                               int origin_count, \
                                                               ompi_datatype_t *origin_datatype, \
                                                               int target_rank, \
                                                               ptrdiff_t target_disp, \
                                                               int target_count, \
                                                               ompi_datatype_t *target_datatype, \
                                                               ompi_op_t *op, ompi_win_t *win, \
                                                               ompi_request_t **request) \
    {                                                                   \
        int world_rank;                                                 \
        /**                                                             \
         * If this fails the destination is not part of my MPI_COM_WORLD \
         * Lookup its name in the rank hastable to get its MPI_COMM_WORLD rank \
         */                                                             \
        if(OPAL_SUCCESS == mca_common_monitoring_get_world_rank(target_rank, win->w_group, &world_rank)) { \
            size_t type_size, data_size;                                \
            ompi_datatype_type_size(origin_datatype, &type_size);       \
            data_size = origin_count*type_size;                         \
            mca_common_monitoring_record_osc(world_rank, data_size, SEND); \
            OPAL_MONITORING_PRINT_INFO("MPI_Raclwmulate to %d intercepted", world_rank); \
        }                                                               \
        return OMPI_OSC_MONITORING_MODULE_VARIABLE(template).osc_raclwmulate(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count, target_datatype, op, win, request); \
    }                                                                   \
                                                                        \
    static int ompi_osc_monitoring_## template ##_aclwmulate (const void *origin_addr, \
                                                              int origin_count, \
                                                              ompi_datatype_t *origin_datatype, \
                                                              int target_rank, \
                                                              ptrdiff_t target_disp, \
                                                              int target_count, \
                                                              ompi_datatype_t *target_datatype, \
                                                              ompi_op_t *op, ompi_win_t *win) \
    {                                                                   \
        int world_rank;                                                 \
        /**                                                             \
         * If this fails the destination is not part of my MPI_COM_WORLD \
         * Lookup its name in the rank hastable to get its MPI_COMM_WORLD rank \
         */                                                             \
        if(OPAL_SUCCESS == mca_common_monitoring_get_world_rank(target_rank, win->w_group, &world_rank)) { \
            size_t type_size, data_size;                                \
            ompi_datatype_type_size(origin_datatype, &type_size);       \
            data_size = origin_count*type_size;                         \
            mca_common_monitoring_record_osc(world_rank, data_size, SEND); \
            OPAL_MONITORING_PRINT_INFO("MPI_Aclwmulate to %d intercepted", world_rank); \
        }                                                               \
        return OMPI_OSC_MONITORING_MODULE_VARIABLE(template).osc_aclwmulate(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count, target_datatype, op, win); \
    }                                                                   \
                                                                        \
    static int ompi_osc_monitoring_## template ##_fetch_and_op (const void *origin_addr, \
                                                                void *result_addr, \
                                                                ompi_datatype_t *dt, \
                                                                int target_rank, \
                                                                ptrdiff_t target_disp, \
                                                                ompi_op_t *op, ompi_win_t *win) \
    {                                                                   \
        int world_rank;                                                 \
        /**                                                             \
         * If this fails the destination is not part of my MPI_COM_WORLD \
         * Lookup its name in the rank hastable to get its MPI_COMM_WORLD rank \
         */                                                             \
        if(OPAL_SUCCESS == mca_common_monitoring_get_world_rank(target_rank, win->w_group, &world_rank)) { \
            size_t type_size;                                           \
            ompi_datatype_type_size(dt, &type_size);                    \
            mca_common_monitoring_record_osc(world_rank, type_size, SEND); \
            mca_common_monitoring_record_osc(world_rank, type_size, RECV); \
            OPAL_MONITORING_PRINT_INFO("MPI_Fetch_and_op to %d intercepted", world_rank); \
        }                                                               \
        return OMPI_OSC_MONITORING_MODULE_VARIABLE(template).osc_fetch_and_op(origin_addr, result_addr, dt, target_rank, target_disp, op, win); \
    }

#endif /* MCA_OSC_MONITORING_ACLWMULATE_H */
