/*
 * Copyright (c) 2014      Cisco Systems, Inc.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "opal_config.h"

#include <stdlib.h>
#include <sys/time.h>

#include "opal/constants.h"
#include "opal/class/opal_list.h"
#include "opal/class/opal_pointer_array.h"
#include "opal/util/bipartite_graph.h"
#include "opal/util/bipartite_graph_internal.h"

#  define test_out(...) fprintf(stderr, __VA_ARGS__)
#  define check(a)                                                           \
    do {                                                                     \
        if (!(a)) {                                                          \
            test_out("%s:%d: check failed, '%s'\n", __func__, __LINE__, #a); \
            return 1;                                              \
        }                                                                    \
    } while (0)
#  define check_str_eq(a,b)                                     \
    do {                                                        \
        const char *a_ = (a);                                   \
        const char *b_ = (b);                                   \
        if (0 != strcmp(a_,b_)) {                               \
            test_out("%s:%d: check failed, \"%s\" != \"%s\"\n", \
                     __func__, __LINE__, a_, b_);               \
            return 1;                                 \
        }                                                       \
    } while (0)
#  define check_int_eq(got, expected)                                   \
    do {                                                                \
        if ((got) != (expected)) {                                      \
            test_out("%s:%d: check failed, \"%s\" != \"%s\", got %d\n", \
                     __func__, __LINE__, #got, #expected, (got));       \
            return 1;                                         \
        }                                                               \
    } while (0)
/* just use check_int_eq for now, no public error code to string routine
 * exists (opal_err2str is static) */
#  define check_err_code(got, expected)                                 \
    check_int_eq(got, expected)
#  define check_msg(a, msg)                                \
    do {                                                   \
        if (!(a)) {                                        \
            test_out("%s:%d: check failed, \"%s\" (%s)\n", \
                     __func__, __LINE__, #a, (msg));       \
            return 1;                            \
        }                                                  \
    } while (0)

#define check_graph_is_consistent(g)                                         \
    do {                                                                     \
        check(opal_bp_graph_order(g) <= opal_pointer_array_get_size(&g->vertices)); \
        check(g->source_idx >= -1 || g->source_idx < opal_bp_graph_order(g));       \
        check(g->sink_idx >= -1 || g->sink_idx < opal_bp_graph_order(g));           \
    } while (0)

#define check_has_in_out_degree(g, u, expected_indegree, expected_outdegree)   \
    do {                                                                       \
        check_int_eq(opal_bp_graph_indegree(g, (u)), expected_indegree);   \
        check_int_eq(opal_bp_graph_outdegree(g, (u)), expected_outdegree); \
    } while (0)

/* Check the given path for sanity and that it does not have a cycle.  Uses
 * the "racing pointers" approach for cycle checking. */
#define check_path_cycle(n, source, sink, pred)    \
    do {                                           \
        int i_, j_;                                \
        check_int_eq(pred[source], -1);            \
        for (i_ = 0; i_ < n; ++i_) {               \
            check(pred[i_] >= -1);                 \
            check(pred[i_] < n);                   \
        }                                          \
        i_ = (sink);                               \
        j_ = pred[(sink)];                       \
        while (i_ != -1 && j_ != -1) {             \
            check_msg(i_ != j_, "CYCLE DETECTED"); \
            i_ = pred[i_];                         \
            j_ = pred[j_];                         \
            if (j_ != -1) {                        \
                j_ = pred[j_];                     \
            }                                      \
        }                                          \
    } while (0)

static int v_cleanup_count = 0;
static int e_cleanup_count = 0;

static void v_cleanup(void *v_data)
{
    ++v_cleanup_count;
}

static void e_cleanup(void *e_data)
{
    ++e_cleanup_count;
}

/* a utility function for comparing integer pairs, useful for sorting the edge
 * list returned by opal_bp_graph_solve_bipartite_assignment */
static int cmp_int_pair(const void *a, const void *b)
{
    int *ia = (int *)a;
    int *ib = (int *)b;

    if (ia[0] < ib[0]) {
        return -1;
    }
    else if (ia[0] > ib[0]) {
        return 1;
    }
    else { /* ia[0] == ib[0] */
        if (ia[1] < ib[1]) {
            return -1;
        }
        else if (ia[1] > ib[1]) {
            return 1;
        }
        else {
            return 0;
        }
    }
}

/* Simple time function so that we don't have to deal with the
   complexity of finding mpi.h to use MPI_Wtime */
static double gettime(void)
{
    double wtime;
    struct timeval tv;
    gettimeofday(&tv, NULL);
    wtime = tv.tv_sec;
    wtime += (double)tv.tv_usec / 1000000.0;

    return wtime;
}

static int test_graph_create(void *ctx)
{
    opal_bp_graph_t *g;
    int i;
    int err;
    int user_data;
    int index;

    /* TEST CASE: check zero-vertex case */
    g = NULL;
    err = opal_bp_graph_create(NULL, NULL, &g);
    check_err_code(err, OPAL_SUCCESS);
    check(g != NULL);
    check(opal_bp_graph_order(g) == 0);
    check_graph_is_consistent(g);
    err = opal_bp_graph_free(g);
    check_err_code(err, OPAL_SUCCESS);

    /* TEST CASE: check nonzero-vertex case with no cleanup routines */
    g = NULL;
    err = opal_bp_graph_create(NULL, NULL, &g);
    check_err_code(err, OPAL_SUCCESS);
    check(g != NULL);
    check_graph_is_consistent(g);
    for (i = 0; i < 4; ++i) {
        index = -1;
        err = opal_bp_graph_add_vertex(g, &user_data, &index);
        check_err_code(err, OPAL_SUCCESS);
        check(index == i);
    }
    check(opal_bp_graph_order(g) == 4);
    check_graph_is_consistent(g);
    err = opal_bp_graph_free(g);
    check_err_code(err, OPAL_SUCCESS);

    /* TEST CASE: make sure cleanup routines are ilwoked properly */
    g = NULL;
    v_cleanup_count = 0;
    e_cleanup_count = 0;
    err = opal_bp_graph_create(&v_cleanup, &e_cleanup, &g);
    check_err_code(err, OPAL_SUCCESS);
    check(g != NULL);
    check_graph_is_consistent(g);
    for (i = 0; i < 5; ++i) {
        err = opal_bp_graph_add_vertex(g, &user_data, &index);
        check_err_code(err, OPAL_SUCCESS);
        check(index == i);
    }
    check(opal_bp_graph_order(g) == 5);
    check_graph_is_consistent(g);
    err = opal_bp_graph_add_edge(g, /*u=*/0, /*v=*/3, /*cost=*/1,
                                     /*capacity=*/2, &user_data);
    check_graph_is_consistent(g);
    check(v_cleanup_count == 0);
    check(e_cleanup_count == 0);
    err = opal_bp_graph_free(g);
    check_err_code(err, OPAL_SUCCESS);
    check(v_cleanup_count == 5);
    check(e_cleanup_count == 1);

    return 0;
}

static int test_graph_clone(void *ctx)
{
    opal_bp_graph_t *g, *gx;
    int i;
    int err;
    int user_data;
    int index;

    /* TEST CASE: make sure that simple cloning works fine */
    g = NULL;
    v_cleanup_count = 0;
    e_cleanup_count = 0;
    err = opal_bp_graph_create(&v_cleanup, &e_cleanup, &g);
    check_err_code(err, OPAL_SUCCESS);
    check(g != NULL);
    check_graph_is_consistent(g);

    /* add 5 edges */
    for (i = 0; i < 5; ++i) {
        err = opal_bp_graph_add_vertex(g, &user_data, &index);
        check_err_code(err, OPAL_SUCCESS);
    }
    check(opal_bp_graph_order(g) == 5);
    check_graph_is_consistent(g);

    /* and two edges */
    err = opal_bp_graph_add_edge(g, /*u=*/0, /*v=*/3, /*cost=*/1,
                                     /*capacity=*/2, &user_data);
    check_err_code(err, OPAL_SUCCESS);
    check_graph_is_consistent(g);
    err = opal_bp_graph_add_edge(g, /*u=*/3, /*v=*/1, /*cost=*/2,
                                     /*capacity=*/100, &user_data);
    check_err_code(err, OPAL_SUCCESS);
    check_graph_is_consistent(g);

    /* now clone it and ensure that we get the same kind of graph */
    gx = NULL;
    err = opal_bp_graph_clone(g, /*copy_user_data=*/false, &gx);
    check_err_code(err, OPAL_SUCCESS);
    check(gx != NULL);

    /* double check that cleanups still happen as expected after cloning */
    err = opal_bp_graph_free(gx);
    check_err_code(err, OPAL_SUCCESS);
    check(v_cleanup_count == 0);
    check(e_cleanup_count == 0);
    err = opal_bp_graph_free(g);
    check_err_code(err, OPAL_SUCCESS);
    check(v_cleanup_count == 5);
    check(e_cleanup_count == 2);

    return 0;
}

static int test_graph_accessors(void *ctx)
{
    opal_bp_graph_t *g;
    int i;
    int err;

    /* TEST CASE: check _indegree/_outdegree/_order work correctly */
    err = opal_bp_graph_create(NULL, NULL, &g);
    check_err_code(err, OPAL_SUCCESS);
    check(g != NULL);

    for (i = 0; i < 4; ++i) {
        err = opal_bp_graph_add_vertex(g, NULL, NULL);
        check_err_code(err, OPAL_SUCCESS);

        check(opal_bp_graph_indegree(g, i) == 0);
        check(opal_bp_graph_outdegree(g, i) == 0);
    }

    check(opal_bp_graph_order(g) == 4);

    err = opal_bp_graph_add_edge(g, /*u=*/0, /*v=*/2, /*cost=*/2,
                                     /*capacity=*/1, NULL);
    err = opal_bp_graph_add_edge(g, /*u=*/0, /*v=*/1, /*cost=*/2,
                                     /*capacity=*/1, NULL);

    check(opal_bp_graph_indegree(g,  0) == 0);
    check(opal_bp_graph_outdegree(g, 0) == 2);
    check(opal_bp_graph_indegree(g,  1) == 1);
    check(opal_bp_graph_outdegree(g, 1) == 0);
    check(opal_bp_graph_indegree(g,  2) == 1);
    check(opal_bp_graph_outdegree(g, 2) == 0);
    check(opal_bp_graph_indegree(g,  3) == 0);
    check(opal_bp_graph_outdegree(g, 3) == 0);

    err = opal_bp_graph_free(g);
    check_err_code(err, OPAL_SUCCESS);

    return 0;
}

static int test_graph_assignment_solver(void *ctx)
{
    opal_bp_graph_t *g;
    int i;
    int err;
    int nme;
    int *me;
    int iter;
    double start, end;

    /* TEST CASE: check that simple cases are solved correctly
     *
     * 0 --> 2
     * 1 --> 3
     */
    err = opal_bp_graph_create(NULL, NULL, &g);
    check_err_code(err, OPAL_SUCCESS);
    check(g != NULL);

    for (i = 0; i < 4; ++i) {
        err = opal_bp_graph_add_vertex(g, NULL, NULL);
        check_err_code(err, OPAL_SUCCESS);
    }

    err = opal_bp_graph_add_edge(g, /*u=*/0, /*v=*/2, /*cost=*/10,
                                     /*capacity=*/1, NULL);
    err = opal_bp_graph_add_edge(g, /*u=*/1, /*v=*/3, /*cost=*/2,
                                     /*capacity=*/1, NULL);

    me = NULL;
    err = opal_bp_graph_solve_bipartite_assignment(g,
                                                    &nme,
                                                    &me);
    check_err_code(err, OPAL_SUCCESS);
    check_int_eq(nme, 2);
    check(me != NULL);
    qsort(me, nme, 2*sizeof(int), &cmp_int_pair);
    check(me[0] == 0 && me[1] == 2);
    check(me[2] == 1 && me[3] == 3);

    err = opal_bp_graph_free(g);
    check_err_code(err, OPAL_SUCCESS);


    /* TEST CASE: left side has more vertices than the right side
     *
     * 0 --> 3
     * 1 --> 4
     * 2 --> 4
     */
    err = opal_bp_graph_create(NULL, NULL, &g);
    check_err_code(err, OPAL_SUCCESS);
    check(g != NULL);

    for (i = 0; i < 5; ++i) {
        err = opal_bp_graph_add_vertex(g, NULL, NULL);
        check_err_code(err, OPAL_SUCCESS);
    }

    err = opal_bp_graph_add_edge(g, /*u=*/0, /*v=*/3, /*cost=*/10,
                                     /*capacity=*/1, NULL);
    check_err_code(err, OPAL_SUCCESS);
    err = opal_bp_graph_add_edge(g, /*u=*/1, /*v=*/4, /*cost=*/2,
                                     /*capacity=*/1, NULL);
    check_err_code(err, OPAL_SUCCESS);
    err = opal_bp_graph_add_edge(g, /*u=*/2, /*v=*/4, /*cost=*/1,
                                     /*capacity=*/1, NULL);
    check_err_code(err, OPAL_SUCCESS);

    me = NULL;
    err = opal_bp_graph_solve_bipartite_assignment(g,
                                                    &nme,
                                                    &me);
    check_err_code(err, OPAL_SUCCESS);
    check_int_eq(nme, 2);
    check(me != NULL);
    qsort(me, nme, 2*sizeof(int), &cmp_int_pair);
    check(me[0] == 0 && me[1] == 3);
    check(me[2] == 2 && me[3] == 4);
    free(me);

    err = opal_bp_graph_free(g);
    check_err_code(err, OPAL_SUCCESS);


    /* test Christian's case:
     * 0 --> 2
     * 0 --> 3
     * 1 --> 3
     *
     * make sure that 0-->2 & 1-->3 get chosen.
     */
    err = opal_bp_graph_create(NULL, NULL, &g);
    check_err_code(err, OPAL_SUCCESS);
    check(g != NULL);

    for (i = 0; i < 4; ++i) {
        err = opal_bp_graph_add_vertex(g, NULL, NULL);
        check_err_code(err, OPAL_SUCCESS);
    }

    err = opal_bp_graph_add_edge(g, /*u=*/0, /*v=*/2, /*cost=*/10,
                                     /*capacity=*/1, NULL);
    check_err_code(err, OPAL_SUCCESS);
    err = opal_bp_graph_add_edge(g, /*u=*/0, /*v=*/3, /*cost=*/1,
                                     /*capacity=*/1, NULL);
    check_err_code(err, OPAL_SUCCESS);
    err = opal_bp_graph_add_edge(g, /*u=*/1, /*v=*/3, /*cost=*/5,
                                     /*capacity=*/1, NULL);
    check_err_code(err, OPAL_SUCCESS);

    me = NULL;
    err = opal_bp_graph_solve_bipartite_assignment(g,
                                                    &nme,
                                                    &me);
    check_err_code(err, OPAL_SUCCESS);
    check_int_eq(nme, 2);
    check(me != NULL);
    qsort(me, nme, 2*sizeof(int), &cmp_int_pair);
    check(me[0] == 0 && me[1] == 2);
    check(me[2] == 1 && me[3] == 3);
    free(me);

    err = opal_bp_graph_free(g);
    check_err_code(err, OPAL_SUCCESS);

    /* Also need to do this version of it to be safe:
     * 0 --> 2
     * 1 --> 2
     * 1 --> 3
     *
     * Should choose 0-->2 & 1-->3 here too.
     */
    err = opal_bp_graph_create(NULL, NULL, &g);
    check_err_code(err, OPAL_SUCCESS);
    check(g != NULL);

    for (i = 0; i < 4; ++i) {
        err = opal_bp_graph_add_vertex(g, NULL, NULL);
        check_err_code(err, OPAL_SUCCESS);
    }

    err = opal_bp_graph_add_edge(g, /*u=*/0, /*v=*/2, /*cost=*/10,
                                     /*capacity=*/1, NULL);
    check_err_code(err, OPAL_SUCCESS);
    err = opal_bp_graph_add_edge(g, /*u=*/1, /*v=*/2, /*cost=*/1,
                                     /*capacity=*/1, NULL);
    check_err_code(err, OPAL_SUCCESS);
    err = opal_bp_graph_add_edge(g, /*u=*/1, /*v=*/3, /*cost=*/5,
                                     /*capacity=*/1, NULL);
    check_err_code(err, OPAL_SUCCESS);

    me = NULL;
    err = opal_bp_graph_solve_bipartite_assignment(g,
                                                    &nme,
                                                    &me);
    check_err_code(err, OPAL_SUCCESS);
    check_int_eq(nme, 2);
    check(me != NULL);
    qsort(me, nme, 2*sizeof(int), &cmp_int_pair);
    check(me[0] == 0 && me[1] == 2);
    check(me[2] == 1 && me[3] == 3);
    free(me);

    err = opal_bp_graph_free(g);
    check_err_code(err, OPAL_SUCCESS);

    /* TEST CASE: test Christian's case with negative weights:
     * 0 --> 2
     * 0 --> 3
     * 1 --> 3
     *
     * make sure that 0-->2 & 1-->3 get chosen.
     */
    err = opal_bp_graph_create(NULL, NULL, &g);
    check_err_code(err, OPAL_SUCCESS);
    check(g != NULL);

    for (i = 0; i < 4; ++i) {
        err = opal_bp_graph_add_vertex(g, NULL, NULL);
        check_err_code(err, OPAL_SUCCESS);
    }

    err = opal_bp_graph_add_edge(g, /*u=*/0, /*v=*/2, /*cost=*/-1,
                                     /*capacity=*/1, NULL);
    check_err_code(err, OPAL_SUCCESS);
    err = opal_bp_graph_add_edge(g, /*u=*/0, /*v=*/3, /*cost=*/-10,
                                     /*capacity=*/1, NULL);
    check_err_code(err, OPAL_SUCCESS);
    err = opal_bp_graph_add_edge(g, /*u=*/1, /*v=*/3, /*cost=*/-5,
                                     /*capacity=*/1, NULL);
    check_err_code(err, OPAL_SUCCESS);

    me = NULL;
    err = opal_bp_graph_solve_bipartite_assignment(g,
                                                    &nme,
                                                    &me);
    check_err_code(err, OPAL_SUCCESS);
    check_int_eq(nme, 2);
    check(me != NULL);
    qsort(me, nme, 2*sizeof(int), &cmp_int_pair);
    check(me[0] == 0 && me[1] == 2);
    check(me[2] == 1 && me[3] == 3);
    free(me);

    err = opal_bp_graph_free(g);
    check_err_code(err, OPAL_SUCCESS);


    /* TEST CASE: add some disconnected vertices
     * 0 --> 2
     * 0 --> 3
     * 1 --> 3
     * x --> 4
     *
     * make sure that 0-->2 & 1-->3 get chosen.
     */
    err = opal_bp_graph_create(NULL, NULL, &g);
    check_err_code(err, OPAL_SUCCESS);
    check(g != NULL);

    for (i = 0; i < 5; ++i) {
        err = opal_bp_graph_add_vertex(g, NULL, NULL);
        check_err_code(err, OPAL_SUCCESS);
    }

    err = opal_bp_graph_add_edge(g, /*u=*/0, /*v=*/2, /*cost=*/-1,
                                     /*capacity=*/1, NULL);
    check_err_code(err, OPAL_SUCCESS);
    err = opal_bp_graph_add_edge(g, /*u=*/0, /*v=*/3, /*cost=*/-10,
                                     /*capacity=*/1, NULL);
    check_err_code(err, OPAL_SUCCESS);
    err = opal_bp_graph_add_edge(g, /*u=*/1, /*v=*/3, /*cost=*/-5,
                                     /*capacity=*/1, NULL);
    check_err_code(err, OPAL_SUCCESS);

    me = NULL;
    err = opal_bp_graph_solve_bipartite_assignment(g,
                                                    &nme,
                                                    &me);
    check_err_code(err, OPAL_SUCCESS);
    check_int_eq(nme, 2);
    check(me != NULL);
    qsort(me, nme, 2*sizeof(int), &cmp_int_pair);
    check(me[0] == 0 && me[1] == 2);
    check(me[2] == 1 && me[3] == 3);
    free(me);

    err = opal_bp_graph_free(g);
    check_err_code(err, OPAL_SUCCESS);

    /* TEST CASE: sample UDP graph from bldsb005 + bldsb007
     * 0 --> 2 (cost -4294967296)
     * 1 --> 2 (cost -4294967296)
     * 0 --> 3 (cost -4294967296)
     * 1 --> 3 (cost -4294967296)
     *
     * Make sure that either (0-->2 && 1-->3) or (0-->3 && 1-->2) get chosen.
     */
    err = opal_bp_graph_create(NULL, NULL, &g);
    check_err_code(err, OPAL_SUCCESS);
    check(g != NULL);

    for (i = 0; i < 4; ++i) {
        err = opal_bp_graph_add_vertex(g, NULL, NULL);
        check_err_code(err, OPAL_SUCCESS);
    }

    err = opal_bp_graph_add_edge(g, /*u=*/0, /*v=*/2, /*cost=*/-4294967296,
                                     /*capacity=*/1, NULL);
    check_err_code(err, OPAL_SUCCESS);
    err = opal_bp_graph_add_edge(g, /*u=*/1, /*v=*/2, /*cost=*/-4294967296,
                                     /*capacity=*/1, NULL);
    check_err_code(err, OPAL_SUCCESS);
    err = opal_bp_graph_add_edge(g, /*u=*/0, /*v=*/3, /*cost=*/-4294967296,
                                     /*capacity=*/1, NULL);
    check_err_code(err, OPAL_SUCCESS);
    err = opal_bp_graph_add_edge(g, /*u=*/1, /*v=*/3, /*cost=*/-4294967296,
                                     /*capacity=*/1, NULL);
    check_err_code(err, OPAL_SUCCESS);

    me = NULL;
    err = opal_bp_graph_solve_bipartite_assignment(g,
                                                    &nme,
                                                    &me);
    check_err_code(err, OPAL_SUCCESS);
    check_int_eq(nme, 2);
    check(me != NULL);
    qsort(me, nme, 2*sizeof(int), &cmp_int_pair);
    if (me[1] == 2) {
        check(me[0] == 0 && me[1] == 2);
        check(me[2] == 1 && me[3] == 3);
    } else {
        check(me[0] == 0 && me[1] == 3);
        check(me[2] == 1 && me[3] == 2);
    }
    free(me);

    err = opal_bp_graph_free(g);
    check_err_code(err, OPAL_SUCCESS);


    /* TEST CASE: check that simple cases are solved correctly
     *
     * 0 --> 2
     * 1 --> 2
     */
    err = opal_bp_graph_create(NULL, NULL, &g);
    check_err_code(err, OPAL_SUCCESS);
    check(g != NULL);

    for (i = 0; i < 3; ++i) {
        err = opal_bp_graph_add_vertex(g, NULL, NULL);
        check_err_code(err, OPAL_SUCCESS);
    }

    err = opal_bp_graph_add_edge(g, /*u=*/0, /*v=*/2, /*cost=*/-100,
                                     /*capacity=*/1, NULL);
    err = opal_bp_graph_add_edge(g, /*u=*/1, /*v=*/2, /*cost=*/-100,
                                     /*capacity=*/1, NULL);

    me = NULL;
    err = opal_bp_graph_solve_bipartite_assignment(g,
                                                    &nme,
                                                    &me);
    check_err_code(err, OPAL_SUCCESS);
    check_int_eq(nme, 1);
    check(me != NULL);
    qsort(me, nme, 2*sizeof(int), &cmp_int_pair);
    check((me[0] == 0 || me[0] == 1) && me[1] == 2);

    err = opal_bp_graph_free(g);
    check_err_code(err, OPAL_SUCCESS);


    /* TEST CASE: performance sanity check
     *
     * Construct this graph and ensure that it doesn't take too long on a large
     * cluster (1000 nodes).
     * 0 --> 3
     * 1 --> 4
     * 2 --> 4
     */
#define NUM_ITER (10000)
    start = gettime();
    for (iter = 0; iter < NUM_ITER; ++iter) {
        err = opal_bp_graph_create(NULL, NULL, &g);
        check_err_code(err, OPAL_SUCCESS);
        check(g != NULL);

        for (i = 0; i < 5; ++i) {
            err = opal_bp_graph_add_vertex(g, NULL, NULL);
            check_err_code(err, OPAL_SUCCESS);
        }

        err = opal_bp_graph_add_edge(g, /*u=*/0, /*v=*/3, /*cost=*/10,
                                        /*capacity=*/1, NULL);
        check_err_code(err, OPAL_SUCCESS);
        err = opal_bp_graph_add_edge(g, /*u=*/1, /*v=*/4, /*cost=*/2,
                                        /*capacity=*/1, NULL);
        check_err_code(err, OPAL_SUCCESS);
        err = opal_bp_graph_add_edge(g, /*u=*/2, /*v=*/4, /*cost=*/1,
                                        /*capacity=*/1, NULL);
        check_err_code(err, OPAL_SUCCESS);

        me = NULL;
        err = opal_bp_graph_solve_bipartite_assignment(g,
                                                        &nme,
                                                        &me);
        check_err_code(err, OPAL_SUCCESS);
        check_int_eq(nme, 2);
        check(me != NULL);
        qsort(me, nme, 2*sizeof(int), &cmp_int_pair);
        check(me[0] == 0 && me[1] == 3);
        check(me[2] == 2 && me[3] == 4);
        free(me);

        err = opal_bp_graph_free(g);
        check_err_code(err, OPAL_SUCCESS);
    }
    end = gettime();
    /* ensure that this operation on a 1000 node cluster will take less than one second */
    check(((end - start) / NUM_ITER) < 0.001);
#if 0
    fprintf(stderr, "timing for %d iterations is %f seconds (%f s/iter)\n",
            NUM_ITER, end - start, (end - start) / NUM_ITER);
#endif

    return 0;
}

static int test_graph_bellman_ford(void *ctx)
{
    opal_bp_graph_t *g;
    int i;
    int err;
    bool path_found;
    int *pred;

    /* TEST CASE: check that simple cases are solved correctly
     *    -> 0 --> 2
     *   /           \
     * 4              --> 5
     *   \            /
     *    -> 1 --> 3 /
     *
     * should yield the path 5,1,3,6 (see costs in code below)
     */
    err = opal_bp_graph_create(NULL, NULL, &g);
    check_err_code(err, OPAL_SUCCESS);
    check(g != NULL);

    for (i = 0; i < 6; ++i) {
        err = opal_bp_graph_add_vertex(g, NULL, NULL);
        check_err_code(err, OPAL_SUCCESS);
    }

    err = opal_bp_graph_add_edge(g, /*u=*/0, /*v=*/2, /*cost=*/10,
                                     /*capacity=*/1, NULL);
    check_err_code(err, OPAL_SUCCESS);
    err = opal_bp_graph_add_edge(g, /*u=*/1, /*v=*/3, /*cost=*/2,
                                     /*capacity=*/1, NULL);
    check_err_code(err, OPAL_SUCCESS);
    err = opal_bp_graph_add_edge(g, /*u=*/4, /*v=*/0, /*cost=*/0,
                                     /*capacity=*/1, NULL);
    check_err_code(err, OPAL_SUCCESS);
    err = opal_bp_graph_add_edge(g, /*u=*/4, /*v=*/1, /*cost=*/0,
                                     /*capacity=*/1, NULL);
    check_err_code(err, OPAL_SUCCESS);
    err = opal_bp_graph_add_edge(g, /*u=*/2, /*v=*/5, /*cost=*/0,
                                     /*capacity=*/1, NULL);
    check_err_code(err, OPAL_SUCCESS);
    err = opal_bp_graph_add_edge(g, /*u=*/3, /*v=*/5, /*cost=*/0,
                                     /*capacity=*/1, NULL);
    check_err_code(err, OPAL_SUCCESS);

    pred = malloc(6*sizeof(*pred));
    check(pred != NULL);
    path_found = opal_bp_graph_bellman_ford(g, /*source=*/4, /*target=*/5, pred);
    check(path_found);
    check_path_cycle(6, /*source=*/4, /*target=*/5, pred);
    check_int_eq(pred[5], 3);
    check_int_eq(pred[3], 1);
    check_int_eq(pred[1], 4);
    free(pred);

    err = opal_bp_graph_free(g);
    check_err_code(err, OPAL_SUCCESS);


    /* TEST CASE: left side has more vertices than the right side, then
     * colwert to a flow network
     *
     * 0 --> 3
     * 1 --> 4
     * 2 --> 4
     */
    err = opal_bp_graph_create(NULL, NULL, &g);
    check_err_code(err, OPAL_SUCCESS);
    check(g != NULL);

    for (i = 0; i < 5; ++i) {
        err = opal_bp_graph_add_vertex(g, NULL, NULL);
        check_err_code(err, OPAL_SUCCESS);
    }

    err = opal_bp_graph_add_edge(g, /*u=*/0, /*v=*/3, /*cost=*/10,
                                     /*capacity=*/1, NULL);
    check_err_code(err, OPAL_SUCCESS);
    err = opal_bp_graph_add_edge(g, /*u=*/1, /*v=*/4, /*cost=*/2,
                                     /*capacity=*/1, NULL);
    check_err_code(err, OPAL_SUCCESS);
    err = opal_bp_graph_add_edge(g, /*u=*/2, /*v=*/4, /*cost=*/1,
                                     /*capacity=*/1, NULL);
    check_err_code(err, OPAL_SUCCESS);

    err = opal_bp_graph_bipartite_to_flow(g);
    check_err_code(err, OPAL_SUCCESS);

    pred = malloc(7*sizeof(*pred));
    check(pred != NULL);
    path_found = opal_bp_graph_bellman_ford(g, /*source=*/5, /*target=*/6, pred);
    check(path_found);
    check_int_eq(g->source_idx, 5);
    check_int_eq(g->sink_idx, 6);
    check_path_cycle(7, /*source=*/5, /*target=*/6, pred);
    check_int_eq(pred[6], 4);
    check_int_eq(pred[4], 2);
    check_int_eq(pred[2], 5);
    free(pred);

    err = opal_bp_graph_free(g);
    check_err_code(err, OPAL_SUCCESS);

    /* TEST CASE: same as previous, but with very large cost values (try to
     * catch incorrect integer colwersions)
     *
     * 0 --> 3
     * 1 --> 4
     * 2 --> 4
     */
    err = opal_bp_graph_create(NULL, NULL, &g);
    check_err_code(err, OPAL_SUCCESS);
    check(g != NULL);

    for (i = 0; i < 5; ++i) {
        err = opal_bp_graph_add_vertex(g, NULL, NULL);
        check_err_code(err, OPAL_SUCCESS);
    }

    err = opal_bp_graph_add_edge(g, /*u=*/0, /*v=*/3, /*cost=*/INT32_MAX+10LL,
                                     /*capacity=*/1, NULL);
    check_err_code(err, OPAL_SUCCESS);
    err = opal_bp_graph_add_edge(g, /*u=*/1, /*v=*/4, /*cost=*/INT32_MAX+2LL,
                                     /*capacity=*/1, NULL);
    check_err_code(err, OPAL_SUCCESS);
    err = opal_bp_graph_add_edge(g, /*u=*/2, /*v=*/4, /*cost=*/INT32_MAX+1LL,
                                     /*capacity=*/1, NULL);
    check_err_code(err, OPAL_SUCCESS);

    err = opal_bp_graph_bipartite_to_flow(g);
    check_err_code(err, OPAL_SUCCESS);

    pred = malloc(7*sizeof(*pred));
    check(pred != NULL);
    path_found = opal_bp_graph_bellman_ford(g, /*source=*/5, /*target=*/6, pred);
    check(path_found);
    check_int_eq(g->source_idx, 5);
    check_int_eq(g->sink_idx, 6);
    check_path_cycle(7, /*source=*/5, /*target=*/6, pred);
    check_int_eq(pred[6], 4);
    check_int_eq(pred[4], 2);
    check_int_eq(pred[2], 5);
    free(pred);

    err = opal_bp_graph_free(g);
    check_err_code(err, OPAL_SUCCESS);

    /* TEST CASE: left side has more vertices than the right side, then
     * colwert to a flow network.  Negative costs are used, but should not
     * result in a negative cycle.
     *
     * 0 --> 3
     * 1 --> 4
     * 2 --> 4
     */
    err = opal_bp_graph_create(NULL, NULL, &g);
    check_err_code(err, OPAL_SUCCESS);
    check(g != NULL);

    for (i = 0; i < 5; ++i) {
        err = opal_bp_graph_add_vertex(g, NULL, NULL);
        check_err_code(err, OPAL_SUCCESS);
    }

    err = opal_bp_graph_add_edge(g, /*u=*/0, /*v=*/3, /*cost=*/-1,
                                     /*capacity=*/1, NULL);
    check_err_code(err, OPAL_SUCCESS);
    err = opal_bp_graph_add_edge(g, /*u=*/1, /*v=*/4, /*cost=*/-2,
                                     /*capacity=*/1, NULL);
    check_err_code(err, OPAL_SUCCESS);
    err = opal_bp_graph_add_edge(g, /*u=*/2, /*v=*/4, /*cost=*/-10,
                                     /*capacity=*/1, NULL);
    check_err_code(err, OPAL_SUCCESS);

    err = opal_bp_graph_bipartite_to_flow(g);
    check_err_code(err, OPAL_SUCCESS);

    pred = malloc(7*sizeof(*pred));
    check(pred != NULL);
    path_found = opal_bp_graph_bellman_ford(g, /*source=*/5, /*target=*/6, pred);
    check(path_found);
    check_int_eq(g->source_idx, 5);
    check_int_eq(g->sink_idx, 6);
    check_path_cycle(7, /*source=*/5, /*target=*/6, pred);
    check_int_eq(pred[6], 4);
    check_int_eq(pred[4], 2);
    check_int_eq(pred[2], 5);
    free(pred);

    err = opal_bp_graph_free(g);
    check_err_code(err, OPAL_SUCCESS);

    return 0;
}

static int test_graph_flow_colwersion(void *ctx)
{
    opal_bp_graph_t *g;
    int i;
    int err;

    /* TEST CASE: left side has more vertices than the right side, then
     * colwert to a flow network
     *
     * 0 --> 3
     * 1 --> 4
     * 2 --> 4
     */
    err = opal_bp_graph_create(NULL, NULL, &g);
    check_err_code(err, OPAL_SUCCESS);
    check(g != NULL);

    for (i = 0; i < 5; ++i) {
        err = opal_bp_graph_add_vertex(g, NULL, NULL);
        check_err_code(err, OPAL_SUCCESS);
    }

    err = opal_bp_graph_add_edge(g, /*u=*/0, /*v=*/3, /*cost=*/10,
                                     /*capacity=*/1, NULL);
    check_err_code(err, OPAL_SUCCESS);
    err = opal_bp_graph_add_edge(g, /*u=*/1, /*v=*/4, /*cost=*/2,
                                     /*capacity=*/1, NULL);
    check_err_code(err, OPAL_SUCCESS);
    err = opal_bp_graph_add_edge(g, /*u=*/2, /*v=*/4, /*cost=*/1,
                                     /*capacity=*/1, NULL);
    check_err_code(err, OPAL_SUCCESS);

    check_int_eq(opal_bp_graph_order(g), 5);
    check_has_in_out_degree(g, 0, /*exp_indeg=*/0, /*exp_outdeg=*/1);
    check_has_in_out_degree(g, 1, /*exp_indeg=*/0, /*exp_outdeg=*/1);
    check_has_in_out_degree(g, 2, /*exp_indeg=*/0, /*exp_outdeg=*/1);
    check_has_in_out_degree(g, 3, /*exp_indeg=*/1, /*exp_outdeg=*/0);
    check_has_in_out_degree(g, 4, /*exp_indeg=*/2, /*exp_outdeg=*/0);

    /* this should add two nodes and a bunch of edges */
    err = opal_bp_graph_bipartite_to_flow(g);
    check_err_code(err, OPAL_SUCCESS);

    check_int_eq(opal_bp_graph_order(g), 7);
    check_has_in_out_degree(g, 0, /*exp_indeg=*/2, /*exp_outdeg=*/2);
    check_has_in_out_degree(g, 1, /*exp_indeg=*/2, /*exp_outdeg=*/2);
    check_has_in_out_degree(g, 2, /*exp_indeg=*/2, /*exp_outdeg=*/2);
    check_has_in_out_degree(g, 3, /*exp_indeg=*/2, /*exp_outdeg=*/2);
    check_has_in_out_degree(g, 4, /*exp_indeg=*/3, /*exp_outdeg=*/3);
    check_has_in_out_degree(g, 5, /*exp_indeg=*/3, /*exp_outdeg=*/3);
    check_has_in_out_degree(g, 6, /*exp_indeg=*/2, /*exp_outdeg=*/2);

    err = opal_bp_graph_free(g);
    check_err_code(err, OPAL_SUCCESS);


    /* TEST CASE: empty graph
     *
     * there's no reason that the code should bother to support this, it's not
     * useful
     */
    err = opal_bp_graph_create(NULL, NULL, &g);
    check_err_code(err, OPAL_SUCCESS);
    check(g != NULL);
    check_int_eq(opal_bp_graph_order(g), 0);
    err = opal_bp_graph_bipartite_to_flow(g);
    check_err_code(err, OPAL_ERR_BAD_PARAM);
    err = opal_bp_graph_free(g);
    check_err_code(err, OPAL_SUCCESS);

    return 0;
}

static int test_graph_param_checking(void *ctx)
{
    opal_bp_graph_t *g;
    int i;
    int err;

    err = opal_bp_graph_create(NULL, NULL, &g);
    check_err_code(err, OPAL_SUCCESS);
    check(g != NULL);

    /* try with no vertices */
    err = opal_bp_graph_add_edge(g, /*u=*/3, /*v=*/5, /*cost=*/0,
                                     /*capacity=*/1, NULL);
    check_err_code(err, OPAL_ERR_BAD_PARAM);

    for (i = 0; i < 6; ++i) {
        err = opal_bp_graph_add_vertex(g, NULL, NULL);
        check_err_code(err, OPAL_SUCCESS);
    }

    /* try u out of range */
    err = opal_bp_graph_add_edge(g, /*u=*/9, /*v=*/5, /*cost=*/0,
                                     /*capacity=*/1, NULL);
    check_err_code(err, OPAL_ERR_BAD_PARAM);
    err = opal_bp_graph_add_edge(g, /*u=*/6, /*v=*/5, /*cost=*/0,
                                     /*capacity=*/1, NULL);
    check_err_code(err, OPAL_ERR_BAD_PARAM);

    /* try v out of range */
    err = opal_bp_graph_add_edge(g, /*u=*/2, /*v=*/8, /*cost=*/0,
                                     /*capacity=*/1, NULL);
    check_err_code(err, OPAL_ERR_BAD_PARAM);
    err = opal_bp_graph_add_edge(g, /*u=*/2, /*v=*/6, /*cost=*/0,
                                     /*capacity=*/1, NULL);
    check_err_code(err, OPAL_ERR_BAD_PARAM);

    /* try adding an edge that already exists */
    err = opal_bp_graph_add_edge(g, /*u=*/2, /*v=*/4, /*cost=*/0,
                                     /*capacity=*/1, NULL);
    check_err_code(err, OPAL_SUCCESS);
    err = opal_bp_graph_add_edge(g, /*u=*/2, /*v=*/4, /*cost=*/0,
                                     /*capacity=*/1, NULL);
    check_err_code(err, OPAL_EXISTS);

    /* try an edge with an out of range cost */
    err = opal_bp_graph_add_edge(g, /*u=*/2, /*v=*/3, /*cost=*/INT64_MAX,
                                     /*capacity=*/1, NULL);
    check_err_code(err, OPAL_ERR_BAD_PARAM);
    err = opal_bp_graph_add_edge(g, /*u=*/2, /*v=*/3, /*cost=*/INT64_MAX-1,
                                     /*capacity=*/1, NULL);
    check_err_code(err, OPAL_SUCCESS);

    err = opal_bp_graph_free(g);
    check_err_code(err, OPAL_SUCCESS);

    return 0;
}

static int test_graph_helper_macros(void *ctx)
{
    int u, v;
    int pred[6];
    bool visited[6][6];
    int pair1[2];
    int pair2[2];

#define RESET_ARRAYS(n, pred, visited) \
    do {                               \
        for (u = 0; u < 6; ++u) {      \
            pred[u] = -1;              \
            for (v = 0; v < 6; ++v) {  \
                visited[u][v] = false; \
            }                          \
        }                              \
    } while (0)

    /* TEST CASE: make sure that an empty path does not cause any edges to be
     * visited */
    RESET_ARRAYS(6, pred, visited);
    FOREACH_UV_ON_PATH(pred, 3, 5, u, v) {
        visited[u][v] = true;
    }
    for (u = 0; u < 6; ++u) {
        for (v = 0; v < 6; ++v) {
            check(visited[u][v] == false);
        }
    }

    /* TEST CASE: make sure that every edge in the given path gets visited */
    RESET_ARRAYS(6, pred, visited);
    pred[5] = 2;
    pred[2] = 1;
    pred[1] = 3;
    FOREACH_UV_ON_PATH(pred, 3, 5, u, v) {
        visited[u][v] = true;
    }
    for (u = 0; u < 6; ++u) {
        for (v = 0; v < 6; ++v) {
            if ((u == 2 && v == 5) ||
                (u == 1 && v == 2) ||
                (u == 3 && v == 1)) {
                check(visited[u][v] == true);
            }
            else {
                check(visited[u][v] == false);
            }
        }
    }

#undef RESET_ARRAYS

    /* not technically a macro, but make sure that the pair comparison function
     * isn't broken (because it was in an earlier revision...) */
    pair1[0] = 0; pair1[1] = 1;
    pair2[0] = 0; pair2[1] = 1;
    check(cmp_int_pair(&pair1[0], &pair2[0]) == 0);

    pair1[0] = 1; pair1[1] = 1;
    pair2[0] = 0; pair2[1] = 1;
    check(cmp_int_pair(pair1, pair2) > 0);

    pair1[0] = 0; pair1[1] = 1;
    pair2[0] = 1; pair2[1] = 1;
    check(cmp_int_pair(pair1, pair2) < 0);

    pair1[0] = 1; pair1[1] = 0;
    pair2[0] = 1; pair2[1] = 1;
    check(cmp_int_pair(pair1, pair2) < 0);

    pair1[0] = 1; pair1[1] = 1;
    pair2[0] = 1; pair2[1] = 0;
    check(cmp_int_pair(pair1, pair2) > 0);

    return 0;
}

int main(int argc, char *argv[])
{
    check(test_graph_create(NULL) == 0);
    check(test_graph_clone(NULL) == 0);
    check(test_graph_accessors(NULL) == 0);
    check(test_graph_assignment_solver(NULL) == 0);
    check(test_graph_bellman_ford(NULL) == 0);
    check(test_graph_flow_colwersion(NULL) == 0);
    check(test_graph_param_checking(NULL) == 0);
    check(test_graph_helper_macros(NULL) == 0);

    return 0;
}
