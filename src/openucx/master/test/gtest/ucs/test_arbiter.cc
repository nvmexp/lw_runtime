/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/


#include <common/test.h>
#include <common/test_helpers.h>

extern "C" {
#include <ucs/sys/sys.h>
#include <ucs/datastruct/arbiter.h>
}
#include <set>

class test_arbiter : public ucs::test {
protected:

    static ucs_arbiter_cb_result_t resched_groups(ucs_arbiter_t *arbitrer,
                                                  ucs_arbiter_elem_t *elem,
                                                  void *arg)
    {
        int *counter = (int*)arg;
        if (*counter == 0) {
            return UCS_ARBITER_CB_RESULT_REMOVE_ELEM;
        } else {
            --(*counter);
            return UCS_ARBITER_CB_RESULT_RESCHED_GROUP;
        }
    }

    struct arb_elem {
        unsigned           group_idx;
        unsigned           elem_idx;
        int                count;
        bool               last;
        bool               release;
        ucs_arbiter_elem_t elem;
    };

    void skip_empty_groups()
    {
        while (m_empty_groups.find(m_expected_group_idx) != m_empty_groups.end()) {
            advance_expected_group();
        }
    }

    void advance_expected_group()
    {
        ++m_expected_group_idx;
        if (m_expected_group_idx >= m_num_groups) {
            m_expected_group_idx = 0;
        }
    }

    static void release_element(arb_elem *e)
    {
        memset(e, 0xCC, sizeof(*e)); /* Ilwalidate memory to catch use-after-free bugs */
        delete e;
    }

    void prepare_groups(ucs_arbiter_group_t *groups, ucs_arbiter_elem_t *elems,
                        const int N, const int nelems_per_group, bool push_head)
    {
        int i, j;

        for (i = 0; i < N; i++) {
            ucs_arbiter_group_init(&groups[i]);
            for (j = 0; j < nelems_per_group; j++) {
                ucs_arbiter_elem_init(&elems[nelems_per_group*i+j]);
            }
            for (j = 0; j < nelems_per_group; j++) {
                if (push_head) {
                    int rev_j = nelems_per_group - 1 - j;
                    ucs_arbiter_group_push_head_elem(NULL, &groups[i],
                                                     &elems[nelems_per_group*i+rev_j]);
                } else {
                    ucs_arbiter_group_push_elem(&groups[i],
                                                &elems[nelems_per_group*i+j]);
                }
            }
            ucs_arbiter_group_schedule(&m_arb1, &groups[i]);
        }
    }

    ucs_arbiter_cb_result_t dispatch(ucs_arbiter_t *arbiter,
                                     ucs_arbiter_elem_t *elem)
    {
        arb_elem *e = ucs_container_of(elem, arb_elem, elem);

        skip_empty_groups();

        EXPECT_EQ(m_expected_group_idx,               e->group_idx);
        EXPECT_EQ(m_expected_elem_idx[e->group_idx],  e->elem_idx);

        advance_expected_group();

        /* Sometimes we just move to the next group */
        if ((ucs::rand() % 5) == 0) {
            return UCS_ARBITER_CB_RESULT_NEXT_GROUP;
        }

        /* Sometimes we want to detach the whole group */
        if ((ucs::rand() % 10) == 0) {
            m_empty_groups.insert(e->group_idx);
            m_detached_groups.insert(e->group_idx);
            return UCS_ARBITER_CB_RESULT_DESCHED_GROUP;
        }

        ++m_expected_elem_idx[e->group_idx];

        if (e->last) {
            m_empty_groups.insert(e->group_idx);
        }
        release_element(e);

        return UCS_ARBITER_CB_RESULT_REMOVE_ELEM;
    }

    static ucs_arbiter_cb_result_t dispatch_cb(ucs_arbiter_t *arbiter,
                                               ucs_arbiter_elem_t *elem,
                                               void *arg)
    {
        test_arbiter *self = (test_arbiter *)arg;
        return self->dispatch(arbiter, elem);
    }

    static ucs_arbiter_cb_result_t dispatch_dummy_cb(ucs_arbiter_t *arbiter,
                                                     ucs_arbiter_elem_t *elem,
                                                     void *arg)
    {
        return UCS_ARBITER_CB_RESULT_REMOVE_ELEM;
    }

    ucs_arbiter_cb_result_t desched_group(ucs_arbiter_elem_t *elem)
    {
        ucs_arbiter_group_t *g = ucs_arbiter_elem_group(elem);
        //ucs_warn("desched group %d", m_count);
        m_count++;
        ucs_arbiter_group_schedule(&m_arb2, g);
        return UCS_ARBITER_CB_RESULT_DESCHED_GROUP;
    }

    ucs_arbiter_cb_result_t remove_elem(ucs_arbiter_elem_t *elem)
    {
        m_count++;
        return UCS_ARBITER_CB_RESULT_REMOVE_ELEM;
    }

    static ucs_arbiter_cb_result_t desched_cb(ucs_arbiter_t *arbiter,
                                              ucs_arbiter_elem_t *elem,
                                              void *arg)
    {
        test_arbiter *self = (test_arbiter *)arg;
        return self->desched_group(elem);
    }

    static ucs_arbiter_cb_result_t remove_cb(ucs_arbiter_t *arbiter,
                                              ucs_arbiter_elem_t *elem,
                                              void *arg)
    {
        test_arbiter *self = (test_arbiter *)arg;
        return self->remove_elem(elem);
    }

    static ucs_arbiter_cb_result_t stop_cb(ucs_arbiter_t *arbiter,
                                           ucs_arbiter_elem_t *elem,
                                           void *arg)
    {
        return UCS_ARBITER_CB_RESULT_STOP;
    }

    static ucs_arbiter_cb_result_t purge_cb(ucs_arbiter_t *arbiter,
                                            ucs_arbiter_elem_t *elem,
                                            void *arg)
    {
        arb_elem *e = ucs_container_of(elem, arb_elem, elem);
        release_element(e);
        return UCS_ARBITER_CB_RESULT_REMOVE_ELEM;
    }

    static ucs_arbiter_cb_result_t count_cb(ucs_arbiter_t *arbiter,
                                            ucs_arbiter_elem_t *elem,
                                            void *arg)
    {
        test_arbiter *self = static_cast<test_arbiter*>(arg);
        arb_elem *e = ucs_container_of(elem, arb_elem, elem);

        ++e->count;
        ++self->m_count;
        return UCS_ARBITER_CB_RESULT_RESCHED_GROUP;
    }

    static ucs_arbiter_cb_result_t purge_cond_cb(ucs_arbiter_t *arbiter,
                                                 ucs_arbiter_elem_t *elem,
                                                 void *arg)
    {
        test_arbiter *self = static_cast<test_arbiter*>(arg);
        arb_elem *e        = ucs_container_of(elem, arb_elem, elem);

        if (e->release) {
            ++self->m_count;
            return UCS_ARBITER_CB_RESULT_REMOVE_ELEM;
        }

        return UCS_ARBITER_CB_RESULT_NEXT_GROUP;
    }


    static ucs_arbiter_cb_result_t purge_dummy_cb(ucs_arbiter_t *arbiter,
                                                  ucs_arbiter_elem_t *elem,
                                                  void *arg)
    {
        test_arbiter *self = static_cast<test_arbiter*>(arg);
        ++self->m_count;
        return UCS_ARBITER_CB_RESULT_REMOVE_ELEM;
    }

    void test_move_groups(int N, int nelems, bool push_head = false)
    {

        ucs_arbiter_group_t *groups;
        ucs_arbiter_elem_t  *elems;

        ucs_arbiter_init(&m_arb1);
        ucs_arbiter_init(&m_arb2);

        groups = new ucs_arbiter_group_t [N];
        elems  = new ucs_arbiter_elem_t [nelems*N];

        prepare_groups(groups, elems, N, nelems, push_head);

        m_count = 0;
        ucs_arbiter_dispatch(&m_arb1, 1, desched_cb, this);
        EXPECT_EQ(N, m_count);

        m_count = 0;
        ucs_arbiter_dispatch(&m_arb2, 1, remove_cb, this);
        EXPECT_EQ(nelems*N, m_count);

        m_count = 0;
        ucs_arbiter_dispatch(&m_arb1, 1, remove_cb, this);
        EXPECT_EQ(0, m_count);

        delete [] groups;
        delete [] elems;

        ucs_arbiter_cleanup(&m_arb1);
        ucs_arbiter_cleanup(&m_arb2);
    }

protected:
    std::set<unsigned>    m_empty_groups;
    std::set<unsigned>    m_detached_groups;
    std::vector<unsigned> m_expected_elem_idx;
    unsigned              m_expected_group_idx;
    unsigned              m_num_groups;
    ucs_arbiter_t         m_arb1;
    ucs_arbiter_t         m_arb2;
    int                   m_count;
};


UCS_TEST_F(test_arbiter, add_purge) {

    ucs_arbiter_t arbiter;

    ucs_arbiter_group_t group1;
    ucs_arbiter_group_t group2;

    ucs_arbiter_init(&arbiter);
    ucs_arbiter_group_init(&group1);
    ucs_arbiter_group_init(&group2);


    ucs_arbiter_elem_t elem1;
    ucs_arbiter_elem_t elem2_1;
    ucs_arbiter_elem_t elem2_2;

    /* test internal function */
    ucs_arbiter_group_push_elem_always(&group1, &elem1);
    ucs_arbiter_group_push_elem_always(&group2, &elem2_1);
    ucs_arbiter_group_push_elem_always(&group2, &elem2_2);

    ucs_arbiter_group_schedule(&arbiter, &group1);
    ucs_arbiter_group_schedule(&arbiter, &group2);

    m_count = 0;
    ucs_arbiter_dispatch_nonempty(&arbiter, 3, remove_cb, this);

    EXPECT_EQ(3, m_count);

    ucs_arbiter_group_cleanup(&group2);
    ucs_arbiter_group_cleanup(&group1);
    ucs_arbiter_cleanup(&arbiter);
}

UCS_TEST_F(test_arbiter, purge_cond) {

    int num_elems = m_num_groups = 25;
    ucs_arbiter_group_t groups[m_num_groups];
    int purged_count[m_num_groups];
    ucs::ptr_vector<arb_elem> elems;

    ucs_arbiter_t arbiter;
    ucs_arbiter_init(&arbiter);
    memset(purged_count, 0, sizeof(int) * m_num_groups);

    for (unsigned i = 0; i < m_num_groups; ++i) {
        ucs_arbiter_group_init(&groups[i]);

        for (int j = 0; j < num_elems; ++j) {
            arb_elem *e = new arb_elem;
            if (ucs::rand() % 2) {
                e->release = true;
                ++purged_count[i];
            } else {
                e->release = false;
            }
            ucs_arbiter_elem_init(&e->elem);
            elems.push_back(e);
            ucs_arbiter_group_push_elem(&groups[i], &e->elem);
            /* coverity[leaked_storage] */
        }
        ucs_arbiter_group_schedule(&arbiter, &groups[i]);
    }

    // All groups are scheduled, start purging them from some non-current group
    // (purge just half of the groups, the rest will be dispatched)
    unsigned start = ucs::rand() % m_num_groups;
    for (unsigned i = 0; i < m_num_groups / 2; ++i) {
        unsigned idx = (start + i) % m_num_groups;
        m_count = 0;
        ucs_arbiter_group_purge(&arbiter, &groups[idx], purge_cond_cb, this);
        EXPECT_EQ(m_count, purged_count[idx]);

        m_count = 0;
        ucs_arbiter_group_purge(&arbiter, &groups[idx], purge_dummy_cb, this);
        EXPECT_EQ(m_count, num_elems - purged_count[idx]);

        ucs_arbiter_group_cleanup(&groups[idx]);
    }

    ucs_arbiter_dispatch(&arbiter, 1, dispatch_dummy_cb, NULL);

    ucs_arbiter_cleanup(&arbiter);
}

UCS_TEST_F(test_arbiter, purge_corner) {

    ucs_arbiter_t arbiter;
    ucs_arbiter_group_t group;
    arb_elem elems[2];

    ucs_arbiter_init(&arbiter);
    ucs_arbiter_group_init(&group);

    for (int i = 0; i < 2; ++i) {
        ucs_arbiter_elem_init(&elems[i].elem);
        elems[i].release = !i; // try to purge first
        ucs_arbiter_group_push_elem(&group, &elems[i].elem);
    }
    m_count = 0;
    ucs_arbiter_group_purge(&arbiter, &group, purge_cond_cb, this);
    EXPECT_EQ(1, m_count);
    EXPECT_FALSE(ucs_arbiter_group_is_empty(&group));
    EXPECT_FALSE(ucs_arbiter_elem_is_scheduled(&elems[0].elem));
    EXPECT_TRUE(ucs_arbiter_elem_is_scheduled(&elems[1].elem));

    // try to reuse 0-th element and purge it (now it is last element)
    ucs_arbiter_group_push_elem(&group, &elems[0].elem);
    EXPECT_EQ(true, elems[0].release);
    m_count = 0;
    ucs_arbiter_group_purge(&arbiter, &group, purge_cond_cb, this);
    EXPECT_EQ(1, m_count);
    EXPECT_FALSE(ucs_arbiter_elem_is_scheduled(&elems[0].elem));
    EXPECT_TRUE(ucs_arbiter_elem_is_scheduled(&elems[1].elem));

    // clear the group
    ucs_arbiter_group_purge(&arbiter, &group, purge_dummy_cb, this);
    EXPECT_FALSE(ucs_arbiter_elem_is_scheduled(&elems[0].elem));
    EXPECT_FALSE(ucs_arbiter_elem_is_scheduled(&elems[1].elem));

    ucs_arbiter_group_cleanup(&group);
    ucs_arbiter_cleanup(&arbiter);
}

UCS_TEST_F(test_arbiter, multiple_dispatch) {
    m_num_groups = 20;

    ucs_arbiter_t arbiter;
    ucs_arbiter_init(&arbiter);

    std::vector<ucs_arbiter_group_t> groups(m_num_groups);
    for (unsigned i = 0; i < m_num_groups; ++i) {
        ucs_arbiter_group_init(&groups[i]);

        unsigned num_elems = ucs::rand() % 9;

        for (unsigned j = 0; j < num_elems; ++j) {
            arb_elem *e = new arb_elem;
            e->group_idx = i;
            e->elem_idx  = j;
            e->release   = true;
            e->last      = (j == num_elems - 1);
            ucs_arbiter_elem_init(&e->elem);
            ucs_arbiter_group_push_elem(&groups[i], &e->elem);
            /* coverity[leaked_storage] */
        }

        if (num_elems == 0) {
            m_empty_groups.insert(i);
        }

        ucs_arbiter_group_schedule(&arbiter, &groups[i]);
    }

    m_expected_group_idx = 0;
    m_expected_elem_idx.resize(m_num_groups, 0);
    std::fill(m_expected_elem_idx.begin(), m_expected_elem_idx.end(), 0);

    ucs_arbiter_dispatch(&arbiter, 1, dispatch_cb, this);

    ASSERT_TRUE(ucs_arbiter_is_empty(&arbiter));

    /* Release detached groups */
    for (unsigned i = 0; i < m_num_groups; ++i) {
        if (m_detached_groups.find(i) != m_detached_groups.end()) {
            ucs_arbiter_group_purge(&arbiter, &groups[i], purge_cb, NULL);
        }
        ucs_arbiter_group_cleanup(&groups[i]);
    }

    ucs_arbiter_cleanup(&arbiter);
}

UCS_TEST_F(test_arbiter, resched) {

    ucs_arbiter_t arbiter;

    ucs_arbiter_group_t group1;
    ucs_arbiter_group_t group2;

    ucs_arbiter_init(&arbiter);
    ucs_arbiter_group_init(&group1);
    ucs_arbiter_group_init(&group2);


    ucs_arbiter_elem_t elem1;
    ucs_arbiter_elem_t elem2_1;

    ucs_arbiter_elem_init(&elem1);
    ucs_arbiter_elem_init(&elem2_1);
    ucs_arbiter_group_push_elem(&group1, &elem1);
    ucs_arbiter_group_push_elem(&group2, &elem2_1);

    ucs_arbiter_group_schedule(&arbiter, &group1);
    ucs_arbiter_group_schedule(&arbiter, &group2);

    int count = 2;
    ucs_arbiter_dispatch_nonempty(&arbiter, 1, resched_groups, &count);

    EXPECT_EQ(0, count);

    count = 1;
    ucs_arbiter_dispatch_nonempty(&arbiter, 1, resched_groups, &count);
    EXPECT_EQ(0, count);

    /* one group with one elem should be there */
    m_count = 0;
    ucs_arbiter_dispatch_nonempty(&arbiter, 3, remove_cb, this);
    EXPECT_EQ(1, m_count);
    ASSERT_TRUE(ucs_arbiter_is_empty(&arbiter));

    ucs_arbiter_group_cleanup(&group2);
    ucs_arbiter_group_cleanup(&group1);
    ucs_arbiter_cleanup(&arbiter);
}

/* check that it is possible to reuse removed 
 * element
 */
UCS_TEST_F(test_arbiter, reuse_elem) {
    int i;
    ucs_arbiter_group_t group1;
    ucs_arbiter_elem_t elem1;
    ucs_arbiter_elem_t elem2;

    ucs_arbiter_init(&m_arb1);
    ucs_arbiter_group_init(&group1);
    ucs_arbiter_elem_init(&elem1);
    ucs_arbiter_elem_init(&elem2);

    for (i = 0; i < 3; i++) {
        ucs_arbiter_group_push_elem(&group1, &elem1);
        ucs_arbiter_group_push_elem(&group1, &elem2);
        ucs_arbiter_group_schedule(&m_arb1, &group1);

        m_count = 0;
        ucs_arbiter_dispatch(&m_arb1, 1, remove_cb, this);
        EXPECT_EQ(2, m_count);
    }

    for (i = 0; i < 3; i++) {
        ucs_arbiter_group_push_elem(&group1, &elem1);
        ucs_arbiter_group_schedule(&m_arb1, &group1);
        m_count = 0;
        ucs_arbiter_dispatch(&m_arb1, 1, remove_cb, this);
        EXPECT_EQ(1, m_count);
    }
}

UCS_TEST_F(test_arbiter, move_group) {

    ucs_arbiter_group_t group1;
    ucs_arbiter_elem_t elem1;
    ucs_arbiter_elem_t elem2;

    ucs_arbiter_init(&m_arb1);
    ucs_arbiter_init(&m_arb2);

    ucs_arbiter_group_init(&group1);
    ucs_arbiter_elem_init(&elem1);
    ucs_arbiter_elem_init(&elem2);
    ucs_arbiter_group_push_elem(&group1, &elem1);
    ucs_arbiter_group_push_elem(&group1, &elem2);
    ucs_arbiter_group_schedule(&m_arb1, &group1);

    m_count = 0;
    ucs_arbiter_dispatch(&m_arb1, 1, desched_cb, this);
    EXPECT_EQ(1, m_count);

    m_count = 0;
    ucs_arbiter_dispatch(&m_arb2, 1, remove_cb, this);
    EXPECT_EQ(2, m_count);

    m_count = 0;
    ucs_arbiter_dispatch(&m_arb1, 1, remove_cb, this);
    EXPECT_EQ(0, m_count);

    ucs_arbiter_cleanup(&m_arb1);
    ucs_arbiter_cleanup(&m_arb2);
}

UCS_TEST_F(test_arbiter, push_head_scheduled) {

    ucs_arbiter_group_t group1;
    ucs_arbiter_group_t group2;
    arb_elem elem1;
    arb_elem elem2;
    arb_elem elem3;

    ucs_arbiter_init(&m_arb1);

    ucs_arbiter_group_init(&group1);
    ucs_arbiter_group_init(&group2);
    ucs_arbiter_elem_init(&elem1.elem);
    ucs_arbiter_elem_init(&elem2.elem);
    ucs_arbiter_elem_init(&elem3.elem);
    elem1.count = elem2.count = elem3.count = 0;

    ucs_arbiter_group_push_head_elem(&m_arb1, &group1, &elem1.elem);
    ucs_arbiter_group_push_head_elem(&m_arb1, &group2, &elem2.elem);

    ucs_arbiter_group_schedule(&m_arb1, &group1);
    ucs_arbiter_group_schedule(&m_arb1, &group2);

    m_count = 0;
    ucs_arbiter_dispatch(&m_arb1, 1, count_cb, this);
    EXPECT_EQ(2, m_count);
    EXPECT_EQ(1, elem1.count);
    EXPECT_EQ(1, elem2.count);
    EXPECT_EQ(0, elem3.count);

    /* Adding new head elem to group2 */
    ucs_arbiter_group_push_head_elem(&m_arb1, &group2, &elem3.elem);

    m_count = 0;
    ucs_arbiter_dispatch(&m_arb1, 1, count_cb, this);
    EXPECT_EQ(2, m_count);
    EXPECT_EQ(2, elem1.count);
    EXPECT_EQ(1, elem2.count);
    EXPECT_EQ(1, elem3.count);

    m_count = 0;
    ucs_arbiter_dispatch(&m_arb1, 2, remove_cb, this);
    EXPECT_EQ(3, m_count);

    /* Add to single scheduled group */
    ucs_arbiter_group_push_head_elem(&m_arb1, &group2, &elem2.elem);
    ucs_arbiter_group_schedule(&m_arb1, &group2);
    ucs_arbiter_group_push_head_elem(&m_arb1, &group2, &elem3.elem);

    m_count = 0;
    elem2.count = elem3.count = 0;
    ucs_arbiter_dispatch(&m_arb1, 2, count_cb, this);
    EXPECT_EQ(0, elem2.count);
    EXPECT_EQ(1, elem3.count);
    EXPECT_EQ(1, m_count);

    m_count = 0;
    ucs_arbiter_dispatch(&m_arb1, 2, remove_cb, this);
    EXPECT_EQ(2, m_count);

    ucs_arbiter_cleanup(&m_arb1);
}

UCS_TEST_F(test_arbiter, move_groups1) {
    test_move_groups(42, 3);
}

UCS_TEST_F(test_arbiter, move_groups2) {
    test_move_groups(42, 1);
}

UCS_TEST_F(test_arbiter, move_groups1_push_head) {
    test_move_groups(42, 3, true);
}

UCS_TEST_F(test_arbiter, move_groups2_push_head) {
    test_move_groups(42, 1, true);
}

UCS_TEST_F(test_arbiter, desched_group) {
    ucs_arbiter_group_t group1;
    ucs_arbiter_elem_t elem1;
    ucs_arbiter_elem_t elem2;

    ucs_arbiter_init(&m_arb1);

    ucs_arbiter_group_init(&group1);
    ucs_arbiter_elem_init(&elem1);
    ucs_arbiter_elem_init(&elem2);
    ucs_arbiter_group_push_elem(&group1, &elem1);
    ucs_arbiter_group_push_elem(&group1, &elem2);

    /* should do nothing */
    ucs_arbiter_group_desched(&m_arb1, &group1);

    ucs_arbiter_group_schedule(&m_arb1, &group1);
    /* arbiter will be empty */
    ucs_arbiter_group_desched(&m_arb1, &group1);
    m_count = 0;
    ucs_arbiter_dispatch(&m_arb1, 1, remove_cb, this);
    EXPECT_EQ(0, m_count);

    /* group must still have 2 elements */
    ucs_arbiter_group_schedule(&m_arb1, &group1);
    m_count = 0;
    ucs_arbiter_dispatch(&m_arb1, 1, remove_cb, this);
    EXPECT_EQ(2, m_count);

    ucs_arbiter_cleanup(&m_arb1);
}

UCS_TEST_F(test_arbiter, desched_groups) {
    ucs_arbiter_group_t *groups;
    ucs_arbiter_elem_t  *elems;
    const int N = 17;

    ucs_arbiter_init(&m_arb1);
    ucs_arbiter_init(&m_arb2);

    groups = new ucs_arbiter_group_t [N];
    elems  = new ucs_arbiter_elem_t [3*N];

    prepare_groups(groups, elems, N, 3, false);

    ucs_arbiter_group_desched(&m_arb1, &groups[N-1]);
    ucs_arbiter_group_desched(&m_arb1, &groups[0]);
    ucs_arbiter_group_desched(&m_arb1, &groups[5]);
    ucs_arbiter_group_desched(&m_arb1, &groups[11]);
    m_count = 0;
    ucs_arbiter_dispatch(&m_arb1, 1, remove_cb, this);
    /* 4 groups with 3 elems each were descheduled */
    EXPECT_EQ(3*(N-4), m_count);

    ucs_arbiter_group_schedule(&m_arb1, &groups[N-1]);
    ucs_arbiter_group_schedule(&m_arb1, &groups[0]);
    ucs_arbiter_group_schedule(&m_arb1, &groups[5]);
    ucs_arbiter_group_schedule(&m_arb1, &groups[11]);

    m_count = 0;
    ucs_arbiter_dispatch(&m_arb1, 1, remove_cb, this);
    /* 4 groups with 3 elems each were scheduled */
    EXPECT_EQ(4*3, m_count);

    delete [] groups;
    delete [] elems;

    ucs_arbiter_cleanup(&m_arb1);
}

/* make sure that next arbiter dispatch
 * continues from the group that stopped
 */
UCS_TEST_F(test_arbiter, result_stop) {

    const int N = 5;
    const int nelems = 1;
    ucs_arbiter_group_t *groups;
    ucs_arbiter_elem_t  *elems;

    groups = new ucs_arbiter_group_t [N];
    elems  = new ucs_arbiter_elem_t [nelems*N];
    ucs_arbiter_init(&m_arb1);

    prepare_groups(groups, elems, N, nelems, false);

    for (int i = 0; i < N + 3; i++) {
       ucs_arbiter_dispatch(&m_arb1, 1, stop_cb, this);
       /* arbiter current position must not change on STOP */
       EXPECT_EQ(m_arb1.list.next, &groups[0].tail->next->list);
    }

    m_count = 0;
    ucs_arbiter_dispatch(&m_arb1, 1, remove_cb, this);
    EXPECT_EQ(N*nelems, m_count);

    ucs_arbiter_cleanup(&m_arb1);

    delete [] groups;
    delete [] elems;
}

class test_arbiter_resched_from_dispatch : public ucs::test {
public:
    virtual void init() {
        ucs::test::init();
        ucs_arbiter_init(&m_arb);
        ucs_arbiter_group_init(&m_group1);
        ucs_arbiter_group_init(&m_group2);
        ucs_arbiter_elem_init(&m_elem);
    }

    virtual void cleanup() {
        ucs_arbiter_cleanup(&m_arb);
        ucs::test::cleanup();
    }

protected:
    ucs_arbiter_cb_result_t dispatch(ucs_arbiter_elem_t *elem) {
        EXPECT_EQ(&m_elem, elem);
        if (m_moved) {
            return UCS_ARBITER_CB_RESULT_STOP;
        } else {
            ucs_arbiter_group_push_elem(&m_group2, elem);
            ucs_arbiter_group_schedule(&m_arb, &m_group2);
            m_moved = true;
        }
        return UCS_ARBITER_CB_RESULT_REMOVE_ELEM;
    }

    static ucs_arbiter_cb_result_t purge_cb(ucs_arbiter_t *arbiter,
                                            ucs_arbiter_elem_t *elem, void *arg)
    {
        return UCS_ARBITER_CB_RESULT_REMOVE_ELEM;
    }

    static ucs_arbiter_cb_result_t dispatch_cb(ucs_arbiter_t *arbiter,
                                               ucs_arbiter_elem_t *elem,
                                               void *arg)
    {
        test_arbiter_resched_from_dispatch *self =
                reinterpret_cast<test_arbiter_resched_from_dispatch*>(arg);
        return self->dispatch(elem);
    }

    void check_group_state(ucs_arbiter_group_t *group, bool is_scheduled)
    {
        EXPECT_EQ(is_scheduled, ucs_arbiter_group_is_scheduled(group));
        EXPECT_EQ(is_scheduled, !ucs_arbiter_group_is_empty(group));
    }

    ucs_arbiter_t       m_arb;
    ucs_arbiter_group_t m_group1, m_group2;
    ucs_arbiter_elem_t  m_elem;
    bool                m_moved;
};

/* from the arbiter dispatch callback, reschedule the element on another group,
 * and remove it from current group
 */
UCS_TEST_F(test_arbiter_resched_from_dispatch, remove_and_resched) {

    m_moved = false;

    ucs_arbiter_group_push_elem(&m_group1, &m_elem);
    ucs_arbiter_group_schedule(&m_arb, &m_group1);

    /* group1 should be scheduled, group2 not */
    check_group_state(&m_group1, true);
    check_group_state(&m_group2, false);

    ucs_arbiter_dispatch(&m_arb, 1, dispatch_cb, this);

    /* the dispatch should deschedule group1 and schedule group2 instead */
    check_group_state(&m_group1, false);
    check_group_state(&m_group2, true);

    ucs_arbiter_group_purge(&m_arb, &m_group2, purge_cb, NULL);
}
