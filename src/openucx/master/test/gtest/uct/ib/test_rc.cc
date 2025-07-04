/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2016. ALL RIGHTS RESERVED.
* Copyright (C) ARM Ltd. 2016.All rights reserved.
* See file LICENSE for terms.
*/

#include "test_rc.h"
#include <uct/ib/rc/verbs/rc_verbs.h>


#define UCT_INSTANTIATE_RC_TEST_CASE(_test_case) \
    _UCT_INSTANTIATE_TEST_CASE(_test_case, rc_verbs) \
    _UCT_INSTANTIATE_TEST_CASE(_test_case, rc_mlx5)

#define UCT_INSTANTIATE_RC_DC_TEST_CASE(_test_case) \
    UCT_INSTANTIATE_RC_TEST_CASE(_test_case) \
    _UCT_INSTANTIATE_TEST_CASE(_test_case, dc_mlx5)

void test_rc::init()
{
    uct_test::init();

    m_e1 = uct_test::create_entity(0);
    m_entities.push_back(m_e1);

    check_skip_test();

    m_e2 = uct_test::create_entity(0);
    m_entities.push_back(m_e2);

    connect();
}

void test_rc::connect()
{
    m_e1->connect(0, *m_e2, 0);
    m_e2->connect(0, *m_e1, 0);

    uct_iface_set_am_handler(m_e1->iface(), 0, am_dummy_handler, NULL, 0);
    uct_iface_set_am_handler(m_e2->iface(), 0, am_dummy_handler, NULL, 0);
}

// Check that iface tx ops buffer and flush comp memory pool are moderated
// properly when we have communication ops + lots of flushes
void test_rc::test_iface_ops(int cq_len)
{
    entity *e = uct_test::create_entity(0);
    m_entities.push_back(e);
    e->connect(0, *m_e2, 0);

    mapped_buffer sendbuf(1024, 0ul, *e);
    mapped_buffer recvbuf(1024, 0ul, *m_e2);
    uct_completion_t comp;
    comp.count = cq_len * 512; // some big value to avoid func invocation
    comp.func  = NULL;

    UCS_TEST_GET_BUFFER_IOV(iov, iovcnt, sendbuf.ptr(), sendbuf.length(),
                            sendbuf.memh(), m_e1->iface_attr().cap.put.max_iov);
    // For _x transports several CQEs can be consumed per WQE, post less put zcopy
    // ops, so that flush would be sucessfull (otherwise flush will return
    // NO_RESOURCES and completion will not be added for it).
    for (int i = 0; i < cq_len / 5; i++) {
        ASSERT_UCS_OK_OR_INPROGRESS(uct_ep_put_zcopy(e->ep(0), iov, iovcnt,
                                                     recvbuf.addr(),
                                                     recvbuf.rkey(), &comp));

        // Create some stress on iface (flush mp):
        // post 10 flushes per every put.
        for (int j = 0; j < 10; j++) {
            ASSERT_UCS_OK_OR_INPROGRESS(uct_ep_flush(e->ep(0), 0, &comp));
        }
    }

    flush();
}

UCS_TEST_SKIP_COND_P(test_rc, stress_iface_ops,
                     !check_caps(UCT_IFACE_FLAG_PUT_ZCOPY)) {
    int cq_len = 16;

    if (UCS_OK != uct_config_modify(m_iface_config, "RC_TX_CQ_LEN",
                                    ucs::to_string(cq_len).c_str())) {
        UCS_TEST_ABORT("Error: cannot modify RC_TX_CQ_LEN");
    }

    test_iface_ops(cq_len);
}

UCS_TEST_P(test_rc, tx_cq_moderation) {
    unsigned tx_mod   = ucs_min(rc_iface(m_e1)->config.tx_moderation / 4, 8);
    int16_t init_rsc  = rc_ep(m_e1)->txqp.available;

    send_am_messages(m_e1, tx_mod, UCS_OK);

    int16_t rsc = rc_ep(m_e1)->txqp.available;

    EXPECT_LE(rsc, init_rsc);

    short_progress_loop(100);

    EXPECT_EQ(rsc, rc_ep(m_e1)->txqp.available);

    flush();

    EXPECT_EQ(init_rsc, rc_ep(m_e1)->txqp.available);
}

UCT_INSTANTIATE_RC_TEST_CASE(test_rc)


class test_rc_max_wr : public test_rc {
protected:
    virtual void init() {
        ucs_status_t status1, status2;
        status1 = uct_config_modify(m_iface_config, "TX_MAX_WR", "32");
        status2 = uct_config_modify(m_iface_config, "RC_TX_MAX_BB", "32");
        if (status1 != UCS_OK && status2 != UCS_OK) {
            UCS_TEST_ABORT("Error: cannot set rc max wr/bb");
        }
        test_rc::init();
    }
};

/* Check that max_wr stops from sending */
UCS_TEST_P(test_rc_max_wr, send_limit)
{
    /* first 32 messages should be OK */
    send_am_messages(m_e1, 32, UCS_OK);

    /* next message - should fail */
    send_am_messages(m_e1, 1, UCS_ERR_NO_RESOURCE);

    progress_loop();
    send_am_messages(m_e1, 1, UCS_OK);
}

UCT_INSTANTIATE_RC_TEST_CASE(test_rc_max_wr)

class test_rc_get_limit : public test_rc {
public:
    test_rc_get_limit() {
        m_num_get_ops = 8;
        modify_config("RC_TX_NUM_GET_OPS",
                      ucs::to_string(m_num_get_ops).c_str());

        m_max_get_zcopy = 4096;
        modify_config("RC_MAX_GET_ZCOPY",
                      ucs::to_string(m_max_get_zcopy).c_str());

        modify_config("RC_TX_QUEUE_LEN", "32");
        modify_config("RC_TM_ENABLE", "y", true);

        m_comp.count = 300000; // some big value to avoid func invocation
        m_comp.func  = NULL;
    }

    void init() {
#ifdef ENABLE_STATS
        stats_activate();
#endif
        test_rc::init();
    }

#ifdef ENABLE_STATS
    void cleanup() {
        uct_test::cleanup();
        stats_restore();
    }

    uint64_t get_no_reads_stat_counter(entity *e) {
        uct_rc_iface_t *iface = ucs_derived_of(e->iface(), uct_rc_iface_t);

        return UCS_STATS_GET_COUNTER(iface->stats, UCT_RC_IFACE_STAT_NO_READS);
    }
#endif

    unsigned reads_available(entity *e) {
        return rc_iface(e)->tx.reads_available;
    }

    void post_max_reads(entity *e, const mapped_buffer &sendbuf,
                        const mapped_buffer &recvbuf) {
        ucs_status_t status, status_exp;

        UCS_TEST_GET_BUFFER_IOV(iov, iovcnt, sendbuf.ptr(), sendbuf.length(),
                                sendbuf.memh(), e->iface_attr().cap.get.max_iov);

        for (unsigned i = 0; i <= m_num_get_ops; ++i) {
            if (i % 2) {
                status = uct_ep_get_zcopy(e->ep(0), iov, iovcnt, recvbuf.addr(),
                                          recvbuf.rkey(), &m_comp);
            } else {
                status = uct_ep_get_bcopy(e->ep(0), (uct_unpack_callback_t)memcpy,
                                          sendbuf.ptr(), sendbuf.length(),
                                          recvbuf.addr(), recvbuf.rkey(), &m_comp);
            }
            status_exp = (i == m_num_get_ops) ? UCS_ERR_NO_RESOURCE :
                                                UCS_INPROGRESS;
            EXPECT_EQ(status_exp, status);
        }

        EXPECT_EQ(0u, reads_available(e));
    }

    static size_t empty_pack_cb(void *dest, void *arg) {
        return 0ul;
    }

protected:
    unsigned         m_num_get_ops;
    unsigned         m_max_get_zcopy;
    uct_completion_t m_comp;
};

UCS_TEST_SKIP_COND_P(test_rc_get_limit, get_ops_limit,
                     !check_caps(UCT_IFACE_FLAG_GET_ZCOPY |
                                 UCT_IFACE_FLAG_GET_BCOPY))
{
    mapped_buffer sendbuf(1024, 0ul, *m_e1);
    mapped_buffer recvbuf(1024, 0ul, *m_e2);

    post_max_reads(m_e1, sendbuf, recvbuf);

#ifdef ENABLE_STATS
    EXPECT_GT(get_no_reads_stat_counter(m_e1), 0ul);
#endif

    // Check that it is possible to add to pending if get returns NO_RESOURCE
    // due to lack of get credits
    uct_pending_req_t pend_req;
    pend_req.func = NULL; // Make valgrind happy
    EXPECT_EQ(UCS_OK, uct_ep_pending_add(m_e1->ep(0), &pend_req, 0));
    uct_ep_pending_purge(m_e1->ep(0), NULL, NULL);

    flush();
    EXPECT_EQ(m_num_get_ops, reads_available(m_e1));
}

// Check that get function fails for messages bigger than MAX_GET_ZCOPY value
UCS_TEST_SKIP_COND_P(test_rc_get_limit, get_size_limit,
                     !check_caps(UCT_IFACE_FLAG_GET_ZCOPY))
{
    EXPECT_EQ(m_max_get_zcopy, m_e1->iface_attr().cap.get.max_zcopy);

    mapped_buffer buf(m_max_get_zcopy + 1, 0ul, *m_e1);

    UCS_TEST_GET_BUFFER_IOV(iov, iovcnt, buf.ptr(), buf.length(), buf.memh(),
                            m_e1->iface_attr().cap.get.max_iov);

    scoped_log_handler wrap_err(wrap_errors_logger);
    ucs_status_t status = uct_ep_get_zcopy(m_e1->ep(0), iov, iovcnt,
                                           buf.addr(), buf.rkey(), &m_comp);
    EXPECT_EQ(UCS_ERR_ILWALID_PARAM, status);

    flush();
    EXPECT_EQ(m_num_get_ops, reads_available(m_e1));
}

// Check that get size value is trimmed by the actual maximum IB msg size
UCS_TEST_SKIP_COND_P(test_rc_get_limit, ilwalid_get_size,
                     !check_caps(UCT_IFACE_FLAG_GET_ZCOPY))
{
    size_t max_ib_msg = uct_ib_iface_port_attr(&rc_iface(m_e1)->super)->max_msg_sz;

    modify_config("RC_MAX_GET_ZCOPY", ucs::to_string(max_ib_msg + 1).c_str());

    scoped_log_handler wrap_warn(hide_warns_logger);
    entity *e = uct_test::create_entity(0);
    m_entities.push_back(e);

    EXPECT_EQ(m_max_get_zcopy, m_e1->iface_attr().cap.get.max_zcopy);
}

// Check that gets resource counter is not affected/changed when the get
// function fails due to lack of some other resources.
UCS_TEST_SKIP_COND_P(test_rc_get_limit, post_get_no_res,
                     !check_caps(UCT_IFACE_FLAG_GET_ZCOPY |
                                 UCT_IFACE_FLAG_AM_BCOPY))
{
    unsigned max_get_ops = reads_available(m_e1);
    ucs_status_t status;

    do {
        status = send_am_message(m_e1, 0, 0);
    } while (status == UCS_OK);

    EXPECT_EQ(UCS_ERR_NO_RESOURCE, status);
    EXPECT_EQ(max_get_ops, reads_available(m_e1));

    mapped_buffer buf(1024, 0ul, *m_e1);
    UCS_TEST_GET_BUFFER_IOV(iov, iovcnt, buf.ptr(), buf.length(), buf.memh(),
                            m_e1->iface_attr().cap.get.max_iov);

    status = uct_ep_get_zcopy(m_e1->ep(0), iov, iovcnt, buf.addr(), buf.rkey(),
                              &m_comp);
    EXPECT_EQ(UCS_ERR_NO_RESOURCE, status);
    EXPECT_EQ(max_get_ops, reads_available(m_e1));
#ifdef ENABLE_STATS
    EXPECT_EQ(get_no_reads_stat_counter(m_e1), 0ul);
#endif

    flush();
}

UCS_TEST_SKIP_COND_P(test_rc_get_limit, check_rma_ops,
                     !check_caps(UCT_IFACE_FLAG_GET_ZCOPY |
                                 UCT_IFACE_FLAG_GET_BCOPY |
                                 UCT_IFACE_FLAG_PUT_SHORT |
                                 UCT_IFACE_FLAG_PUT_BCOPY |
                                 UCT_IFACE_FLAG_PUT_ZCOPY |
                                 UCT_IFACE_FLAG_AM_SHORT  |
                                 UCT_IFACE_FLAG_AM_BCOPY  |
                                 UCT_IFACE_FLAG_AM_ZCOPY))

{
    mapped_buffer sendbuf(1024, 0ul, *m_e1);
    mapped_buffer recvbuf(1024, 0ul, *m_e2);

    post_max_reads(m_e1, sendbuf, recvbuf);

    UCS_TEST_GET_BUFFER_IOV(iov, iovcnt, sendbuf.ptr(), 1, sendbuf.memh(), 1);
    uct_ep_h ep = m_e1->ep(0);

    EXPECT_EQ(UCS_ERR_NO_RESOURCE, uct_ep_put_short(ep, NULL, 0, 0, 0));
    EXPECT_EQ(UCS_ERR_NO_RESOURCE, uct_ep_put_bcopy(ep, NULL, NULL, 0, 0));
    EXPECT_EQ(UCS_ERR_NO_RESOURCE, uct_ep_put_zcopy(ep, iov, iovcnt, 0, 0,
                                                    NULL));

    if (check_atomics(UCS_BIT(UCT_ATOMIC_OP_ADD), FOP64)) {
        ASSERT_TRUE(check_atomics(UCS_BIT(UCT_ATOMIC_OP_ADD), OP64));
        ASSERT_TRUE(check_atomics(UCS_BIT(UCT_ATOMIC_OP_CSWAP), FOP64));
        EXPECT_EQ(UCS_ERR_NO_RESOURCE,
                  uct_ep_atomic64_post(ep, UCT_ATOMIC_OP_ADD, 0, 0, 0));
        EXPECT_EQ(UCS_ERR_NO_RESOURCE,
                  uct_ep_atomic64_fetch(ep, UCT_ATOMIC_OP_ADD, 0, NULL, 0, 0,
                                        NULL));
        EXPECT_EQ(UCS_ERR_NO_RESOURCE,
                  uct_ep_atomic_cswap64(ep, 0, 0, 0, 0, NULL, NULL));
    }

    if (check_atomics(UCS_BIT(UCT_ATOMIC_OP_ADD), FOP32)) {
        ASSERT_TRUE(check_atomics(UCS_BIT(UCT_ATOMIC_OP_ADD), OP32));
        ASSERT_TRUE(check_atomics(UCS_BIT(UCT_ATOMIC_OP_CSWAP), FOP32));
        EXPECT_EQ(UCS_ERR_NO_RESOURCE,
                  uct_ep_atomic32_post(ep, UCT_ATOMIC_OP_ADD, 0, 0, 0));
        EXPECT_EQ(UCS_ERR_NO_RESOURCE,
                  uct_ep_atomic32_fetch(ep, UCT_ATOMIC_OP_ADD, 0, NULL, 0, 0,
                                        NULL));
        EXPECT_EQ(UCS_ERR_NO_RESOURCE,
                  uct_ep_atomic_cswap32(ep, 0, 0, 0, 0, NULL, NULL));
    }

    EXPECT_UCS_OK(uct_ep_am_short(ep, 0, 0, NULL, 0));
    EXPECT_EQ(0l, uct_ep_am_bcopy(ep, 0, empty_pack_cb, NULL, 0));
    EXPECT_FALSE(UCS_STATUS_IS_ERR(uct_ep_am_zcopy(ep, 0, NULL, 0, iov, iovcnt,
                                                   0, NULL)));

    if (check_caps(UCT_IFACE_FLAG_TAG_EAGER_BCOPY)) {
        // we do not have partial tag offload support
        ASSERT_TRUE(check_caps(UCT_IFACE_FLAG_TAG_EAGER_SHORT |
                               UCT_IFACE_FLAG_TAG_EAGER_ZCOPY |
                               UCT_IFACE_FLAG_TAG_RNDV_ZCOPY));

        EXPECT_UCS_OK(uct_ep_tag_eager_short(ep, 0ul, NULL, 0));
        EXPECT_EQ(0l, uct_ep_tag_eager_bcopy(ep, 0ul, 0ul, empty_pack_cb,
                                             NULL, 0));
        EXPECT_FALSE(UCS_STATUS_IS_ERR(uct_ep_tag_eager_zcopy(ep, 0ul, 0ul, iov,
                                                              iovcnt, 0u,
                                                              NULL)));
        void *rndv_op = uct_ep_tag_rndv_zcopy(ep, 0ul, NULL, 0u, iov, iovcnt,
                                              0u, NULL);
        EXPECT_FALSE(UCS_PTR_IS_ERR(rndv_op));
        EXPECT_UCS_OK(uct_ep_tag_rndv_cancel(ep, rndv_op));
        EXPECT_UCS_OK(uct_ep_tag_rndv_request(ep, 0ul, NULL, 0u, 0u));
    }

    flush();
    EXPECT_EQ(m_num_get_ops, reads_available(m_e1));
}

UCT_INSTANTIATE_RC_DC_TEST_CASE(test_rc_get_limit)

uint32_t test_rc_flow_control::m_am_rx_count = 0;

void test_rc_flow_control::init()
{
    /* For correct testing FC needs to be initialized during interface creation */
    if (UCS_OK != uct_config_modify(m_iface_config, "RC_FC_ENABLE", "y")) {
        UCS_TEST_ABORT("Error: cannot enable flow control");
    }
    test_rc::init();

    ucs_assert(rc_iface(m_e1)->config.fc_enabled);
    ucs_assert(rc_iface(m_e2)->config.fc_enabled);

    uct_iface_set_am_handler(m_e1->iface(), FLUSH_AM_ID, am_handler, NULL, 0);
    uct_iface_set_am_handler(m_e2->iface(), FLUSH_AM_ID, am_handler, NULL, 0);

}

void test_rc_flow_control::cleanup()
{
    /* Restore FC state to enabled, so iface cleanup will destroy the grant mpool */
    rc_iface(m_e1)->config.fc_enabled = 1;
    rc_iface(m_e2)->config.fc_enabled = 1;
    test_rc::cleanup();
}

void test_rc_flow_control::send_am_and_flush(entity *e, int num_msg)
{
    m_am_rx_count = 0;

    send_am_messages(e, num_msg - 1, UCS_OK);
    send_am_messages(e, 1, UCS_OK, FLUSH_AM_ID); /* send last msg with FLUSH id */
    wait_for_flag(&m_am_rx_count);
    EXPECT_EQ(m_am_rx_count, 1ul);
}

void test_rc_flow_control::validate_grant(entity *e)
{
    wait_for_flag(&get_fc_ptr(e)->fc_wnd);
    EXPECT_GT(get_fc_ptr(e)->fc_wnd, 0);
}

/* Check that FC window works as expected:
 * - If FC enabled, only 'wnd' messages can be sent in a row
 * - If FC is disabled 'wnd' does not limit senders flow  */
void test_rc_flow_control::test_general(int wnd, int soft_thresh,
                                        int hard_thresh, bool is_fc_enabled)
{
    set_fc_attributes(m_e1, is_fc_enabled, wnd, soft_thresh, hard_thresh);

    send_am_messages(m_e1, wnd, UCS_OK);
    send_am_messages(m_e1, 1, is_fc_enabled ?  UCS_ERR_NO_RESOURCE : UCS_OK);

    validate_grant(m_e1);
    send_am_messages(m_e1, 1, UCS_OK);

    if (!is_fc_enabled) {
        /* Make valgrind happy, need to enable FC for proper cleanup */
        set_fc_attributes(m_e1, true, wnd, wnd, 1);
    }
    flush();
}

void test_rc_flow_control::test_pending_grant(int wnd)
{
    /* Block send capabilities of m_e2 for fc grant to be
     * added to the pending queue. */
    disable_entity(m_e2);
    set_fc_attributes(m_e1, true, wnd, wnd, 1);

    send_am_and_flush(m_e1, wnd);

    /* Now m_e1 should be blocked by FC window and FC grant
     * should be in pending queue of m_e2. */
    send_am_messages(m_e1, 1, UCS_ERR_NO_RESOURCE);
    EXPECT_LE(get_fc_ptr(m_e1)->fc_wnd, 0);

    /* Enable send capabilities of m_e2 and send AM message
     * to force pending queue dispatch */
    enable_entity(m_e2);
    set_tx_moderation(m_e2, 0);
    send_am_messages(m_e2, 1, UCS_OK);

    /* Check that m_e1 got grant */
    validate_grant(m_e1);
    send_am_messages(m_e1, 1, UCS_OK);
}

void test_rc_flow_control::test_flush_fc_disabled()
{
    set_fc_disabled(m_e1);
    ucs_status_t status;

    /* If FC is disabled, wnd=0 should not prevent the flush */
    get_fc_ptr(m_e1)->fc_wnd = 0;
    status = uct_ep_flush(m_e1->ep(0), 0, NULL);
    EXPECT_EQ(UCS_OK, status);

    /* send active message should be OK */
    get_fc_ptr(m_e1)->fc_wnd = 1;
    send_am_messages(m_e1, 1, UCS_OK);
    EXPECT_EQ(0, get_fc_ptr(m_e1)->fc_wnd);

    /* flush must have resources */
    status = uct_ep_flush(m_e1->ep(0), 0, NULL);
    EXPECT_FALSE(UCS_STATUS_IS_ERR(status)) << ucs_status_string(status);
}

void test_rc_flow_control::test_pending_purge(int wnd, int num_pend_sends)
{
    pending_send_request_t reqs[num_pend_sends];

    disable_entity(m_e2);
    set_fc_attributes(m_e1, true, wnd, wnd, 1);

    send_am_and_flush(m_e1, wnd);

    /* Now m2 ep should have FC grant message in the pending queue.
     * Add some user pending requests as well */
    for (int i = 0; i < num_pend_sends; i++) {
        reqs[i].uct.func    = NULL; /* make valgrind happy */
        reqs[i].purge_count = 0;
        EXPECT_EQ(uct_ep_pending_add(m_e2->ep(0), &reqs[i].uct, 0), UCS_OK);
    }
    uct_ep_pending_purge(m_e2->ep(0), purge_cb, NULL);

    for (int i = 0; i < num_pend_sends; i++) {
        EXPECT_EQ(1, reqs[i].purge_count);
    }
}


/* Check that FC window works as expected */
UCS_TEST_P(test_rc_flow_control, general_enabled)
{
    test_general(8, 4, 2, true);
}

UCS_TEST_P(test_rc_flow_control, general_disabled)
{
    test_general(8, 4, 2, false);
}

/* Test the scenario when ep is being destroyed while there is
 * FC grant message in the pending queue */
UCS_TEST_P(test_rc_flow_control, pending_only_fc)
{
    int wnd = 2;

    disable_entity(m_e2);
    set_fc_attributes(m_e1, true, wnd, wnd, 1);

    send_am_and_flush(m_e1, wnd);

    m_e2->destroy_ep(0);
    ASSERT_TRUE(ucs_arbiter_is_empty(&rc_iface(m_e2)->tx.arbiter));
}

/* Check that user callback passed to uct_ep_pending_purge is not
 * ilwoked for FC grant message */
UCS_TEST_P(test_rc_flow_control, pending_purge)
{
    test_pending_purge(2, 5);
}

UCS_TEST_P(test_rc_flow_control, pending_grant)
{
    test_pending_grant(5);
}

UCS_TEST_P(test_rc_flow_control, fc_disabled_flush)
{
    test_flush_fc_disabled();
}

UCT_INSTANTIATE_RC_TEST_CASE(test_rc_flow_control)


#ifdef ENABLE_STATS

void test_rc_flow_control_stats::test_general(int wnd, int soft_thresh,
                                              int hard_thresh)
{
    uint64_t v;

    set_fc_attributes(m_e1, true, wnd, soft_thresh, hard_thresh);

    send_am_messages(m_e1, wnd, UCS_OK);
    send_am_messages(m_e1, 1, UCS_ERR_NO_RESOURCE);

    v = UCS_STATS_GET_COUNTER(get_fc_ptr(m_e1)->stats, UCT_RC_FC_STAT_NO_CRED);
    EXPECT_EQ(1ul, v);

    validate_grant(m_e1);
    send_am_messages(m_e1, 1, UCS_OK);

    v = UCS_STATS_GET_COUNTER(get_fc_ptr(m_e1)->stats, UCT_RC_FC_STAT_TX_HARD_REQ);
    EXPECT_EQ(1ul, v);

    v = UCS_STATS_GET_COUNTER(get_fc_ptr(m_e1)->stats, UCT_RC_FC_STAT_RX_PURE_GRANT);
    EXPECT_EQ(1ul, v);
    flush();
}


UCS_TEST_P(test_rc_flow_control_stats, general)
{
    test_general(5, 2, 1);
}

UCS_TEST_P(test_rc_flow_control_stats, soft_request)
{
    uint64_t v;
    int wnd = 8;
    int s_thresh = 4;
    int h_thresh = 1;

    set_fc_attributes(m_e1, true, wnd, s_thresh, h_thresh);
    send_am_and_flush(m_e1, wnd - (s_thresh - 1));

    v = UCS_STATS_GET_COUNTER(get_fc_ptr(m_e1)->stats, UCT_RC_FC_STAT_TX_SOFT_REQ);
    EXPECT_EQ(1ul, v);
    v = UCS_STATS_GET_COUNTER(get_fc_ptr(m_e2)->stats, UCT_RC_FC_STAT_RX_SOFT_REQ);
    EXPECT_EQ(1ul, v);

    send_am_and_flush(m_e2, wnd - (s_thresh - 1));
    v = UCS_STATS_GET_COUNTER(get_fc_ptr(m_e1)->stats, UCT_RC_FC_STAT_RX_GRANT);
    EXPECT_EQ(1ul, v);
    v = UCS_STATS_GET_COUNTER(get_fc_ptr(m_e2)->stats, UCT_RC_FC_STAT_TX_GRANT);
    EXPECT_EQ(1ul, v);
}

UCT_INSTANTIATE_RC_TEST_CASE(test_rc_flow_control_stats)

#endif

#if HAVE_MLX5_HW
extern "C" {
#include <uct/ib/rc/accel/rc_mlx5_common.h>
}
#endif

test_uct_iface_attrs::attr_map_t test_rc_iface_attrs::get_num_iov() {
    if (has_transport("rc_mlx5")) {
        return get_num_iov_mlx5_common(0ul);
    } else {
        EXPECT_TRUE(has_transport("rc_verbs"));
        m_e->connect(0, *m_e, 0);
        uct_rc_verbs_ep_t *ep = ucs_derived_of(m_e->ep(0), uct_rc_verbs_ep_t);
        uint32_t max_sge;
        ASSERT_UCS_OK(uct_ib_qp_max_send_sge(ep->qp, &max_sge));

        attr_map_t iov_map;
        iov_map["put"] = iov_map["get"] = max_sge;
        iov_map["am"]  = max_sge - 1; // 1 iov reserved for am header
        return iov_map;
    }
}

test_uct_iface_attrs::attr_map_t
test_rc_iface_attrs::get_num_iov_mlx5_common(size_t av_size)
{
    attr_map_t iov_map;

#if HAVE_MLX5_HW
    // For RMA iovs can use all WQE space, remainig from control and
    // remote address segments (and AV if relevant)
    size_t rma_iov = (UCT_IB_MLX5_MAX_SEND_WQE_SIZE -
                      (sizeof(struct mlx5_wqe_raddr_seg) +
                       sizeof(struct mlx5_wqe_ctrl_seg) + av_size)) /
                     sizeof(struct mlx5_wqe_data_seg);

    iov_map["put"] = iov_map["get"] = rma_iov;

    // For am zcopy just small constant number of iovs is allowed
    // (to preserve some inline space for AM zcopy header)
    iov_map["am"]  = UCT_IB_MLX5_AM_ZCOPY_MAX_IOV;

#if IBV_HW_TM
    if (UCT_RC_MLX5_TM_ENABLED(ucs_derived_of(m_e->iface(),
                                              uct_rc_mlx5_iface_common_t))) {
        // For TAG eager zcopy iovs can use all WQE space, remainig from control
        // segment, TMH header (+ inline data segment) and AV (if relevant)
        iov_map["tag"] = (UCT_IB_MLX5_MAX_SEND_WQE_SIZE -
                          (sizeof(struct mlx5_wqe_ctrl_seg) +
                           sizeof(struct mlx5_wqe_inl_data_seg) +
                           sizeof(struct ibv_tmh) + av_size)) /
                         sizeof(struct mlx5_wqe_data_seg);
    }
#endif // IBV_HW_TM
#endif // HAVE_MLX5_HW

    return iov_map;
}

UCS_TEST_P(test_rc_iface_attrs, iface_attrs)
{
    basic_iov_test();
}

UCT_INSTANTIATE_RC_TEST_CASE(test_rc_iface_attrs)

