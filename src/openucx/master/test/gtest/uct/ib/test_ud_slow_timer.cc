/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "ud_base.h"

#include <uct/uct_test.h>

extern "C" {
#include <ucs/time/time.h>
#include <ucs/datastruct/queue.h>
#include <ucs/datastruct/ptr_array.h>
#include <uct/ib/ud/base/ud_ep.h>
#include <uct/ib/ud/base/ud_iface.h>
}


class test_ud_slow_timer : public ud_base_test {
public:
    /* ack while doing retransmit */
    static int packet_count, rx_limit;
    static ucs_status_t rx_npackets(uct_ud_ep_t *ep, uct_ud_neth_t *neth)
    {
        if (packet_count++ < rx_limit) {
            return UCS_OK;
        }
        else { 
            return UCS_ERR_ILWALID_PARAM;
        }
    }
    /* test slow timer and restransmit */
    static int tick_count;

    static ucs_status_t tick_counter(uct_ud_ep_t *ep, uct_ud_neth_t *neth)
    {
        uct_ud_iface_t *iface = ucs_derived_of(ep->super.super.iface,
                                               uct_ud_iface_t);

        /* hack to disable retransmit */
        ep->tx.send_time = ucs_twheel_get_time(&iface->async.slow_timer);
        tick_count++;
        return UCS_OK;
    }

    static ucs_status_t drop_packet(uct_ud_ep_t *ep, uct_ud_neth_t *neth)
    {
        return UCS_ERR_ILWALID_PARAM;
    }

    void wait_for_rx_sn(unsigned sn)
    {
        ucs_time_t deadline = ucs_get_time() +
                              ucs_time_from_sec(10) * ucs::test_time_multiplier();
        while ((ucs_get_time() < deadline) && (ucs_frag_list_sn(&ep(m_e2)->rx.ooo_pkts) < sn)) {
            usleep(1000);
        }
    }

    void wait_for_ep_destroyed(uct_ud_iface_t *iface, uint32_t ep_idx)
    {
        ucs_time_t deadline = ucs_get_time() +
                              ucs_time_from_sec(60) * ucs::test_time_multiplier();
        void *ud_ep_tmp GTEST_ATTRIBUTE_UNUSED_;

        while ((ucs_get_time() < deadline) &&
               ucs_ptr_array_lookup(&iface->eps, ep_idx, ud_ep_tmp)) {
            usleep(1000);
        }
    }
};

int test_ud_slow_timer::rx_limit = 10;
int test_ud_slow_timer::packet_count = 0;
int test_ud_slow_timer::tick_count = 0;


/* single packet received without progress */
UCS_TEST_SKIP_COND_P(test_ud_slow_timer, tx1,
                     !check_caps(UCT_IFACE_FLAG_PUT_SHORT)) {
    connect();
    EXPECT_UCS_OK(tx(m_e1));
    wait_for_rx_sn(1);
    EXPECT_EQ(2, ep(m_e1)->tx.psn);
    EXPECT_EQ(1, ucs_frag_list_sn(&ep(m_e2)->rx.ooo_pkts));
}

/* multiple packets received without progress */
UCS_TEST_SKIP_COND_P(test_ud_slow_timer, txn,
                     !check_caps(UCT_IFACE_FLAG_PUT_SHORT)) {
    unsigned i, N = 42;

    connect();
    set_tx_win(m_e1, 1024);
    for (i = 0; i < N; i++) {
        EXPECT_UCS_OK(tx(m_e1));
    }
    wait_for_rx_sn(N);
    EXPECT_EQ(N+1, ep(m_e1)->tx.psn);
    EXPECT_EQ(N, ucs_frag_list_sn(&ep(m_e2)->rx.ooo_pkts));
}

UCS_TEST_P(test_ud_slow_timer, ep_destroy, "UD_TIMEOUT=1s") {
    void *ud_ep_tmp GTEST_ATTRIBUTE_UNUSED_;
    connect();

    uct_ud_ep_t    *ud_ep = ep(m_e1);
    uct_ud_iface_t *iface = ucs_derived_of(ud_ep->super.super.iface,
                                           uct_ud_iface_t);
    uint32_t       ep_idx = ud_ep->ep_id;
    EXPECT_TRUE(ucs_ptr_array_lookup(&iface->eps, ep_idx, ud_ep_tmp));

    m_e1->destroy_eps();
    wait_for_ep_destroyed(iface, ep_idx);
    EXPECT_FALSE(ucs_ptr_array_lookup(&iface->eps, ep_idx, ud_ep_tmp));
}

UCS_TEST_P(test_ud_slow_timer, backoff_config) {
    /* check minimum allowed value */
    ASSERT_UCS_OK(uct_config_modify(m_iface_config,
                  "UD_SLOW_TIMER_BACKOFF",
                  ucs::to_string(UCT_UD_MIN_TIMER_TIMER_BACKOFF).c_str()));
    entity *e = uct_test::create_entity(0);
    m_entities.push_back(e);

    {
        /* iface creation should fail with back off value less than
         * UCT_UD_MIN_TIMER_TIMER_BACKOFF */
        ASSERT_UCS_OK(uct_config_modify(m_iface_config,
                      "UD_SLOW_TIMER_BACKOFF",
                      ucs::to_string(UCT_UD_MIN_TIMER_TIMER_BACKOFF - 0.1).c_str()));
        scoped_log_handler wrap_err(wrap_errors_logger);
        uct_iface_h iface;
        ucs_status_t status = uct_iface_open(e->md(), e->worker(),
                                             &e->iface_params(),
                                             m_iface_config, &iface);
        EXPECT_EQ(UCS_ERR_ILWALID_PARAM, status);
        EXPECT_EQ(NULL, iface);
    }
}

#if UCT_UD_EP_DEBUG_HOOKS
/* no traffic - no ticks */
UCS_TEST_P(test_ud_slow_timer, tick1) {
    connect();
    tick_count = 0;
    ep(m_e1)->timer_hook = tick_counter;
    twait(500);
    EXPECT_EQ(0, tick_count);
}

/* ticks while tx  window is not empty */
UCS_TEST_SKIP_COND_P(test_ud_slow_timer, tick2,
                     !check_caps(UCT_IFACE_FLAG_PUT_SHORT)) {
    connect();
    tick_count = 0;
    ep(m_e1)->timer_hook = tick_counter;
    EXPECT_UCS_OK(tx(m_e1));
    twait(500);
    EXPECT_LT(0, tick_count);
}

/* retransmit one packet */

UCS_TEST_SKIP_COND_P(test_ud_slow_timer, retransmit1,
                     !check_caps(UCT_IFACE_FLAG_PUT_SHORT)) {
    connect();
    ep(m_e2)->rx.rx_hook = drop_packet;
    EXPECT_UCS_OK(tx(m_e1));
    short_progress_loop();
    EXPECT_EQ(0, ucs_frag_list_sn(&ep(m_e2)->rx.ooo_pkts));
    ep(m_e2)->rx.rx_hook = uct_ud_ep_null_hook;
    EXPECT_EQ(2, ep(m_e1)->tx.psn);
    wait_for_rx_sn(1);
    EXPECT_EQ(2, ep(m_e1)->tx.psn);
    EXPECT_EQ(1, ucs_frag_list_sn(&ep(m_e2)->rx.ooo_pkts));
}

/* retransmit many packets */
UCS_TEST_SKIP_COND_P(test_ud_slow_timer, retransmitn,
                     !check_caps(UCT_IFACE_FLAG_PUT_SHORT)) {

    unsigned i, N = 42;

    connect();
    set_tx_win(m_e1, 1024);
    ep(m_e2)->rx.rx_hook = drop_packet;
    for (i = 0; i < N; i++) {
        EXPECT_UCS_OK(tx(m_e1));
    }
    short_progress_loop();
    EXPECT_EQ(0, ucs_frag_list_sn(&ep(m_e2)->rx.ooo_pkts));
    ep(m_e2)->rx.rx_hook = uct_ud_ep_null_hook;
    EXPECT_EQ(N+1, ep(m_e1)->tx.psn);
    wait_for_rx_sn(N);
    EXPECT_EQ(N+1, ep(m_e1)->tx.psn);
    EXPECT_EQ(N, ucs_frag_list_sn(&ep(m_e2)->rx.ooo_pkts));
}


UCS_TEST_SKIP_COND_P(test_ud_slow_timer, partial_drop,
                     !check_caps(UCT_IFACE_FLAG_PUT_SHORT)) {

    unsigned i, N = 24;
    int orig_avail;

    connect();
    set_tx_win(m_e1, 1024);
    packet_count = 0;
    rx_limit = 10;
    ep(m_e2)->rx.rx_hook = rx_npackets;
    for (i = 0; i < N; i++) {
        EXPECT_UCS_OK(tx(m_e1));
    }
    short_progress_loop();
    EXPECT_EQ(rx_limit, ucs_frag_list_sn(&ep(m_e2)->rx.ooo_pkts));
    ep(m_e2)->rx.rx_hook = uct_ud_ep_null_hook;
    EXPECT_EQ(N+1, ep(m_e1)->tx.psn);
    orig_avail = iface(m_e1)->tx.available;
    /* allow only 6 outgoing packets. It will allow to get ack
     * from receiver
     */
    iface(m_e1)->tx.available = 6;
    twait(500);
    iface(m_e1)->tx.available = orig_avail-6;
    short_progress_loop();
    
    EXPECT_EQ(N+1, ep(m_e1)->tx.psn);
    wait_for_rx_sn(N);
    EXPECT_EQ(N, ucs_frag_list_sn(&ep(m_e2)->rx.ooo_pkts));
}
#endif

UCT_INSTANTIATE_UD_TEST_CASE(test_ud_slow_timer)
