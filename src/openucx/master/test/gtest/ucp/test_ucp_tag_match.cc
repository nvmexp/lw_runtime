/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "test_ucp_tag.h"

#include <common/test_helpers.h>
extern "C" {
#include <ucp/core/ucp_request.h>
#include <ucp/core/ucp_types.h>
}

using namespace ucs; /* For vector<char> serialization */


class test_ucp_tag_match : public test_ucp_tag {
public:
    test_ucp_tag_match() {
        // TODO: test offload and offload MP as different variants
        enable_tag_mp_offload();
        if (RUNNING_ON_VALGRIND) {
            m_elw.push_back(new ucs::scoped_setelw("UCX_RC_TM_SEG_SIZE", "8k"));
            m_elw.push_back(new ucs::scoped_setelw("UCX_TCP_RX_SEG_SIZE", "8k"));
        }
    }

    virtual void init()
    {
        modify_config("TM_THRESH",  "1");

        test_ucp_tag::init();
        ucp_test_param param = GetParam();
    }

    static std::vector<ucp_test_param> enum_test_params(const ucp_params_t& ctx_params,
                                                        const std::string& name,
                                                        const std::string& test_case_name,
                                                        const std::string& tls)
    {
        std::vector<ucp_test_param> result;
        generate_test_params_variant(ctx_params, name, test_case_name, tls,
                                     RECV_REQ_INTERNAL, result);
        generate_test_params_variant(ctx_params, name, test_case_name, tls,
                                     RECV_REQ_EXTERNAL, result);
        return result;
    }

    virtual bool is_external_request()
    {
        return GetParam().variant == RECV_REQ_EXTERNAL;
    }

protected:
    static void recv_callback_release_req(void *request, ucs_status_t status,
                                          ucp_tag_recv_info_t *info)
    {
        ucp_request_free(request);
        m_req_status = status;
    }

    static ucs_status_t m_req_status;
};

ucs_status_t test_ucp_tag_match::m_req_status = UCS_OK;


UCS_TEST_P(test_ucp_tag_match, send_recv_unexp) {
    ucp_tag_recv_info_t info;
    ucs_status_t        status;

    uint64_t send_data = 0xdeadbeefdeadbeef;
    uint64_t recv_data = 0;

    send_b(&send_data, sizeof(send_data), DATATYPE, 0x111337);

    short_progress_loop(); /* Receive messages as unexpected */

    status = recv_b(&recv_data, sizeof(recv_data), DATATYPE, 0x1337, 0xffff, &info);
    ASSERT_UCS_OK(status);

    EXPECT_EQ(sizeof(send_data),   info.length);
    EXPECT_EQ((ucp_tag_t)0x111337, info.sender_tag);
    EXPECT_EQ(send_data, recv_data);
}

UCS_TEST_SKIP_COND_P(test_ucp_tag_match, send_recv_unexp_rqfree,
                     /* request free cannot be used for external requests */
                     (GetParam().variant == RECV_REQ_EXTERNAL)) {
    request *my_recv_req;
    uint64_t send_data = 0xdeadbeefdeadbeef;
    uint64_t recv_data = 0;

    my_recv_req = recv_nb(&recv_data, sizeof(recv_data), DATATYPE, 0x1337, 0xffff);
    ASSERT_TRUE(!UCS_PTR_IS_ERR(my_recv_req));

    request_free(my_recv_req);

    send_b(&send_data, sizeof(send_data), DATATYPE, 0x1337);

    wait_for_flag(&recv_data);
    EXPECT_EQ(send_data, recv_data);
}

UCS_TEST_P(test_ucp_tag_match, send_recv_exp_medium) {
    static const size_t size = 50000;
    request *my_recv_req;

    std::vector<char> sendbuf(size, 0);
    std::vector<char> recvbuf(size, 0);

    ucs::fill_random(sendbuf);

    my_recv_req = recv_nb(&recvbuf[0], recvbuf.size(), DATATYPE, 0x1337, 0xffff);
    ASSERT_TRUE(!UCS_PTR_IS_ERR(my_recv_req));
    ASSERT_TRUE(my_recv_req != NULL); /* Couldn't be completed because didn't send yet */

    send_b(&sendbuf[0], sendbuf.size(), DATATYPE, 0x111337);

    wait(my_recv_req);

    EXPECT_EQ(sendbuf.size(),      my_recv_req->info.length);
    EXPECT_EQ((ucp_tag_t)0x111337, my_recv_req->info.sender_tag);
    EXPECT_EQ(sendbuf, recvbuf);
    request_release(my_recv_req);
}

UCS_TEST_P(test_ucp_tag_match, send2_nb_recv_exp_medium) {
    static const size_t size = 50000;
    request *my_recv_req;

    std::vector<char> sendbuf(size, 0);
    std::vector<char> recvbuf(size, 0);

    /* 1st send */

    my_recv_req = recv_nb(&recvbuf[0], recvbuf.size(), DATATYPE, 0x1337, 0xffff);
    ASSERT_TRUE(!UCS_PTR_IS_ERR(my_recv_req));
    ASSERT_TRUE(my_recv_req != NULL); /* Couldn't be completed because didn't send yet */

    send_b(&sendbuf[0], sendbuf.size(), DATATYPE, 0x111337);

    wait(my_recv_req);
    request_release(my_recv_req);

    /* 2nd send */

    ucs::fill_random(sendbuf);

    my_recv_req = recv_nb(&recvbuf[0], recvbuf.size(), DATATYPE, 0x1337, 0xffff);
    ASSERT_TRUE(!UCS_PTR_IS_ERR(my_recv_req));
    ASSERT_TRUE(my_recv_req != NULL); /* Couldn't be completed because didn't send yet */

    request *my_send_req;
    my_send_req = send_nb(&sendbuf[0], sendbuf.size(), DATATYPE, 0x111337);
    ASSERT_TRUE(!UCS_PTR_IS_ERR(my_send_req));

    wait(my_recv_req);

    EXPECT_EQ(sendbuf.size(),      my_recv_req->info.length);
    EXPECT_EQ((ucp_tag_t)0x111337, my_recv_req->info.sender_tag);
    EXPECT_EQ(sendbuf, recvbuf);

    short_progress_loop();

    if (my_send_req != NULL) {
        EXPECT_TRUE(my_send_req->completed);
        EXPECT_EQ(UCS_OK, my_send_req->status);
        request_release(my_send_req);
    }
    request_release(my_recv_req);
}

UCS_TEST_P(test_ucp_tag_match, send2_nb_recv_medium_wildcard, "RNDV_THRESH=inf") {
    static const size_t size = 3000000;

    entity &sender2 = sender();
    create_entity(true);
    sender().connect(&receiver(), get_ep_params());

    for (int is_exp = 0; is_exp <= 1; ++is_exp) {

        UCS_TEST_MESSAGE << "Testing " << (is_exp ? "" : "un") << "expected mode, size " << size;

        std::vector<char> sendbuf1(size, 0);
        std::vector<char> sendbuf2(size, 0);
        std::vector<char> recvbuf1(size, 0);
        std::vector<char> recvbuf2(size, 0);

        ucs::fill_random(sendbuf1);
        ucs::fill_random(sendbuf2);

        /* Two sends with different tags */

        request *sreq1, *sreq2;
        sreq1 = (request*)ucp_tag_send_nb(sender().ep(), &sendbuf1[0], sendbuf1.size(),
                                          DATATYPE, 1, send_callback);
        ASSERT_TRUE(!UCS_PTR_IS_ERR(sreq1));

        sreq2 = (request*)ucp_tag_send_nb(sender2.ep(), &sendbuf2[0], sendbuf2.size(),
                                          DATATYPE, 2, send_callback);
        ASSERT_TRUE(!UCS_PTR_IS_ERR(sreq2));


        /* In unexpected mode, we progress all to put the messages on the
         *  unexpected queue
         */
        if (!is_exp) {
            short_progress_loop();
        }

        /* Two receives with any tag */

        request *rreq1, *rreq2;

        rreq1 = recv_nb(&recvbuf1[0], recvbuf1.size(), DATATYPE, 0, 0);
        ASSERT_TRUE(!UCS_PTR_IS_ERR(rreq1));

        rreq2 = recv_nb(&recvbuf2[0], recvbuf2.size(), DATATYPE, 0, 0);
        ASSERT_TRUE(!UCS_PTR_IS_ERR(rreq2));


        /* Wait for receives */
        wait(rreq1);
        wait(rreq2);

        short_progress_loop();

        /* Release sends */
        if (sreq1 != NULL) {
            wait(sreq1);
            EXPECT_TRUE(sreq1->completed);
            request_release(sreq1);
        }
        if (sreq2 != NULL) {
            wait(sreq2);
            EXPECT_TRUE(sreq2->completed);
            request_release(sreq2);
        }

        /* Receives should be completed with correct length */
        ASSERT_TRUE(rreq1->completed);
        ASSERT_TRUE(rreq2->completed);

        EXPECT_EQ(size, rreq1->info.length);
        EXPECT_EQ(size, rreq2->info.length);

        /* The order may be any, but the messages have to be received correctly */
        if (rreq1->info.sender_tag == 1u) {
            ASSERT_EQ(2u, rreq2->info.sender_tag);
            EXPECT_EQ(sendbuf1, recvbuf1);
            EXPECT_EQ(sendbuf2, recvbuf2);
        } else {
            ASSERT_EQ(2u, rreq1->info.sender_tag);
            ASSERT_EQ(1u, rreq2->info.sender_tag);
            EXPECT_EQ(sendbuf2, recvbuf1);
            EXPECT_EQ(sendbuf1, recvbuf2);
        }

        request_release(rreq1);
        request_release(rreq2);
    }
}

UCS_TEST_P(test_ucp_tag_match, send_recv_nb_partial_exp_medium) {
    static const size_t size = 50000;

    std::vector<char> sendbuf(size, 0);
    std::vector<char> recvbuf(size, 0);

    ucs::fill_random(sendbuf);

    request *my_recv_req;
    my_recv_req = recv_nb(&recvbuf[0], recvbuf.size(), DATATYPE, 0x1337, 0xffff);
    ASSERT_TRUE(!UCS_PTR_IS_ERR(my_recv_req));

    send_b(&sendbuf[0], sendbuf.size(), DATATYPE, 0x111337);

    usleep(1000);
    progress();

    wait(my_recv_req);

    EXPECT_EQ(sendbuf.size(),      my_recv_req->info.length);
    EXPECT_EQ((ucp_tag_t)0x111337, my_recv_req->info.sender_tag);
    EXPECT_EQ(sendbuf, recvbuf);

    request_release(my_recv_req);
}

UCS_TEST_P(test_ucp_tag_match, send_nb_recv_unexp) {
    ucp_tag_recv_info_t info;
    ucs_status_t        status;

    uint64_t send_data = 0xdeadbeefdeadbeef;
    uint64_t recv_data = 0;

    request *my_send_req;
    my_send_req = send_nb(&send_data, sizeof(send_data), DATATYPE, 0x111337);
    ASSERT_TRUE(!UCS_PTR_IS_ERR(my_send_req));

    ucp_worker_progress(receiver().worker());

    status = recv_b(&recv_data, sizeof(recv_data), DATATYPE, 0x1337, 0xffff, &info);
    ASSERT_UCS_OK(status);

    EXPECT_EQ(sizeof(send_data),   info.length);
    EXPECT_EQ((ucp_tag_t)0x111337, info.sender_tag);
    EXPECT_EQ(send_data, recv_data);

    if (my_send_req != NULL) {
        EXPECT_TRUE(my_send_req->completed);
        EXPECT_EQ(UCS_OK, my_send_req->status);
        request_release(my_send_req);
    }
}

UCS_TEST_P(test_ucp_tag_match, send_recv_cb_release) {

    uint64_t send_data = 0xdeadbeefdeadbeef;

    send_b(&send_data, sizeof(send_data), DATATYPE, 0x111337);

    short_progress_loop(); /* Receive messages as unexpected */

    m_req_status = UCS_INPROGRESS;

    uint64_t recv_data;
    request *recv_req = (request*)ucp_tag_recv_nb(receiver().worker(), &recv_data,
                                                  sizeof(recv_data), DATATYPE, 0, 0,
                                                  recv_callback_release_req);
    if (UCS_PTR_IS_ERR(recv_req)) {
        ASSERT_UCS_OK(UCS_PTR_STATUS(recv_req));
    } else if (recv_req == NULL) {
        UCS_TEST_ABORT("ucp_tag_recv_nb returned NULL");
    } else {
        /* request would be completed and released by the callback */
        while (m_req_status == UCS_INPROGRESS) {
            progress();
        }
        ASSERT_UCS_OK(m_req_status);
    }
}

UCS_TEST_P(test_ucp_tag_match, send_recv_truncated) {
    ucp_tag_recv_info_t info;
    ucs_status_t        status;

    uint64_t send_data = 0xdeadbeefdeadbeef;

    send_b(&send_data, sizeof(send_data), DATATYPE, 0x111337);

    short_progress_loop(); /* Receive messages as unexpected */

    status = recv_b(NULL, 0, DATATYPE, 0x1337, 0xffff, &info);
    EXPECT_EQ(UCS_ERR_MESSAGE_TRUNCATED, status);
}

UCS_TEST_P(test_ucp_tag_match, send_recv_nb_exp) {

    uint64_t send_data = 0xdeadbeefdeadbeef;
    uint64_t recv_data = 0;

    request *my_recv_req;
    my_recv_req = recv_nb(&recv_data, sizeof(recv_data), DATATYPE, 0x1337, 0xffff);

    ASSERT_TRUE(!UCS_PTR_IS_ERR(my_recv_req));
    ASSERT_TRUE(my_recv_req != NULL); /* Couldn't be completed because didn't send yet */

    send_b(&send_data, sizeof(send_data), DATATYPE, 0x111337);

    wait(my_recv_req);

    EXPECT_TRUE(my_recv_req->completed);
    EXPECT_EQ(UCS_OK,              my_recv_req->status);
    EXPECT_EQ(sizeof(send_data),   my_recv_req->info.length);
    EXPECT_EQ((ucp_tag_t)0x111337, my_recv_req->info.sender_tag);
    EXPECT_EQ(send_data, recv_data);
    request_release(my_recv_req);
}

UCS_TEST_P(test_ucp_tag_match, send_nb_multiple_recv_unexp) {
    const unsigned      num_requests = 1000;
    ucp_tag_recv_info_t info;
    ucs_status_t        status;

    uint64_t send_data = 0xdeadbeefdeadbeef;
    uint64_t recv_data = 0;

    std::vector<request*> send_reqs(num_requests);

    skip_loopback();

    for (unsigned i = 0; i < num_requests; ++i) {
        send_reqs[i] = send_nb(&send_data, sizeof(send_data), DATATYPE, 0x111337);
        ASSERT_TRUE(!UCS_PTR_IS_ERR(send_reqs[i]));
    }

    ucp_worker_progress(receiver().worker());

    for (unsigned i = 0; i < num_requests; ++i) {
        status = recv_b(&recv_data, sizeof(recv_data), DATATYPE, 0x1337, 0xffff,
                        &info);
        ASSERT_UCS_OK(status);
        ASSERT_EQ(num_requests, send_reqs.size());

        EXPECT_EQ(sizeof(send_data),   info.length);
        EXPECT_EQ((ucp_tag_t)0x111337, info.sender_tag);
        EXPECT_EQ(send_data, recv_data);
    }

    for (unsigned i = 0; i < num_requests; ++i) {
        if (send_reqs[i] != NULL) {
            EXPECT_TRUE(send_reqs[i]->completed);
            EXPECT_EQ(UCS_OK, send_reqs[i]->status);
            request_release(send_reqs[i]);
        }
    }
}

UCS_TEST_P(test_ucp_tag_match, sync_send_unexp) {
    ucp_tag_recv_info_t info;
    ucs_status_t        status;

    uint64_t send_data = 0x0102030405060708;
    uint64_t recv_data = 0;

    request *my_send_req = send_sync_nb(&send_data, sizeof(send_data), DATATYPE,
                                        0x111337);
    short_progress_loop();

    ASSERT_TRUE(my_send_req != NULL);
    EXPECT_FALSE(my_send_req->completed);

    ucp_worker_progress(receiver().worker());

    status = recv_b(&recv_data, sizeof(recv_data), DATATYPE, 0x1337, 0xffff, &info);
    ASSERT_UCS_OK(status);

    EXPECT_EQ(sizeof(send_data),   info.length);
    EXPECT_EQ((ucp_tag_t)0x111337, info.sender_tag);
    EXPECT_EQ(send_data, recv_data);

    short_progress_loop();

    EXPECT_TRUE(my_send_req->completed);
    EXPECT_EQ(UCS_OK, my_send_req->status);
    request_release(my_send_req);
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_tag_match)

class test_ucp_tag_match_rndv : public test_ucp_tag_match {
public:
    enum {
        RNDV_SCHEME_AUTO = 0,
        RNDV_SCHEME_PUT_ZCOPY,
        RNDV_SCHEME_GET_ZCOPY
    };

    static const std::string rndv_schemes[];

    void init() {
        ASSERT_LE(GetParam().variant, (int)RNDV_SCHEME_GET_ZCOPY);
        modify_config("RNDV_SCHEME", rndv_schemes[GetParam().variant]);

        test_ucp_tag_match::init();
    }

    std::vector<ucp_test_param>
    static enum_test_params(const ucp_params_t& ctx_params,
                            const std::string& name,
                            const std::string& test_case_name,
                            const std::string& tls)
    {
        std::vector<ucp_test_param> result;
        generate_test_params_variant(ctx_params, name,
                                     test_case_name + "/rndv_" +
                                     rndv_schemes[RNDV_SCHEME_AUTO],
                                     tls, RNDV_SCHEME_AUTO, result);
        generate_test_params_variant(ctx_params, name,
                                     test_case_name + "/rndv_" +
                                     rndv_schemes[RNDV_SCHEME_PUT_ZCOPY],
                                     tls, RNDV_SCHEME_PUT_ZCOPY, result);
        generate_test_params_variant(ctx_params, name,
                                     test_case_name + "/rndv_" +
                                     rndv_schemes[RNDV_SCHEME_GET_ZCOPY],
                                     tls, RNDV_SCHEME_GET_ZCOPY, result);
        return result;
    }
};

const std::string test_ucp_tag_match_rndv::rndv_schemes[] = { "auto",
                                                              "put_zcopy",
                                                              "get_zcopy" };

UCS_TEST_P(test_ucp_tag_match_rndv, sync_send_unexp, "RNDV_THRESH=1048576") {
    static const size_t size = 1148576;
    request             *my_send_req;
    ucp_tag_recv_info_t info;
    ucs_status_t        status;

    std::vector<char> sendbuf(size, 0);
    std::vector<char> recvbuf(size, 0);

    ucs::fill_random(sendbuf);

    /* sender - send the rts*/
    my_send_req = send_sync_nb(&sendbuf[0], sendbuf.size(), DATATYPE, 0x111337);
    /* receiver - get the rts and put in unexpected */
    short_progress_loop();

    ASSERT_TRUE(my_send_req != NULL);
    EXPECT_FALSE(my_send_req->completed);

    /* receiver - issue a recv req, match the rts, perform rndv-get and send ats to sender */
    status = recv_b(&recvbuf[0], recvbuf.size(), DATATYPE, 0x1337, 0xffff, &info);
    ASSERT_UCS_OK(status);

    EXPECT_EQ(sendbuf.size(), info.length);
    EXPECT_EQ((ucp_tag_t)0x111337, info.sender_tag);
    EXPECT_EQ(sendbuf, recvbuf);

    /* sender - get the ATS and set send request to completed */
    wait_for_flag(&my_send_req->completed);

    EXPECT_TRUE(my_send_req->completed);
    EXPECT_EQ(UCS_OK, my_send_req->status);
    request_release(my_send_req);
}

UCS_TEST_P(test_ucp_tag_match_rndv, req_exp, "RNDV_THRESH=1048576") {
    static const size_t size = 1148576;
    request *my_send_req, *my_recv_req;

    std::vector<char> sendbuf(size, 0);
    std::vector<char> recvbuf(size, 0);

    skip_loopback();

    ucs::fill_random(sendbuf);

    /* receiver - put the receive request into expected */
    my_recv_req = recv_nb(&recvbuf[0], recvbuf.size(), DATATYPE, 0x1337, 0xffff);
    ASSERT_TRUE(!UCS_PTR_IS_ERR(my_recv_req));
    EXPECT_FALSE(my_recv_req->completed);

    /* sender - send the RTS */
    my_send_req = send_nb(&sendbuf[0], sendbuf.size(), DATATYPE, 0x111337);
    ASSERT_TRUE(!UCS_PTR_IS_ERR(my_send_req));

    /* receiver - match the rts, perform rndv get and send an ack upon finishing */
    short_progress_loop();
    /* for UCTs that cannot perform real rndv and may do eager send-recv bcopy instead */
    wait(my_recv_req);

    EXPECT_EQ(sendbuf.size(),      my_recv_req->info.length);
    EXPECT_EQ((ucp_tag_t)0x111337, my_recv_req->info.sender_tag);
    EXPECT_TRUE(my_recv_req->completed);
    EXPECT_EQ(sendbuf, recvbuf);

    wait_and_validate(my_send_req);
    request_release(my_recv_req);
}

UCS_TEST_P(test_ucp_tag_match_rndv, rts_unexp, "RNDV_THRESH=1048576") {
    static const size_t size = 1148576;
    request             *my_send_req;
    ucp_tag_recv_info_t info;
    ucs_status_t        status;

    std::vector<char> sendbuf(size, 0);
    std::vector<char> recvbuf(size, 0);

    skip_loopback();

    ucs::fill_random(sendbuf);

    /* sender - send the RTS */
    my_send_req = send_nb(&sendbuf[0], sendbuf.size(), DATATYPE, 0x111337);
    ASSERT_TRUE(!UCS_PTR_IS_ERR(my_send_req));

    /* receiver - get the RTS and put it into unexpected */
    short_progress_loop();

    /* receiver - issue a receive request, match it with the RTS and perform rndv get */
    status = recv_b(&recvbuf[0], recvbuf.size(), DATATYPE, 0x1337, 0xffff, &info);
    ASSERT_UCS_OK(status);

    /* sender - get the ATS and set send request to completed */
    wait_and_validate(my_send_req);

    EXPECT_EQ(sendbuf.size()     , info.length);
    EXPECT_EQ((ucp_tag_t)0x111337, info.sender_tag);
    EXPECT_EQ(sendbuf, recvbuf);
}

UCS_TEST_P(test_ucp_tag_match_rndv, truncated, "RNDV_THRESH=1048576") {
    static const size_t size = 1148576;
    request *my_send_req;
    ucp_tag_recv_info_t info;
    ucs_status_t status;

    std::vector<char> sendbuf(size, 0);

    skip_loopback();

    ucs::fill_random(sendbuf);

    /* sender - send the RTS */
    my_send_req = send_nb(&sendbuf[0], sendbuf.size(), DATATYPE, 0x111337);
    ASSERT_TRUE(!UCS_PTR_IS_ERR(my_send_req));

    /* receiver - get the RTS and put it into unexpected */
    short_progress_loop();

    /* receiver - issue a receive request with zero length,
     * no assertions should occur */
    status = recv_b(NULL, 0, DATATYPE, 0x1337, 0xffff, &info);
    EXPECT_EQ(UCS_ERR_MESSAGE_TRUNCATED, status);

    /* sender - get the ATS and set send request to completed */
    wait_and_validate(my_send_req);
}

UCS_TEST_P(test_ucp_tag_match_rndv, post_larger_recv, "RNDV_THRESH=0") {
    /* small send size should probably be lower than minimum GET Zcopy
     * size supported by IB TLs */
    static const size_t small_send_size = 16;
    static const size_t small_recv_size = small_send_size * 2;
    static const size_t large_send_size = 1148576;
    static const size_t large_recv_size = large_send_size + 1 * UCS_KBYTE;
    /* array of [send][recv] sizes */
    static const size_t sizes[][2] = { { small_send_size, small_recv_size },
                                       { large_send_size, large_recv_size } };
    request *my_send_req, *my_recv_req;

    for (unsigned i = 0; i < ucs_array_size(sizes); i++) {
        size_t send_size = sizes[i][0];
        size_t recv_size = sizes[i][1];
        std::vector<char> sendbuf(send_size, 0);
        std::vector<char> recvbuf(recv_size, 0);

        ucs::fill_random(sendbuf);
        ucs::fill_random(recvbuf);

        my_recv_req = recv_nb(&recvbuf[0], recvbuf.size(), DATATYPE, 0x1337, 0xffff);
        ASSERT_TRUE(!UCS_PTR_IS_ERR(my_recv_req));
        EXPECT_FALSE(my_recv_req->completed);

        my_send_req = send_nb(&sendbuf[0], sendbuf.size(), DATATYPE, 0x111337);
        ASSERT_TRUE(!UCS_PTR_IS_ERR(my_send_req));

        wait(my_recv_req);

        EXPECT_EQ(sendbuf.size(), my_recv_req->info.length);
        EXPECT_EQ(recvbuf.size(), ((ucp_request_t*)my_recv_req - 1)->recv.length);
        EXPECT_EQ((ucp_tag_t)0x111337, my_recv_req->info.sender_tag);
        EXPECT_TRUE(my_recv_req->completed);
        EXPECT_NE(sendbuf, recvbuf);
        EXPECT_TRUE(std::equal(sendbuf.begin(), sendbuf.end(), recvbuf.begin()));

        wait_and_validate(my_send_req);
        request_release(my_recv_req);
    }
}

UCS_TEST_P(test_ucp_tag_match_rndv, req_exp_auto_thresh, "RNDV_THRESH=auto") {
    static const size_t size = 1148576;
    request *my_send_req, *my_recv_req;

    std::vector<char> sendbuf(size, 0);
    std::vector<char> recvbuf(size, 0);

    skip_loopback();

    ucs::fill_random(sendbuf);

    /* receiver - put the receive request into expected */
    my_recv_req = recv_nb(&recvbuf[0], recvbuf.size(), DATATYPE, 0x1337, 0xffff);
    ASSERT_TRUE(!UCS_PTR_IS_ERR(my_recv_req));
    EXPECT_FALSE(my_recv_req->completed);

    /* sender - send the RTS */
    my_send_req = send_nb(&sendbuf[0], sendbuf.size(), DATATYPE, 0x111337);
    ASSERT_TRUE(!UCS_PTR_IS_ERR(my_send_req));

    /* receiver - match the rts, perform rndv get and send an ack upon finishing */
    short_progress_loop();
    /* for UCTs that cannot perform real rndv and may do eager send-recv bcopy instead */
    wait(my_recv_req);

    EXPECT_EQ(sendbuf.size(),      my_recv_req->info.length);
    EXPECT_EQ((ucp_tag_t)0x111337, my_recv_req->info.sender_tag);
    EXPECT_TRUE(my_recv_req->completed);
    EXPECT_EQ(sendbuf, recvbuf);

    /* sender - get the ATS and set send request to completed */
    wait_and_validate(my_send_req);
    request_release(my_recv_req);
}

UCS_TEST_P(test_ucp_tag_match_rndv, exp_huge_mix) {
    const size_t sizes[] = { 1000, 2000, 8000, 2500ul * UCS_MBYTE };

    /* small sizes should warm-up tag cache */
    for (unsigned i = 0; i < ucs_array_size(sizes); ++i) {
        const size_t size = sizes[i] / ucs::test_time_multiplier();
        request *my_send_req, *my_recv_req;

        std::vector<char> sendbuf(size, 0);
        std::vector<char> recvbuf(size, 0);

        ucs::fill_random(sendbuf);

        my_recv_req = recv_nb(&recvbuf[0], recvbuf.size(), DATATYPE, 0x1337, 0xffff);
        ASSERT_TRUE(!UCS_PTR_IS_ERR(my_recv_req));
        EXPECT_FALSE(my_recv_req->completed);

        my_send_req = send_nb(&sendbuf[0], sendbuf.size(), DATATYPE, 0x111337);
        ASSERT_TRUE(!UCS_PTR_IS_ERR(my_send_req));

        wait(my_recv_req);

        EXPECT_EQ(sendbuf.size(),      my_recv_req->info.length);
        EXPECT_EQ((ucp_tag_t)0x111337, my_recv_req->info.sender_tag);
        EXPECT_TRUE(my_recv_req->completed);
        EXPECT_EQ(sendbuf, recvbuf);

        wait_and_validate(my_send_req);
        request_free(my_recv_req);
    }
}

UCS_TEST_P(test_ucp_tag_match_rndv, bidir_multi_exp_post, "RNDV_THRESH=0") {
    const size_t sizes[] = { 8 * UCS_KBYTE, 128 * UCS_KBYTE, 512 * UCS_KBYTE,
                             8 * UCS_MBYTE, 128 * UCS_MBYTE, 512 * UCS_MBYTE };

    receiver().connect(&sender(), get_ep_params());

    for (unsigned i = 0; i < ucs_array_size(sizes); ++i) {
        const size_t size = sizes[i] /
                            ucs::test_time_multiplier() /
                            ucs::test_time_multiplier();
        const size_t count = ucs_max((size_t)(5000.0 / sqrt(sizes[i]) /
                                              ucs::test_time_multiplier()), 3lu);
        std::vector<request*> sreqs;
        std::vector<request*> rreqs;
        std::vector<std::vector<char> > sbufs;
        std::vector<std::vector<char> > rbufs;

        sbufs.resize(count * 2);
        rbufs.resize(count * 2);

        for (size_t repeat = 0; repeat < count * 2; ++repeat) {
            entity &send_e = repeat < count ? sender() : receiver();
            entity &recv_e = repeat < count ? receiver() : sender();
            request *my_send_req, *my_recv_req;

            sbufs[repeat].resize(size, 0);
            rbufs[repeat].resize(size, 0);
            ucs::fill_random(sbufs[repeat]);

            my_recv_req = recv(recv_e, RECV_NB,
                               &rbufs[repeat][0], rbufs[repeat].size(),
                               DATATYPE, 0x1337, 0xffff, NULL);
            ASSERT_TRUE(!UCS_PTR_IS_ERR(my_recv_req));
            EXPECT_FALSE(my_recv_req->completed);

            my_send_req = send(send_e, SEND_NB,
                               &sbufs[repeat][0], sbufs[repeat].size(),
                               DATATYPE, 0x111337);
            ASSERT_TRUE(!UCS_PTR_IS_ERR(my_send_req));

            sreqs.push_back(my_send_req);
            rreqs.push_back(my_recv_req);
        }

        for (size_t repeat = 0; repeat < count * 2; ++repeat) {
            request *my_send_req, *my_recv_req;

            my_recv_req = rreqs[repeat];
            my_send_req = sreqs[repeat];

            wait(my_recv_req);

            EXPECT_EQ(sbufs[repeat].size(), my_recv_req->info.length);
            EXPECT_EQ((ucp_tag_t)0x111337, my_recv_req->info.sender_tag);
            EXPECT_TRUE(my_recv_req->completed);
            EXPECT_EQ(sbufs[repeat], rbufs[repeat]);

            wait_and_validate(my_send_req);
            request_free(my_recv_req);
        }
    }
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_tag_match_rndv)
