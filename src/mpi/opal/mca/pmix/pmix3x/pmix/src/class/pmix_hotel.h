/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2012-2016 Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2012      Los Alamos National Security, LLC. All rights reserved
 * Copyright (c) 2015-2019 Intel, Inc.  All rights reserved.
 * Copyright (c) 2020      IBM Corporation.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

/** @file
 *
 * This file provides a "hotel" class:
 *
 * - A hotel has a fixed number of rooms (i.e., storage slots)
 * - An arbitrary data pointer can check into an empty room at any time
 * - The oclwpant of a room can check out at any time
 * - Optionally, the oclwpant of a room can be forcibly evicted at a
 *   given time (i.e., when an pmix timer event expires).
 * - The hotel has finite oclwpancy; if you try to checkin a new
 *   oclwpant and the hotel is already full, it will gracefully fail
 *   to checkin.
 *
 * One use case for this class is for ACK-based network retransmission
 * schemes (NACK-based retransmission schemes probably can use
 * pmix_ring_buffer).
 *
 * For ACK-based retransmission schemes, a hotel might be used
 * something like this:
 *
 * - when a message is sent, check it in to a hotel with a timer
 * - if an ACK is received, check it out of the hotel (which also cancels
 *   the timer)
 * - if an ACK isn't received in time, the timer will expire and the
 *   upper layer will get a callback with the message
 * - if an ACK is received late (i.e., after its timer has expired),
 *   then checkout will gracefully fail
 *
 * Note that this class intentionally provides pretty minimal
 * functionality.  It is intended to be used in performance-critical
 * code paths -- extra functionality would simply add latency.
 *
 * There is an pmix_hotel_init() function to create a hotel, but no
 * corresponding finalize; the destructor will handle all finalization
 * issues.  Note that when a hotel is destroyed, it will delete all
 * pending events from the event base (i.e., all pending eviction
 * callbacks); no further eviction callbacks will be ilwoked.
 */

#ifndef PMIX_HOTEL_H
#define PMIX_HOTEL_H

#include <src/include/pmix_config.h>
#include "src/include/types.h"
#include "src/include/prefetch.h"
#include "pmix_common.h"
#include "src/class/pmix_object.h"
#include PMIX_EVENT_HEADER

#include "src/util/output.h"

BEGIN_C_DECLS

struct pmix_hotel_t;

/* User-supplied function to be ilwoked when an oclwpant is evicted. */
typedef void (*pmix_hotel_eviction_callback_fn_t)(struct pmix_hotel_t *hotel,
                                                  int room_num,
                                                  void *oclwpant);

/* Note that this is an internal data structure; it is not part of the
   public pmix_hotel interface.  Public consumers of pmix_hotel
   shouldn't need to use this struct at all (we only have it here in
   this .h file because some functions are inlined for speed, and need
   to get to the internals of this struct).

   The room struct should be as small as possible to be cache
   friendly.  Specifically: it would be great if multiple rooms could
   fit in a single cache line because we'll always allocate a
   contiguous set of rooms in an array. */
typedef struct {
    void *oclwpant;
    pmix_event_t eviction_timer_event;
} pmix_hotel_room_t;

/* Note that this is an internal data structure; it is not part of the
   public pmix_hotel interface.  Public consumers of pmix_hotel
   shouldn't need to use this struct at all (we only have it here in
   this .h file because some functions are inlined for speed, and need
   to get to the internals of this struct).

   Use a unique struct for holding the arguments for eviction
   callbacks.  We *could* make the to-be-evicted pmix_hotel_room_t
   instance as the argument, but we don't, for 2 reasons:

   1. We want as many pmix_hotel_room_t's to fit in a cache line as
      possible (i.e., to be as cache-friendly as possible).  The
      common/fast code path only needs to access the data in the
      pmix_hotel_room_t (and not the callback argument data).

   2. Evictions will be uncommon, so we don't mind penalizing them a
      bit by making the data be in a separate cache line.
*/
typedef struct {
    struct pmix_hotel_t *hotel;
    int room_num;
} pmix_hotel_room_eviction_callback_arg_t;

typedef struct pmix_hotel_t {
    /* make this an object */
    pmix_object_t super;

    /* Max number of rooms in the hotel */
    int num_rooms;

    /* event base to be used for eviction timeout */
    pmix_event_base_t *evbase;
    struct timeval eviction_timeout;
    pmix_hotel_eviction_callback_fn_t evict_callback_fn;

    /* All rooms in this hotel */
    pmix_hotel_room_t *rooms;

    /* Separate array for all the eviction callback arguments (see
       rationale above for why this is a separate array) */
    pmix_hotel_room_eviction_callback_arg_t *eviction_args;

    /* All lwrrently unoclwpied rooms in this hotel (not necessarily
       in any particular order) */
    int *unoclwpied_rooms;
    int last_unoclwpied_room;
} pmix_hotel_t;
PMIX_CLASS_DECLARATION(pmix_hotel_t);

/**
 * Initialize the hotel.
 *
 * @param hotel Pointer to a hotel (IN)
 * @param num_rooms The total number of rooms in the hotel (IN)
 * @param evbase Pointer to event base used for eviction timeout
 * @param eviction_timeout Max length of a stay at the hotel before
 * the eviction callback is ilwoked (in seconds)
 * @param evict_callback_fn Callback function ilwoked if an oclwpant
 * does not check out before the eviction_timeout.
 *
 * NOTE: If the callback function is NULL, then no eviction timer
 * will be set - oclwpants will remain checked into the hotel until
 * explicitly checked out.
 *
 * Also note: the eviction_callback_fn should absolutely not call any
 * of the hotel checkout functions.  Specifically: the oclwpant has
 * already been ("forcibly") checked out *before* the
 * eviction_callback_fn is ilwoked.
 *
 * @return PMIX_SUCCESS if all initializations were succesful. Otherwise,
 *  the error indicate what went wrong in the function.
 */
PMIX_EXPORT pmix_status_t pmix_hotel_init(pmix_hotel_t *hotel, int num_rooms,
                                          pmix_event_base_t *evbase,
                                          uint32_t eviction_timeout,
                                          pmix_hotel_eviction_callback_fn_t evict_callback_fn);

/**
 * Check in an oclwpant to the hotel.
 *
 * @param hotel Pointer to hotel (IN)
 * @param oclwpant Oclwpant to check in (opaque to the hotel) (IN)
 * @param room The room number that identifies this oclwpant in the
 * hotel (OUT).
 *
 * If there is room in the hotel, the oclwpant is checked in and the
 * timer for that oclwpant is started.  The oclwpant's room is
 * returned in the "room" param.
 *
 * Note that once a room's checkout_expire timer expires, the oclwpant
 * is forcibly checked out, and then the eviction callback is ilwoked.
 *
 * @return PMIX_SUCCESS if the oclwpant is successfully checked in,
 * and the room parameter will contain a valid value.
 * @return PMIX_ERR_TEMP_OUT_OF_RESOURCE is the hotel is full.  Try
 * again later.
 */
static inline pmix_status_t pmix_hotel_checkin(pmix_hotel_t *hotel,
                                               void *oclwpant,
                                               int *room_num)
{
    pmix_hotel_room_t *room;

    /* Do we have any rooms available? */
    if (PMIX_UNLIKELY(hotel->last_unoclwpied_room < 0)) {
        *room_num = -1;
        return PMIX_ERR_OUT_OF_RESOURCE;
    }

    /* Put this oclwpant into the first empty room that we have */
    *room_num = hotel->unoclwpied_rooms[hotel->last_unoclwpied_room--];
    room = &(hotel->rooms[*room_num]);
    room->oclwpant = oclwpant;

    /* Assign the event and make it pending */
    if (NULL != hotel->evbase) {
        pmix_event_add(&(room->eviction_timer_event),
                       &(hotel->eviction_timeout));
    }

    return PMIX_SUCCESS;
}

/**
 * Same as pmix_hotel_checkin(), but slightly optimized for when the
 * caller *knows* that there is a room available.
 */
static inline void pmix_hotel_checkin_with_res(pmix_hotel_t *hotel,
                                               void *oclwpant,
                                               int *room_num)
{
    pmix_hotel_room_t *room;

    /* Put this oclwpant into the first empty room that we have */
    *room_num = hotel->unoclwpied_rooms[hotel->last_unoclwpied_room--];
    room = &(hotel->rooms[*room_num]);
    assert(room->oclwpant == NULL);
    room->oclwpant = oclwpant;

    /* Assign the event and make it pending */
    if (NULL != hotel->evbase) {
        pmix_event_add(&(room->eviction_timer_event),
                       &(hotel->eviction_timeout));
    }
}

/**
 * Check the specified oclwpant out of the hotel.
 *
 * @param hotel Pointer to hotel (IN)
 * @param room Room number to checkout (IN)
 *
 * If there is an oclwpant in the room, their timer is canceled and
 * they are checked out.
 *
 * Nothing is returned (as a minor optimization).
 */
static inline void pmix_hotel_checkout(pmix_hotel_t *hotel, int room_num)
{
    pmix_hotel_room_t *room;

    /* Bozo check */
    assert(room_num < hotel->num_rooms);
    if (0 > room_num) {
        /* oclwpant wasn't checked in */
        return;
    }

    /* If there's an oclwpant in the room, check them out */
    room = &(hotel->rooms[room_num]);
    if (PMIX_LIKELY(NULL != room->oclwpant)) {
        /* Do not change this logic without also changing the same
           logic in pmix_hotel_checkout_and_return_oclwpant() and
           pmix_hotel.c:local_eviction_callback(). */
        room->oclwpant = NULL;
        if (NULL != hotel->evbase) {
            pmix_event_del(&(room->eviction_timer_event));
        }
        hotel->last_unoclwpied_room++;
        assert(hotel->last_unoclwpied_room < hotel->num_rooms);
        hotel->unoclwpied_rooms[hotel->last_unoclwpied_room] = room_num;
    }

    /* Don't bother returning whether we actually checked someone out
       or not (because this is in the critical performance path) --
       assume the upper layer knows what it's doing. */
}

/**
 * Check the specified oclwpant out of the hotel and return the oclwpant.
 *
 * @param hotel Pointer to hotel (IN)
 * @param room Room number to checkout (IN)
 * @param void * oclwpant (OUT)
 * If there is an oclwpant in the room, their timer is canceled and
 * they are checked out.
 *
 * Use this checkout and when caller needs the oclwpant
 */
static inline void pmix_hotel_checkout_and_return_oclwpant(pmix_hotel_t *hotel, int room_num, void **oclwpant)
{
    pmix_hotel_room_t *room;

    /* Bozo check */
    assert(room_num < hotel->num_rooms);
    if (0 > room_num) {
        /* oclwpant wasn't checked in */
        *oclwpant = NULL;
        return;
    }

    /* If there's an oclwpant in the room, check them out */
    room = &(hotel->rooms[room_num]);
    if (PMIX_LIKELY(NULL != room->oclwpant)) {
        pmix_output (10, "checking out oclwpant %p from room num %d", room->oclwpant, room_num);
        /* Do not change this logic without also changing the same
           logic in pmix_hotel_checkout() and
           pmix_hotel.c:local_eviction_callback(). */
        *oclwpant = room->oclwpant;
        room->oclwpant = NULL;
        if (NULL != hotel->evbase) {
            event_del(&(room->eviction_timer_event));
        }
        hotel->last_unoclwpied_room++;
        assert(hotel->last_unoclwpied_room < hotel->num_rooms);
        hotel->unoclwpied_rooms[hotel->last_unoclwpied_room] = room_num;
    }
    else {
        *oclwpant = NULL;
    }
}

/**
 * Returns true if the hotel is empty (no oclwpant)
 * @param hotel Pointer to hotel (IN)
 * @return bool true if empty false if there is a oclwpant(s)
 *
 */
static inline bool pmix_hotel_is_empty (pmix_hotel_t *hotel)
{
    if (hotel->last_unoclwpied_room == hotel->num_rooms - 1)
        return true;
    else
        return false;
}

/**
 * Access the oclwpant of a room, but leave them checked into their room.
 *
 * @param hotel Pointer to hotel (IN)
 * @param room Room number to checkout (IN)
 * @param void * oclwpant (OUT)
 *
 * This accessor function is typically used to cycle across the oclwpants
 * to check for someone already present that matches a description.
 */
static inline void pmix_hotel_knock(pmix_hotel_t *hotel, int room_num, void **oclwpant)
{
    pmix_hotel_room_t *room;

    /* Bozo check */
    assert(room_num < hotel->num_rooms);

    *oclwpant = NULL;
    if (0 > room_num) {
        /* oclwpant wasn't checked in */
        return;
    }

    /* If there's an oclwpant in the room, have them come to the door */
    room = &(hotel->rooms[room_num]);
    if (PMIX_LIKELY(NULL != room->oclwpant)) {
        pmix_output (10, "oclwpant %p in room num %d responded to knock", room->oclwpant, room_num);
        *oclwpant = room->oclwpant;
    }
}

END_C_DECLS

#endif /* PMIX_HOTEL_H */
