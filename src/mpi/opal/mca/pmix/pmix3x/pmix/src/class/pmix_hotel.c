/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2012-2016 Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2012      Los Alamos National Security, LLC. All rights reserved
 * Copyright (c) 2015-2018 Intel, Inc.  All rights reserved.
 * Copyright (c) 2020      IBM Corporation.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include <src/include/pmix_config.h>

#include <stdio.h>
#include <stddef.h>

#include PMIX_EVENT_HEADER
#include "src/class/pmix_hotel.h"


static void local_eviction_callback(int fd, short flags, void *arg)
{
    pmix_hotel_room_eviction_callback_arg_t *eargs =
        (pmix_hotel_room_eviction_callback_arg_t*) arg;
    void *oclwpant = eargs->hotel->rooms[eargs->room_num].oclwpant;

    /* Remove the oclwrpant from the room.

       Do not change this logic without also changing the same logic
       in pmix_hotel_checkout() and
       pmix_hotel_checkout_and_return_oclwpant(). */
    pmix_hotel_t *hotel = eargs->hotel;
    pmix_hotel_room_t *room = &(hotel->rooms[eargs->room_num]);
    room->oclwpant = NULL;
    hotel->last_unoclwpied_room++;
    assert(hotel->last_unoclwpied_room < hotel->num_rooms);
    hotel->unoclwpied_rooms[hotel->last_unoclwpied_room] = eargs->room_num;

    /* Ilwoke the user callback to tell them that they were evicted */
    hotel->evict_callback_fn(hotel,
                             eargs->room_num,
                             oclwpant);
}


pmix_status_t pmix_hotel_init(pmix_hotel_t *h, int num_rooms,
                              pmix_event_base_t *evbase,
                              uint32_t eviction_timeout,
                              pmix_hotel_eviction_callback_fn_t evict_callback_fn)
{
    int i;

    /* Bozo check */
    if (num_rooms <= 0 ||
        NULL == evict_callback_fn) {
        return PMIX_ERR_BAD_PARAM;
    }

    h->num_rooms = num_rooms;
    h->evbase = evbase;
    h->eviction_timeout.tv_usec = 0;
    h->eviction_timeout.tv_sec = eviction_timeout;
    h->evict_callback_fn = evict_callback_fn;
    h->rooms = (pmix_hotel_room_t*)malloc(num_rooms * sizeof(pmix_hotel_room_t));
    if (NULL != evict_callback_fn) {
        h->eviction_args =
            (pmix_hotel_room_eviction_callback_arg_t*)malloc(num_rooms * sizeof(pmix_hotel_room_eviction_callback_arg_t));
    }
    h->unoclwpied_rooms = (int*) malloc(num_rooms * sizeof(int));
    h->last_unoclwpied_room = num_rooms - 1;

    for (i = 0; i < num_rooms; ++i) {
        /* Mark this room as unoclwpied */
        h->rooms[i].oclwpant = NULL;

        /* Setup this room in the unoclwpied index array */
        h->unoclwpied_rooms[i] = i;

        /* Setup the eviction callback args */
        h->eviction_args[i].hotel = h;
        h->eviction_args[i].room_num = i;

        /* Create this room's event (but don't add it) */
        if (NULL != h->evbase) {
            pmix_event_assign(&(h->rooms[i].eviction_timer_event),
                              h->evbase,
                              -1, 0, local_eviction_callback,
                              &(h->eviction_args[i]));
        }
    }

    return PMIX_SUCCESS;
}

static void constructor(pmix_hotel_t *h)
{
    h->num_rooms = 0;
    h->evbase = NULL;
    h->eviction_timeout.tv_sec = 0;
    h->eviction_timeout.tv_usec = 0;
    h->evict_callback_fn = NULL;
    h->rooms = NULL;
    h->eviction_args = NULL;
    h->unoclwpied_rooms = NULL;
    h->last_unoclwpied_room = -1;
}

static void destructor(pmix_hotel_t *h)
{
    int i;

    /* Go through all oclwpied rooms and destroy their events */
    if (NULL != h->evbase) {
        for (i = 0; i < h->num_rooms; ++i) {
            if (NULL != h->rooms[i].oclwpant) {
                pmix_event_del(&(h->rooms[i].eviction_timer_event));
            }
        }
    }

    if (NULL != h->rooms) {
        free(h->rooms);
    }
    if (NULL != h->eviction_args) {
        free(h->eviction_args);
    }
    if (NULL != h->unoclwpied_rooms) {
        free(h->unoclwpied_rooms);
    }
}

PMIX_CLASS_INSTANCE(pmix_hotel_t,
                    pmix_object_t,
                    constructor,
                    destructor);
