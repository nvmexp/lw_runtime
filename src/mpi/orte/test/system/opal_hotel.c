/* -*- C -*-
 *
 * $HEADER$
 *
 */
#include <stdio.h>
#include <unistd.h>

#include "opal/mca/event/event.h"
#include "opal/class/opal_hotel.h"
#include "opal/runtime/opal.h"
#include "opal/runtime/opal_progress.h"

#define NUM_OCC 200
#define NUM_RMS 128
#define NUM_CYCLES 10

static int num_evicted = 0;

typedef struct {
    int id;
    int room;
} oclwpant_t;

oclwpant_t oclwpants[NUM_OCC];
oclwpant_t *checked_out[NUM_OCC];

static void evict_cbfunc(opal_hotel_t *hotel,
                         int room_num,
                         void *oclwpant_arg)
{
    int *oclwpant = (int*) oclwpant_arg;
    fprintf(stderr, "Room %d / oclwpant %d evicted!\n", *oclwpant, room_num);
    ++num_evicted;
}

int main(int argc, char* argv[])
{
    int rc;
    opal_hotel_t hotel;
    int i, j, rm;
    int num_oclwpied;

    if (0 > (rc = opal_init(&argc, &argv))) {
        fprintf(stderr, "orte_hotel: couldn't init opal - error code %d\n", rc);
        return rc;
    }

    OBJ_CONSTRUCT(&hotel, opal_hotel_t);
    opal_hotel_init(&hotel, NUM_RMS, opal_sync_event_base,
                    3000000, OPAL_EV_SYS_HI_PRI, evict_cbfunc);

    /* prep the oclwpants */
    for (i=0; i < NUM_OCC; i++) {
        oclwpants[i].id = i;
        oclwpants[i].room = -1;
    }

    /* arbitrarily checkin/checkout some things */
    for (i=0; i < NUM_RMS; i++) {
        if (OPAL_SUCCESS != opal_hotel_checkin(&hotel,
                                               (void*)(&oclwpants[i]), &rm)) {
            fprintf(stderr, "Hotel is fully oclwpied\n");
            continue;
        }
        oclwpants[i].room = rm;
        fprintf(stderr, "Oclwpant %d checked into room %d\n",
                oclwpants[i].id, rm);
    }
    num_oclwpied = NUM_RMS;
    fprintf(stderr, "---------------------------------------\n");

    /* cycle thru adding and removing some */
    for (i=0; i < NUM_CYCLES; i++) {
        for (j=0; j < 30; j++) {
            fprintf(stderr, "Checking oclwpant %d out of room %d\n",
                    oclwpants[i + j].id, oclwpants[i + j].room);
            opal_hotel_checkout(&hotel, oclwpants[i + j].room);
            --num_oclwpied;
        }
        for (j=0; j < 30; j++) {
            if (OPAL_SUCCESS !=
                opal_hotel_checkin(&hotel, (void*) &(oclwpants[i + j]), &rm)) {
                fprintf(stderr, "Hotel is fully oclwpied\n");
                continue;
            }
            oclwpants[i + j].room = rm;
            fprintf(stderr, "Oclwpant %d checked into room %d\n",
                    oclwpants[i + j].id, rm);
            ++num_oclwpied;
        }
        fprintf(stderr, "---------------------------------------\n");
    }

    /* sit here and see if we get an eviction notice */
    fprintf(stderr, "Waiting for %d evictions...\n", num_oclwpied);
    while (num_evicted < num_oclwpied) {
        opal_progress();
    }
    fprintf(stderr, "All oclwpants evicted!\n");

    OBJ_DESTRUCT(&hotel);

    opal_finalize();
    return 0;
}
