/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright(c) 2010-2014 Intel Corporation
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <getopt.h>
#include <stdarg.h>
#include <errno.h>

#include <rte_memory.h>
#include <rte_ethdev.h>
#include <rte_string_fns.h>

#include "common.h"
#include "args.h"
#include "init.h"

/* global var for number of clients - extern in header */
uint8_t num_clients;

static const char *progname;

/**
 * Prints out usage information to stdout
 */
static void
usage(void)
{
	printf(
	    "%s [EAL options] -- -p PORTMASK -n NUM_CLIENTS [-s NUM_SOCKETS]\n"
	    " -p PORTMASK: hexadecimal bitmask of ports to use\n"
	    " -n NUM_CLIENTS: number of client processes to use\n"
	    , progname);
}

/**
 * The ports to be used by the application are passed in
 * the form of a bitmask. This function parses the bitmask
 * and places the port numbers to be used into the port[]
 * array variable
 */
static int
parse_portmask(const char *portmask)
{
	char *end = NULL;
	unsigned long long pm;
	uint16_t id;

	if (portmask == NULL || *portmask == '\0')
		return -1;

	/* colwert parameter to a number and verify */
	errno = 0;
	pm = strtoull(portmask, &end, 16);
	if (errno != 0 || end == NULL || *end != '\0')
		return -1;

	RTE_ETH_FOREACH_DEV(id) {
		unsigned long msk = 1u << id;

		if ((pm & msk) == 0)
			continue;

		pm &= ~msk;
		ports->id[ports->num_ports++] = id;
	}

	if (pm != 0) {
		printf("WARNING: leftover ports in mask %#llx - ignoring\n",
		       pm);
	}

	return 0;
}

/**
 * Take the number of clients parameter passed to the app
 * and colwert to a number to store in the num_clients variable
 */
static int
parse_num_clients(const char *clients)
{
	char *end = NULL;
	unsigned long temp;

	if (clients == NULL || *clients == '\0')
		return -1;

	temp = strtoul(clients, &end, 10);
	if (end == NULL || *end != '\0' || temp == 0)
		return -1;

	num_clients = (uint8_t)temp;
	return 0;
}

/**
 * The application specific arguments follow the DPDK-specific
 * arguments which are stripped by the DPDK init. This function
 * processes these application arguments, printing usage info
 * on error.
 */
int
parse_app_args(int argc, char *argv[])
{
	int option_index, opt;
	char **argvopt = argv;
	static struct option lgopts[] = { /* no long options */
		{NULL, 0, 0, 0 }
	};
	progname = argv[0];

	while ((opt = getopt_long(argc, argvopt, "n:p:", lgopts,
		&option_index)) != EOF){
		switch (opt){
			case 'p':
				if (parse_portmask(optarg) != 0) {
					usage();
					return -1;
				}
				break;
			case 'n':
				if (parse_num_clients(optarg) != 0){
					usage();
					return -1;
				}
				break;
			default:
				printf("ERROR: Unknown option '%c'\n", opt);
				usage();
				return -1;
		}
	}

	if (ports->num_ports == 0 || num_clients == 0){
		usage();
		return -1;
	}

	if (ports->num_ports % 2 != 0){
		printf("ERROR: application requires an even number of ports to use\n");
		return -1;
	}
	return 0;
}
