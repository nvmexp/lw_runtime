/*
 * Copyright (c) 2004-2005 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2005 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2008-2014 Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2014-2015 Hewlett-Packard Development Company, LP.
 *                         All rights reserved.
 * Copyright (c) 2014-2015 Mellanox Technologies, Inc.
 *                         All rights reserved.
 * Copyright (c) 2014      Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "opal_config.h"
#include <stdint.h>
#include <string.h>
#include "support.h"
#include "opal/class/opal_object.h"
#include "opal/class/opal_hash_table.h"
#include "opal/runtime/opal.h"
#include "opal/constants.h"

static FILE *error_out=NULL;

char *num_keys[] = {
    "0", "1234", "1234",
    "0", "5678", "5678",
    "1", "12450", "12450",
    "1", "45623", "45623",
    NULL
};


char *str_keys[] = {
    "foo", "bar",
    "2", "this cow jumped over the moon",
    "this is another key", "this is another value",
    "key key", "value value",
    NULL
};


char *perm_keys[] = {
    "abcdef", "abcdef",
    "bcdefa", "bcdefa",
    "cdefab", "cdefab",
    "defabc", "defabc",
    "efabcd", "efabcd",
    "fabcde", "fabcde",
    "badcfe", "badcfe",
    "badcef", "badcef",
    "abdcfe", "abdcfe",
    "bcdaef", "bcdaef",
    NULL
};

#if 0
/*
 * This data specifically knows about the April'2014 version of hash tables.
 * It inserts some keys.
 * It inserts some more with a capacity offset to generate collisions.
 * Then it checks the table via traversal.
 * Then... it removes a key and checks again (via traversal)
 * and removes another key and re-checks.
 */
static char* remove_keys[] = {
    "1", "A", "2", "B", "4", "D", "6", "F", "10", "J", NULL, /* insert as-is: ...AB.D.F...J... */
    "2", "b", "4", "d", "5", "e", "3", "c", NULL, /* insert with capacity-offset: ...ABbDdFec.J... */
    "ABbDdFecJ",		/* traversal expectation */
    "4", "ABbdeFcJ",		/* remove D (...ABbdeFc..J...) then expected traversal */
    "2", "AbcdeFJ",		/* remove B (...AbcdeF...J...) then expected traversal */
    NULL			/* end removals and expectations */
};
#endif

typedef union {
    opal_process_name_t uvalue;
    void *vvalue;
} value_t;

static void validate_table(opal_proc_table_t *table, char *keys[])
{
    int         j, ret;
    value_t value;

    for ( j = 0; keys[j]; j += 3) {
        opal_process_name_t key;
        key.jobid = atoi(keys[j]);
        key.vpid = atoi(keys[j+1]);
        ret = opal_proc_table_get_value(table, key,
                                        (void**) &value.uvalue);
        if (OPAL_SUCCESS != ret) {
            test_failure("opal_proc_table_get_value failed");
        }
        test_verify_str(keys[j+2], value.vvalue);
    }
}

#if 0
static void
validate_remove_traversal(opal_hash_table_t * table, const char * expected_chars)
{
    /* all values are single-character strings */
    /* expected_chars are those single characters as a string */
    const int debug = 0;	/* turn this on if you want to see the details */
    int rc, problems = 0;
    const char * expected_scanner = expected_chars;
    uint32_t key;
    void * raw_value;
    void * node;
    if (debug) {
	fprintf(stderr, "debug: expecting '%s' capacity is %d\n",
		expected_chars, (int) table->ht_capacity);
    }
    for (rc = opal_hash_table_get_first_key_uint32(table, &key, &raw_value, &node);
	 OPAL_SUCCESS == rc;
	 rc = opal_hash_table_get_next_key_uint32(table, &key, &raw_value, node, &node)) {
	const char * value = (const char *) raw_value;
	char expected, actual;
	if (debug) {
	    fprintf(stderr, "key %d (probe at %d) value '%s' excpected_scanner '%s'\n",
		    key, (int) (key%table->ht_capacity), value, expected_scanner);
	}
	if (1 != strlen(value)) {
	    fprintf(stderr, "key %d's value '%s' is not a one-character string\n", key, value);
	    problems += 1;
	    continue;		/* might as well be completely noisy */
	}
	if ('\0' == *expected_scanner) {
	    fprintf(stderr, "Found key %d value '%s' but not expected!\n", key, value);
	    problems += 1;
	    continue;
	}
	expected = *expected_scanner++;
	actual = *value;
	if (actual != expected) {
	    fprintf(stderr, "Expected '%c' but got '%c'\n", expected, actual);
	    problems += 1;
	    continue;
	}
    }
    /* final checks */
    if (OPAL_ERROR != rc) {
	fprintf(stderr, "table traversal did not end in OPAL_ERROR?!?\n");
	problems += 1;
    }
    if ('\0' != *expected_scanner) {
	fprintf(stderr, "Still expecting more key/values: '%s'\n", expected_scanner);
	problems += 1;
    }

    /* resolution */
    if (problems > 0) {
	fflush(stderr);
	test_failure("validate_remove_traversal");
    } else {
	test_success();
    }
}
#endif

static void test_ptable(opal_proc_table_t *table)
{
    int j;
    char *n1, *n2;
    char * v;
    int rc;
    opal_process_name_t key;
    fprintf(error_out, "\nTesting integer keys...\n");
    for ( j = 0; num_keys[j]; j += 3)
    {
        opal_process_name_t key;
        key.jobid = atoi(num_keys[j]);
        key.vpid = atoi(num_keys[j+1]);
        opal_proc_table_set_value(table, key, num_keys[j+2]);
    }
    validate_table(table, num_keys);
    rc = opal_proc_table_get_first_key(table, &key, (void **)&v, (void **)&n1, (void **)&n2);
    if (OPAL_SUCCESS != rc) {
        fprintf(error_out, "*** FAILED opal_proc_table_get_first_key ***\n");
        return ;
    }
    for (j=0; num_keys[j]; j+=3) {
        if (OPAL_SUCCESS != rc) {
            fprintf(error_out, "*** FAILED opal_proc_table_get_next_key (%d) ***\n", j/3);
        }

        if (key.jobid != (opal_jobid_t)atoi(num_keys[j]) ||
            key.vpid != (opal_vpid_t)atoi(num_keys[j+1]) ||
            0 != strcmp(num_keys[j+2], v)) {
                fprintf(error_out, "*** FAILED at %d, expected ((%s,%s),%s) got ((%d,%d),%s)\n", j/3,
                        num_keys[j], num_keys[j+1], num_keys[j+2],
                        key.jobid, key.vpid, v);
        }

        rc = opal_proc_table_get_next_key(table, &key, (void **)&v, n1, (void **)&n1, n2, (void **)&n2);
    }
    if (OPAL_SUCCESS == rc) {
        fprintf(error_out, "*** DID NOT FAIL last opal_proc_table_get_next_key %d\n", j/3);
        return;
    }


#if 0
    /* remove all values for next test */
    opal_proc_table_remove_all(table);
    test_verify_int(0, opal_hash_table_get_size(table));

    fprintf(error_out, "\nTesting removal and traversal...\n");
    j = 0;
    char * str;
    while (NULL != (str = remove_keys[j++])) {
	opal_hash_table_set_value_uint32(table, atoi(str), remove_keys[j++]);
    }
    while (NULL != (str = remove_keys[j++])) {
	/* generate collisions */
	opal_hash_table_set_value_uint32(table, atoi(str) + table->ht_capacity, remove_keys[j++]);
    }
    validate_remove_traversal(table, remove_keys[j++]);
    while (NULL != (str = remove_keys[j++])) {
	opal_hash_table_remove_value_uint32(table, atoi(str));
	validate_remove_traversal(table, remove_keys[j++]);
    }

    /* remove all values for next test */
    opal_hash_table_remove_all(table);
    test_verify_int(0, opal_hash_table_get_size(table));
#endif

    fprintf(error_out, "\n\n");
}


static void test_dynamic(void)
{
    opal_proc_table_t     *table;

    table = OBJ_NEW(opal_proc_table_t);
    if ( NULL == table )
    {
        fprintf(error_out, "Error: Unable to create hash table.\n");
        exit(-1);
    }
    fprintf(error_out, "Testing with dynamically created table...\n");
    opal_proc_table_init(table, 2, 4);
    test_ptable(table);

    OBJ_RELEASE(table);
}


static void test_static(void)
{
    opal_proc_table_t     table;

    OBJ_CONSTRUCT(&table, opal_proc_table_t);
    opal_proc_table_init(&table, 8, 128);

    fprintf(error_out, "Testing with statically created table...\n");
    test_ptable(&table);

    OBJ_DESTRUCT(&table);
}


int main(int argc, char **argv)
{
    int rc;

    test_init("opal_proc_table_t");

    rc = opal_init_util(&argc, &argv);
    test_verify_int(OPAL_SUCCESS, rc);
    if (OPAL_SUCCESS != rc) {
        test_finalize();
        exit(1);
    }

#ifdef STANDALONE
    error_out = stderr;
#else
    error_out = fopen( "./opal_proc_table_test_out.txt", "w" );
    if( error_out == NULL ) error_out = stderr;
#endif

    test_dynamic();
    test_static();
#ifndef STANDALONE
    fclose( error_out );
#endif

    opal_finalize_util ();

    return test_finalize();
}
