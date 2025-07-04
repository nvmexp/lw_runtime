/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright(c) 2018 Intel Corporation
 */

#ifndef PARSE_H_
#define PARSE_H_

#ifdef __cplusplus
extern "C" {
#endif

int
parse_set(const char *, uint16_t [], unsigned int);

int
parse_branch_ratio(const char *input, float *branch_ratio);

#ifdef __cplusplus
}
#endif


#endif /* PARSE_H_ */
