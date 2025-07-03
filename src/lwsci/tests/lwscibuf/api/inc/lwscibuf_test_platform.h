/*
 * Copyright (c) 2021, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

/* This file exists to abstract away platform-specific logic in the integration
 * tests.
 *
 * Lwrrently the liblwos.so library isn't available on Desktop, which makes any
 * use of LwOS APIs diffilwlt. The lwos.h header _is available_ but the
 * implementation is completely different. See lwbugs/1779676 for an attempt
 * and some dislwssion at unifying them.
 */

/* A fork handler intended to be called by the forked child.
 * This initializes any test resources in the forked process
 */
void TestLwSciBufPlatformForkHandler(void);
