/**
 * Multithreaded work queue.
 * Copyright (c) 2012 Ronald Bennett Cemer
 * This software is licensed under the BSD license.
 * See the accompanying LICENSE.txt for details.
 */

#ifndef WORKQUEUE_H
#define WORKQUEUE_H

#include "lwos.h"

#include <queue>

/* Handle to a worker thread */
typedef struct worker {
	LWOSthread* thread; /* Our thread handle */
} worker_t;

/* A single item to be processed from the work queue */
typedef struct job {
	void (*job_function)(struct job *job);  /* Callback to call for this job */
	void *user_data;						/* Job context pointer to pass to job_function */
} job_t;

/* The parent structure of the work queue */
typedef struct workqueue {
	int terminate;                 /* Should all of the work threads exit? 1=yes. 0=no */
	std::queue<job_t *>jobs_queue; /* Actual queue of work items */
	LWOSCriticalSection jobs_mutex;/* Mutex to control access to jobs_queue and jobs_cond */
	lwosCV jobs_cond;              /* Condition to signal to wake up workers */
	int numWorkers;			       /* Number of valid entries in workers[] */
	worker_t *workers;        	   /* Array of worker threads of size numWorkers */
} workqueue_t;

extern 
int workqueue_init(workqueue_t *workqueue, int numWorkers);

void workqueue_shutdown(workqueue_t *workqueue);

void workqueue_add_job(workqueue_t *workqueue, job_t *job);

#endif	/* #ifndef WORKQUEUE_H */
