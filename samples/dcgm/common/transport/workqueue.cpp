/**
 * Multithreaded work queue.
 * Copyright (c) 2012 Ronald Bennett Cemer
 * This software is licensed under the BSD license.
 * See the accompanying LICENSE.txt for details.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "workqueue.h"
#include "logging.h"

static void *worker_function(void *ptr) 
{
	workqueue_t *workqueue = (workqueue_t *)ptr;
	job_t *job;

	while (1) 
	{
		/* Wait until we get notified. */
		pthread_mutex_lock(&workqueue->jobs_mutex);
		while (workqueue->jobs_queue.size() < 1) 
		{
			/* If we're supposed to terminate, break out of our continuous loop. */
			if (workqueue->terminate)
			{
				PRINT_DEBUG("", "Was asked to terminate. Quitting.");
				break;
			}

			pthread_cond_wait(&workqueue->jobs_cond, &workqueue->jobs_mutex);
		}

		/* If we're supposed to terminate, break out of our continuous loop. */
		if (workqueue->terminate)
        {
            /* Don't leave the mutex locked */
            pthread_mutex_unlock(&workqueue->jobs_mutex);
            PRINT_DEBUG("", "Quitting");
			break;
        }
		
		/* Get a pointer to the first element and remove it */
		job = workqueue->jobs_queue.front();
		workqueue->jobs_queue.pop();

		/* Unlock the mutex since we got an item */
		pthread_mutex_unlock(&workqueue->jobs_mutex);

		/* Execute the job. */
		job->job_function(job);
	}

	pthread_exit(NULL);
}

int workqueue_init(workqueue_t *workqueue, int numWorkers) 
{
	int i;
	pthread_cond_t blank_cond = PTHREAD_COND_INITIALIZER;
	pthread_mutex_t blank_mutex = PTHREAD_MUTEX_INITIALIZER;

	if (numWorkers < 1) 
		numWorkers = 1;
	
	/* Don't memset workqueue since it has templated class members */
	workqueue->terminate = 0;
	memcpy(&workqueue->jobs_mutex, &blank_mutex, sizeof(workqueue->jobs_mutex));
	memcpy(&workqueue->jobs_cond, &blank_cond, sizeof(workqueue->jobs_cond));

	workqueue->numWorkers = numWorkers;
	size_t memSize = sizeof(worker_t) * numWorkers;
	workqueue->workers = (worker_t *)malloc(memSize);
	if(!workqueue->workers)
	{
		PRINT_ERROR("", "Unable to alloc workers.");
		return 1;
	}
	memset(workqueue->workers, 0, memSize);

	for (i = 0; i < numWorkers; i++) 
	{	
		if (pthread_create(&workqueue->workers[i].thread, NULL, worker_function, (void *)workqueue)) 
		{
			PRINT_ERROR("", "Failed to start all worker threads");
			free(workqueue->workers);
			return 1;
		}
	}

	return 0;
}

void workqueue_shutdown(workqueue_t *workqueue) 
{
	int i;

	/* Set all workers to terminate. */
    PRINT_DEBUG("", "Telling workers to terminate");
	workqueue->terminate = 1;

	/* Remove all workers and jobs from the work queue.
	 * wake up all workers so that they will terminate. */
	pthread_mutex_lock(&workqueue->jobs_mutex);
	pthread_cond_broadcast(&workqueue->jobs_cond);
	pthread_mutex_unlock(&workqueue->jobs_mutex);

	/* Wait for all of the workers to exit before we delete the workers[] array from under them */
	for(i = 0; i < workqueue->numWorkers; i++)
	{
		PRINT_DEBUG("%d %d", "wating for thread %d/%d to exit", i, workqueue->numWorkers);;
		pthread_join(workqueue->workers[i].thread, NULL);
	}

	PRINT_DEBUG("", "All workers joined.");

	if(workqueue->workers)
	{
		free(workqueue->workers);
		workqueue->workers = NULL;
	}
}

void workqueue_add_job(workqueue_t *workqueue, job_t *job) 
{
	/* Add the job to the job queue, and notify a worker. */
	pthread_mutex_lock(&workqueue->jobs_mutex);
	workqueue->jobs_queue.push(job);
	pthread_cond_signal(&workqueue->jobs_cond);
	pthread_mutex_unlock(&workqueue->jobs_mutex);
}
