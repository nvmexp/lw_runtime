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
#include "fm_log.h"

static int worker_function(void *ptr) 
{
	workqueue_t *workqueue = (workqueue_t *)ptr;
	job_t *job;

	while (1) 
	{
		/* Wait until we get notified. */
		lwosEnterCriticalSection(&workqueue->jobs_mutex);

		while (workqueue->jobs_queue.size() < 1) 
		{
			/* If we're supposed to terminate, break out of our continuous loop. */
			if (workqueue->terminate)
			{
				FM_LOG_DEBUG("Was asked to terminate. Quitting.");
				break;
			}

			lwosCondWait(&workqueue->jobs_cond, &workqueue->jobs_mutex, 1000);
		}

		/* If we're supposed to terminate, break out of our continuous loop. */
		if (workqueue->terminate)
        {
            /* Don't leave the mutex locked */
            lwosLeaveCriticalSection(&workqueue->jobs_mutex);
            FM_LOG_DEBUG("Quitting");
			break;
        }
		
		/* Get a pointer to the first element and remove it */
		job = workqueue->jobs_queue.front();
		workqueue->jobs_queue.pop();

		/* Unlock the mutex since we got an item */
        lwosLeaveCriticalSection(&workqueue->jobs_mutex);

		/* Execute the job. */
		job->job_function(job);
	}
    
    return 0;
}

int workqueue_init(workqueue_t *workqueue, int numWorkers) 
{
	int i;
    
    if (numWorkers < 1) 
		numWorkers = 1;
    workqueue->terminate = 0;
    
    lwosInitializeCriticalSection(&workqueue->jobs_mutex);
    lwosCondCreate(&workqueue->jobs_cond);

	workqueue->numWorkers = numWorkers;
	size_t memSize = sizeof(worker_t) * numWorkers;
	workqueue->workers = (worker_t *)malloc(memSize);
	if(!workqueue->workers)
	{
		FM_LOG_ERROR("unable to allocate required number of worker queues.");
		return 1;
	}
	memset(workqueue->workers, 0, memSize);

	for (i = 0; i < numWorkers; i++) 
	{
		if (lwosThreadCreate(&workqueue->workers[i].thread, worker_function, (void *)workqueue)) 
		{
			FM_LOG_ERROR("failed to start all worker threads for processing");
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
    FM_LOG_DEBUG("Telling workers to terminate");
	workqueue->terminate = 1;

	/* Remove all workers and jobs from the work queue.
	 * wake up all workers so that they will terminate. */
	lwosEnterCriticalSection(&workqueue->jobs_mutex);
	lwosCondBroadcast(&workqueue->jobs_cond);
	lwosLeaveCriticalSection(&workqueue->jobs_mutex);

	/* Wait for all of the workers to exit before we delete the workers[] array from under them */
	for(i = 0; i < workqueue->numWorkers; i++)
	{
		FM_LOG_DEBUG("wating for thread %d/%d to exit", i, workqueue->numWorkers);;
		lwosThreadJoin(workqueue->workers[i].thread, NULL);
	}

	FM_LOG_DEBUG("All workers joined.");

	if(workqueue->workers)
	{
		free(workqueue->workers);
		workqueue->workers = NULL;
	}
    
    lwosDeleteCriticalSection(&workqueue->jobs_mutex);
    lwosCondDestroy(&workqueue->jobs_cond);
}

void workqueue_add_job(workqueue_t *workqueue, job_t *job) 
{
	/* Add the job to the job queue, and notify a worker. */
	lwosEnterCriticalSection(&workqueue->jobs_mutex);
	workqueue->jobs_queue.push(job);
	lwosCondSignal(&workqueue->jobs_cond);
	lwosLeaveCriticalSection(&workqueue->jobs_mutex);
}
