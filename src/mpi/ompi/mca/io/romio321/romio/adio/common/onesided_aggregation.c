/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 *  (C) 2015 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#include "adio.h"
#include "adio_extern.h"
#ifdef ROMIO_GPFS
/* right now this is GPFS only but TODO: extend this to all file systems */
#include "../ad_gpfs/ad_gpfs_tuning.h"
#else
int gpfsmpio_onesided_no_rmw = 0;
int gpfsmpio_write_aggmethod = 0;
int gpfsmpio_read_aggmethod = 0;
int gpfsmpio_onesided_always_rmw = 0;
#endif

#include <pthread.h>

//  #define onesidedtrace 1


/* This data structure holds the number of extents, the index into the flattened buffer and the remnant length
 * beyond the flattened buffer index corresponding to the base buffer offset for non-contiguous source data
 * for the range to be written coresponding to the round and target agg.
 */
typedef struct NonContigSourceBufOffset {
  int dataTypeExtent;
  int flatBufIndice;
  ADIO_Offset indiceOffset;
} NonContigSourceBufOffset;

/* This data structure holds the access state of the source buffer for target
 * file domains within aggregators corresponding to the target data blocks.  It
 * is designed to be initialized with a starting point for a given file domain
 * with an aggregator, after which the data access for data written to a given
 * file domain from this compute is linear and uninterupted, and this serves as
 * a key optimization for feeding the target aggs.  For contigous source data
 * the starting point is a single-value offset, for non-contiguous data it is
 * the number of extents, the index into the flattened buffer and the remnant
 * length beyond the flattened buffer index.  The validity of the usage of this
 * structure relies on the requirement that only 1 aggregator can write to a
 * given file domain.  */
typedef struct FDSourceBufferState {

    ADIO_Offset indiceOffset;
    MPI_Aint bufTypeExtent;
    int dataTypeExtent;
    int flatBufIndice;

    ADIO_Offset sourceBufferOffset;

} FDSourceBufferState;


static int ADIOI_OneSidedSetup(ADIO_File fd, int procs) {
    int ret = MPI_SUCCESS;

    ret = MPI_Win_create(fd->io_buf,fd->hints->cb_buffer_size,1,
	    MPI_INFO_NULL,fd->comm, &fd->io_buf_window);
    if (ret != MPI_SUCCESS) goto fn_exit;
    fd->io_buf_put_amounts = (int *) ADIOI_Malloc(procs*sizeof(int));
    ret =MPI_Win_create(fd->io_buf_put_amounts,procs*sizeof(int),sizeof(int),
	    MPI_INFO_NULL,fd->comm, &fd->io_buf_put_amounts_window);
fn_exit:
    return ret;
}

int ADIOI_OneSidedCleanup(ADIO_File fd)
{
    int ret = MPI_SUCCESS;
    if (fd->io_buf_window != MPI_WIN_NULL)
	ret = MPI_Win_free(&fd->io_buf_window);
    if (fd->io_buf_put_amounts_window != MPI_WIN_NULL)
	ret = MPI_Win_free(&fd->io_buf_put_amounts_window);
    if (fd->io_buf_put_amounts != NULL)
	ADIOI_Free(fd->io_buf_put_amounts);

    return ret;
}

/* This funtion packs a contigous buffer of data from the non-contgious source
 * buffer for a specified chunk of data and advances the FDSourceBufferState
 * machinery, so subsequent calls with the FDSourceBufferState will return the
 * next linear chunk.
 * Parameters:
 * in:     sourceDataBuffer - pointer to source data buffer.
 * in:    flatBuf - pointer to flattened source data buffer
 * in:     targetNumBytes - number of bytes to return and advance.
 * in:     packing - whether data is being packed from the source buffer to the
 *         packed buffer (1) or unpacked from the packed buffer to the source
 *         buffer (0)
 * in/out: lwrrentFDSourceBufferState - pointer to FDSourceBufferState structure, current
 *                                      data used as starting point, will be updated with
 *                                      the new state after targetNumBytes advance.
 * out:    packedDataBufer - pointer to the output packed data buffer.  If the
 *                           value is NULL then no data will be written.
 *
 */
inline static void nonContigSourceDataBufferAdvance(char *sourceDataBuffer,
	ADIOI_Flatlist_node *flatBuf, int targetNumBytes, int packing,
	FDSourceBufferState *lwrrentFDSourceBufferState, char *packedDataBufer)
{
    // make lwrrentDataTypeExtent and bufTypeExtent ADIO_Offset since they are
    // used in offset callwlations
    ADIO_Offset lwrrentIndiceOffset = lwrrentFDSourceBufferState->indiceOffset;
    ADIO_Offset bufTypeExtent = (ADIO_Offset)lwrrentFDSourceBufferState->bufTypeExtent;
    ADIO_Offset lwrrentDataTypeExtent =
	(ADIO_Offset)lwrrentFDSourceBufferState->dataTypeExtent;
    int lwrrentFlatBufIndice = lwrrentFDSourceBufferState->flatBufIndice;

    int targetSendDataIndex = 0;

#ifdef onesidedtrace
    printf("nonContigSourceDataBufferAdvance: lwrrentFlatBufIndice is %d lwrrentDataTypeExtent is %ld lwrrentIndiceOffset is %ld\n",lwrrentFlatBufIndice,lwrrentDataTypeExtent,lwrrentIndiceOffset);
#endif

    int remainingBytesToLoad = targetNumBytes;
    while (remainingBytesToLoad > 0) {
      if ((flatBuf->blocklens[lwrrentFlatBufIndice] - lwrrentIndiceOffset) >= remainingBytesToLoad) { // we can get the rest of our data from this indice
        ADIO_Offset physicalSourceBufferOffset = (lwrrentDataTypeExtent * bufTypeExtent) + flatBuf->indices[lwrrentFlatBufIndice] + lwrrentIndiceOffset;

#ifdef onesidedtrace
        printf("loading remainingBytesToLoad %d from src buffer offset %ld to targetSendDataIndex %d\n",remainingBytesToLoad,physicalSourceBufferOffset,targetSendDataIndex);
#endif

        if (packedDataBufer != NULL) {
        if (packing)
          memcpy(&(packedDataBufer[targetSendDataIndex]),&(sourceDataBuffer[physicalSourceBufferOffset]),remainingBytesToLoad);
        else
          memcpy(&(sourceDataBuffer[physicalSourceBufferOffset]),&(packedDataBufer[targetSendDataIndex]),remainingBytesToLoad);
        }

        targetSendDataIndex += remainingBytesToLoad;
        lwrrentIndiceOffset += (ADIO_Offset)remainingBytesToLoad;
        if (lwrrentIndiceOffset >= flatBuf->blocklens[lwrrentFlatBufIndice]) {
          lwrrentIndiceOffset = (ADIO_Offset)0;
          lwrrentFlatBufIndice++;
          if (lwrrentFlatBufIndice == flatBuf->count) {
            lwrrentFlatBufIndice = 0;
            lwrrentDataTypeExtent++;
          }
        }
        remainingBytesToLoad = 0;

      }
      else { // we can only get part of our data from this indice
        int amountDataToLoad = (flatBuf->blocklens[lwrrentFlatBufIndice] - lwrrentIndiceOffset);
        ADIO_Offset physicalSourceBufferOffset = (lwrrentDataTypeExtent * bufTypeExtent) + flatBuf->indices[lwrrentFlatBufIndice] + lwrrentIndiceOffset;

#ifdef onesidedtrace
        printf("loading amountDataToLoad %d from src buffer offset %ld to targetSendDataIndex %d\n",amountDataToLoad,physicalSourceBufferOffset,targetSendDataIndex);
#endif
        if (packedDataBufer != NULL) {
        if (packing)
            memcpy(&(packedDataBufer[targetSendDataIndex]),&(sourceDataBuffer[physicalSourceBufferOffset]),amountDataToLoad);
        else
          memcpy(&(sourceDataBuffer[physicalSourceBufferOffset]),&(packedDataBufer[targetSendDataIndex]),amountDataToLoad);
        }

        targetSendDataIndex += amountDataToLoad;
        lwrrentIndiceOffset = (ADIO_Offset)0;
        lwrrentFlatBufIndice++;
        if (lwrrentFlatBufIndice == flatBuf->count) {
          lwrrentFlatBufIndice = 0;
          lwrrentDataTypeExtent++;
        }
        remainingBytesToLoad -= amountDataToLoad;
      }
    } // while

    /* update machinery with new flatbuf position
     */
    lwrrentFDSourceBufferState->indiceOffset = lwrrentIndiceOffset;
    lwrrentFDSourceBufferState->dataTypeExtent = (int) lwrrentDataTypeExtent;
    lwrrentFDSourceBufferState->flatBufIndice = lwrrentFlatBufIndice;
#ifdef onesidedtrace
    printf("source buf advanced to lwrrentFlatBufIndice %d lwrrentDataTypeExtent %ld lwrrentIndiceOffset %ld\n",lwrrentFlatBufIndice,lwrrentDataTypeExtent,lwrrentIndiceOffset);
#endif
}


void ADIOI_OneSidedWriteAggregation(ADIO_File fd,
    ADIO_Offset *offset_list,
    ADIO_Offset *len_list,
    int contig_access_count,
    const void *buf,
    MPI_Datatype datatype,
    int *error_code,
    ADIO_Offset *st_offsets,
    ADIO_Offset *end_offsets,
    int numNonZeroDataOffsets,
    ADIO_Offset *fd_start,
    ADIO_Offset* fd_end,
    int *hole_found)

{
    int i,j; /* generic iterators */

#ifdef onesidedtrace
    if (buf == NULL) {
      printf("ADIOI_OneSidedWriteAggregation - buf is NULL contig_access_count is %d\n",contig_access_count);
      for (i=0;i<contig_access_count;i++)
        printf("offset_list[%d] is %ld len_list[%d] is %ld\n",
		i,offset_list[i],i,len_list[i]);
    }
    if (contig_access_count < 0)
      printf("ADIOI_OneSidedWriteAggregation - contig_access_count "
	      "of %d is less than 0\n",contig_access_count);
#endif

    int lenListOverZero = 0;
    for (i=0;((i<contig_access_count) && (!lenListOverZero));i++)
      if (len_list[i] > 0)
        lenListOverZero = 1;


    *error_code = MPI_SUCCESS; /* initialize to success */

#ifdef ROMIO_GPFS
    double startTimeBase,endTimeBase;
    startTimeBase = MPI_Wtime();
#endif

    MPI_Status status;
    pthread_t io_thread;
    void *thread_ret;
    ADIOI_IO_ThreadFuncData io_thread_args;

    int nprocs,myrank;
    MPI_Comm_size(fd->comm, &nprocs);
    MPI_Comm_rank(fd->comm, &myrank);
#ifdef onesidedtrace
printf("ADIOI_OneSidedWriteAggregation started on rank %d\n",myrank);
#endif


    if (fd->io_buf_window == MPI_WIN_NULL ||
	    fd->io_buf_put_amounts_window == MPI_WIN_NULL)
    {
	ADIOI_OneSidedSetup(fd, nprocs);
    }

    /* This flag denotes whether the source datatype is contiguous, which is referenced throughout the algorithm
     * and defines how the source buffer offsets and data chunks are determined.  If the value is 1 (true - contiguous data)
     * things are profoundly simpler in that the source buffer offset for a given target offset simply linearly increases
     * by the chunk sizes being written.  If the value is 0 (non-contiguous) then these values are based on callwlations
     * from the flattened source datatype.
     */
    int bufTypeIsContig;

    MPI_Aint bufTypeExtent, lb;
    ADIOI_Flatlist_node *flatBuf=NULL;
    ADIOI_Datatype_iscontig(datatype, &bufTypeIsContig);

    if (!bufTypeIsContig) {
   /* Flatten the non-contiguous source datatype and set the extent. */
      flatBuf = ADIOI_Flatten_and_find(datatype);
      MPI_Type_get_extent(datatype, &lb, &bufTypeExtent);
#ifdef onesidedtrace
      printf("flatBuf->count is %d bufTypeExtent is %d\n", flatBuf->count,bufTypeExtent);
      for (i=0;i<flatBuf->count;i++)
        printf("flatBuf->blocklens[%d] is %d flatBuf->indices[%d] is %ld\n",i,flatBuf->blocklens[i],i,flatBuf->indices[i]);
#endif
    }

    int naggs = fd->hints->cb_nodes;

    /* Track the state of the source buffer for feeding the target data blocks.
     * For GPFS the number of file domains per agg is always 1 so we just need 1 agg
     * dimension to track the data, in the case of lustre we will need 2 dimensions
     * agg and file domain since aggs write to multiple file domains in the case of lustre.
     * This structure will be modified as the data is written to reflect the current state
     * of the offset.
     */

#ifdef onesidedtrace
    printf("sizeof(FDSourceBufferState) is %d - make sure is 32 for 32-byte memalign optimal\n",sizeof(FDSourceBufferState));
#endif
    FDSourceBufferState *lwrrentFDSourceBufferState;

    lwrrentFDSourceBufferState = (FDSourceBufferState *) ADIOI_Malloc(naggs * sizeof(FDSourceBufferState));
    for (i=0;i<naggs;i++) {
      /* initialize based on the bufType to indicate that it is unset.
       */
      if (bufTypeIsContig) {
        lwrrentFDSourceBufferState[i].sourceBufferOffset = -1;
      }
      else {
        lwrrentFDSourceBufferState[i].indiceOffset = -1;
      }
    }

#ifdef onesidedtrace
    printf(" ADIOI_OneSidedWriteAggregation bufTypeIsContig is %d contig_access_count is %d\n",bufTypeIsContig,contig_access_count);
#endif

    /* maxNumContigOperations keeps track of how many different chunks we will need to send
     * for the purpose of pre-allocating the data structures to hold them.
     */
    int maxNumContigOperations = contig_access_count;

    ADIO_Offset lastFileOffset = 0, firstFileOffset = -1;
    /* Get the total range being written - in the case of just 1 byte the starting and ending offsets
     * will match the same as they would for 0 bytes so to distinguish we need the actual data count.
     */
    for (j=0;j<numNonZeroDataOffsets;j++) {
#ifdef onesidedtrace
printf("end_offsets[%d] is %ld st_offsets[%d] is %ld\n",j,end_offsets[j],j,st_offsets[j]);
#endif
        lastFileOffset = ADIOI_MAX(lastFileOffset,end_offsets[j]);
        if (firstFileOffset == -1)
          firstFileOffset = st_offsets[j];
        else
          firstFileOffset = ADIOI_MIN(firstFileOffset,st_offsets[j]);
    }

    int myAggRank = -1; /* if I am an aggregor this is my index into fd->hints->ranklist */
    int iAmUsedAgg = 0; /* whether or not this rank is used as an aggregator. */

    /* Make coll_bufsize an ADIO_Offset since it is used in callwlations with offsets.
     */
    ADIO_Offset coll_bufsize = (ADIO_Offset)(fd->hints->cb_buffer_size);
#ifdef ROMIO_GPFS
    if (gpfsmpio_pthreadio == 1) {
      /* split buffer in half for a kind of double buffering with the threads*/
      coll_bufsize = (ADIO_Offset)(fd->hints->cb_buffer_size/2);
    }
#endif

    /* This logic defines values that are used later to determine what offsets define the portion
     * of the file domain the agg is writing this round.
     */
    int greatestFileDomainAggRank = -1,smallestFileDomainAggRank = -1;
    ADIO_Offset greatestFileDomainOffset = 0;
    ADIO_Offset smallestFileDomainOffset = lastFileOffset;
    for (j=0;j<naggs;j++) {
      if (fd_end[j] > greatestFileDomainOffset) {
        greatestFileDomainOffset = fd_end[j];
        greatestFileDomainAggRank = j;
      }
      if (fd_start[j] < smallestFileDomainOffset) {
        smallestFileDomainOffset = fd_start[j];
        smallestFileDomainAggRank = j;
      }
      if (fd->hints->ranklist[j] == myrank) {
        myAggRank = j;
        if (fd_end[j] > fd_start[j]) {
          iAmUsedAgg = 1;
        }
      }
    }

#ifdef onesidedtrace
    printf("contig_access_count is %d lastFileOffset is %ld firstFileOffset is %ld\n",contig_access_count,lastFileOffset,firstFileOffset);
    for (j=0;j<contig_access_count;j++) {
      printf("offset_list[%d]: %ld , len_list[%d]: %ld\n",j,offset_list[j],j,len_list[j]);

    }
#endif

    /* Compute number of rounds.
     */
    int numberOfRounds = 0;
    for (j=0;j<naggs;j++) {
	  int lwrrentNumberOfRounds = (int)(((fd_end[j] - fd_start[j])+(ADIO_Offset)1)/coll_bufsize);
      if (((ADIO_Offset)lwrrentNumberOfRounds*coll_bufsize) < ((fd_end[j] - fd_start[j])+(ADIO_Offset)1))
        lwrrentNumberOfRounds++;
	  if (lwrrentNumberOfRounds > numberOfRounds)
	    numberOfRounds = lwrrentNumberOfRounds;
    }

    /* Data structures to track what data this compute needs to send to whom.
     * For lustre they will all need another dimension for the file domain.
     */
    int *targetAggsForMyData = (int *)ADIOI_Malloc(naggs * sizeof(int));
    ADIO_Offset *targetAggsForMyDataFDStart = (ADIO_Offset *)ADIOI_Malloc(naggs * sizeof(ADIO_Offset));
    ADIO_Offset *targetAggsForMyDataFDEnd = (ADIO_Offset *)ADIOI_Malloc(naggs * sizeof(ADIO_Offset));
    int numTargetAggs = 0;

    /* This data structure holds the beginning offset and len list index for the range to be written
     * coresponding to the round and target agg.  Initialize to -1 to denote being unset.
     */
    int **targetAggsForMyDataFirstOffLenIndex = (int **)ADIOI_Malloc(numberOfRounds * sizeof(int *));
    for (i=0;i<numberOfRounds;i++) {
      targetAggsForMyDataFirstOffLenIndex[i] = (int *)ADIOI_Malloc(naggs * sizeof(int));
      for (j=0;j<naggs;j++)
        targetAggsForMyDataFirstOffLenIndex[i][j] = -1;
    }

    /* This data structure holds the ending offset and len list index for the range to be written
     * coresponding to the round and target agg.
     */
    int **targetAggsForMyDataLastOffLenIndex = (int **)ADIOI_Malloc(numberOfRounds * sizeof(int *));
    for (i=0;i<numberOfRounds;i++)
      targetAggsForMyDataLastOffLenIndex[i] = (int *)ADIOI_Malloc(naggs * sizeof(int));

#ifdef onesidedtrace
   printf("NumberOfRounds is %d\n",numberOfRounds);
   for (i=0;i<naggs;i++)
     printf("fd->hints->ranklist[%d]is %d fd_start is %ld fd_end is %ld\n",i,fd->hints->ranklist[i],fd_start[i],fd_end[i]);
   for (j=0;j<contig_access_count;j++)
     printf("offset_list[%d] is %ld len_list is %ld\n",j,offset_list[j],len_list[j]);
#endif

    int lwrrentAggRankListIndex = 0;
    int maxNumNonContigSourceChunks = 0;

    ADIO_Offset lwrrentSourceBufferOffset = 0;
    int lwrrentDataTypeExtent = 0;
    int lwrrentFlatBufIndice=0;
    ADIO_Offset lwrrentIndiceOffset = 0;

    /* This denotes the coll_bufsize boundaries within the source buffer for writing for the same round.
     */
    ADIO_Offset intraRoundCollBufsizeOffset = 0;

    /* This data structure tracks what target aggs need to be written to on what rounds.
     */
    int *targetAggsForMyDataLwrrentRoundIter = (int *)ADIOI_Malloc(naggs * sizeof(int));
    for (i=0;i<naggs;i++)
      targetAggsForMyDataLwrrentRoundIter[i] = 0;

    /* This is the first of the two main loops in this algorithm.  The purpose of this loop is essentially to populate
     * the data structures defined above for what source data blocks needs to go where (target agg and file domain) and when
     * (round iter).  For lustre essentially an additional layer of nesting will be required for the multiple file domains
     * within the target agg.
     */
    if ((contig_access_count > 0) && (buf != NULL) && lenListOverZero) {
    int blockIter;
    for (blockIter=0;blockIter<contig_access_count;blockIter++) {

      /* Determine the starting source buffer offset for this block - for iter 0 skip it since that value is 0.
       */
      if (blockIter>0) {
        if (bufTypeIsContig) {
          lwrrentSourceBufferOffset += len_list[blockIter-1];
        }
        else {

          /* Non-contiguous source datatype, count up the extents and indices to this point
           * in the blocks for use in computing the source starting buffer offset for target aggs
           * and file domains.
           */
          ADIO_Offset sourceBlockTotal = 0;
          int lastIndiceUsed = lwrrentFlatBufIndice;
          int numNonContigSourceChunks = 0;

          while (sourceBlockTotal < len_list[blockIter-1]) {
            numNonContigSourceChunks++;
            sourceBlockTotal += (flatBuf->blocklens[lwrrentFlatBufIndice] - lwrrentIndiceOffset);
            lastIndiceUsed = lwrrentFlatBufIndice;
            lwrrentFlatBufIndice++;
            if (lwrrentFlatBufIndice == flatBuf->count) {
              lwrrentFlatBufIndice = 0;
              lwrrentDataTypeExtent++;
            }
            lwrrentIndiceOffset = (ADIO_Offset)0;
          }
          if (sourceBlockTotal > len_list[blockIter-1]) {
            lwrrentFlatBufIndice--;
            if (lwrrentFlatBufIndice < 0 ) {
              lwrrentDataTypeExtent--;
              lwrrentFlatBufIndice = flatBuf->count-1;
            }
            lwrrentIndiceOffset =  len_list[blockIter-1] - (sourceBlockTotal - flatBuf->blocklens[lastIndiceUsed]);
            // ADIOI_Assert((lwrrentIndiceOffset >= 0) && (lwrrentIndiceOffset < flatBuf->blocklens[lwrrentFlatBufIndice]));
          }
          else
            lwrrentIndiceOffset = (ADIO_Offset)0;
          maxNumContigOperations += (numNonContigSourceChunks+2);
          if (numNonContigSourceChunks > maxNumNonContigSourceChunks)
            maxNumNonContigSourceChunks = numNonContigSourceChunks;

#ifdef onesidedtrace
          printf("blockiter %d lwrrentFlatBufIndice is now %d lwrrentDataTypeExtent is now %d lwrrentIndiceOffset is now %ld maxNumContigOperations is now %d\n",blockIter,lwrrentFlatBufIndice,lwrrentDataTypeExtent,lwrrentIndiceOffset,maxNumContigOperations);
#endif
        } // !bufTypeIsContig
      } // blockIter > 0

      /* For the last iteration we need to include these maxNumContigOperations and maxNumNonContigSourceChunks
       * for non-contig case even though we did not need to compute the next starting offset.
       */
      if ((blockIter == (contig_access_count-1)) && (!bufTypeIsContig)) {
        ADIO_Offset sourceBlockTotal = 0;
        int tmpLwrrentFlatBufIndice = lwrrentFlatBufIndice;
        int  lastNumNonContigSourceChunks = 0;
        while (sourceBlockTotal < len_list[blockIter]) {
          lastNumNonContigSourceChunks++;
          sourceBlockTotal += flatBuf->blocklens[tmpLwrrentFlatBufIndice];
          tmpLwrrentFlatBufIndice++;
          if (tmpLwrrentFlatBufIndice == flatBuf->count) {
            tmpLwrrentFlatBufIndice = 0;
          }
        }
        maxNumContigOperations += (lastNumNonContigSourceChunks+2);
        if (lastNumNonContigSourceChunks > maxNumNonContigSourceChunks)
          maxNumNonContigSourceChunks = lastNumNonContigSourceChunks;

      }

      ADIO_Offset blockStart = offset_list[blockIter], blockEnd = offset_list[blockIter]+len_list[blockIter]-(ADIO_Offset)1;

      /* Find the starting target agg for this block - normally it will be the current agg so guard the expensive
       * while loop with a cheap if-check which for large numbers of small blocks will usually be false.
       */
      if (!((blockStart >= fd_start[lwrrentAggRankListIndex]) && (blockStart <= fd_end[lwrrentAggRankListIndex]))) {
        while (!((blockStart >= fd_start[lwrrentAggRankListIndex]) && (blockStart <= fd_end[lwrrentAggRankListIndex])))
          lwrrentAggRankListIndex++;
      };

#ifdef onesidedtrace
      printf("lwrrentAggRankListIndex is %d blockStart %ld blockEnd %ld fd_start[lwrrentAggRankListIndex] %ld fd_end[lwrrentAggRankListIndex] %ld\n",lwrrentAggRankListIndex,blockStart,blockEnd,fd_start[lwrrentAggRankListIndex],fd_end[lwrrentAggRankListIndex]);
#endif

      /* Determine if this is a new target agg.
       */
      if (blockIter>0) {
        if ((offset_list[blockIter-1]+len_list[blockIter-1]-(ADIO_Offset)1) < fd_start[lwrrentAggRankListIndex]) {
          numTargetAggs++;
        }
      }

       /* Determine which round to start writing - data is written coll_bufsize per round from the aggregator
        * so if our starting offset in the file domain is multiple coll_bufsize that will correspond to the round.
        */
      if ((blockStart - fd_start[lwrrentAggRankListIndex]) >= coll_bufsize) {
        ADIO_Offset lwrrentRoundBlockStart = fd_start[lwrrentAggRankListIndex];
        int startingRound = 0;
        while (blockStart > (lwrrentRoundBlockStart + coll_bufsize - (ADIO_Offset)1)) {
          lwrrentRoundBlockStart+=coll_bufsize;
          startingRound++;
        }
        targetAggsForMyDataLwrrentRoundIter[numTargetAggs] = startingRound;
      }

      /* Initialize the data structures if this is the first offset in the round/target agg.
       */
      if (targetAggsForMyDataFirstOffLenIndex[targetAggsForMyDataLwrrentRoundIter[numTargetAggs]][numTargetAggs] == -1) {
        targetAggsForMyData[numTargetAggs] = fd->hints->ranklist[lwrrentAggRankListIndex];
        targetAggsForMyDataFDStart[numTargetAggs] = fd_start[lwrrentAggRankListIndex];
        /* Round up file domain to the first actual offset used if this is the first file domain.
         */
        if (lwrrentAggRankListIndex == smallestFileDomainAggRank) {
          if (targetAggsForMyDataFDStart[numTargetAggs] < firstFileOffset)
            targetAggsForMyDataFDStart[numTargetAggs] = firstFileOffset;
        }
        targetAggsForMyDataFDEnd[numTargetAggs] = fd_end[lwrrentAggRankListIndex];
        /* Round down file domain to the last actual offset used if this is the last file domain.
         */
        if (lwrrentAggRankListIndex == greatestFileDomainAggRank) {
          if (targetAggsForMyDataFDEnd[numTargetAggs] > lastFileOffset)
            targetAggsForMyDataFDEnd[numTargetAggs] = lastFileOffset;
        }
        targetAggsForMyDataFirstOffLenIndex[targetAggsForMyDataLwrrentRoundIter[numTargetAggs]][numTargetAggs] = blockIter;
	/* Set the source buffer state starting point for data access for this
           agg and file domain.  */

        if (bufTypeIsContig) {
          if (lwrrentFDSourceBufferState[numTargetAggs].sourceBufferOffset == -1) {

            lwrrentFDSourceBufferState[numTargetAggs].sourceBufferOffset = lwrrentSourceBufferOffset;
#ifdef onesidedtrace
            printf("For agg %d sourceBufferOffset initialized to %ld\n",lwrrentAggRankListIndex,lwrrentSourceBufferOffset);
#endif
          }
        }
        else {
          if (lwrrentFDSourceBufferState[numTargetAggs].indiceOffset == -1) {
            lwrrentFDSourceBufferState[numTargetAggs].indiceOffset = lwrrentIndiceOffset;
            lwrrentFDSourceBufferState[numTargetAggs].bufTypeExtent = bufTypeExtent;
            lwrrentFDSourceBufferState[numTargetAggs].dataTypeExtent = lwrrentDataTypeExtent;
            lwrrentFDSourceBufferState[numTargetAggs].flatBufIndice = lwrrentFlatBufIndice;
#ifdef onesidedtrace
            printf("For agg %d dataTypeExtent initialized to %d flatBufIndice to %d indiceOffset to %ld\n",numTargetAggs,lwrrentDataTypeExtent,lwrrentFlatBufIndice,lwrrentIndiceOffset);
#endif
          }
        }

        intraRoundCollBufsizeOffset = fd_start[lwrrentAggRankListIndex] + ((ADIO_Offset)(targetAggsForMyDataLwrrentRoundIter[numTargetAggs]+1) * coll_bufsize);

#ifdef onesidedtrace
        printf("Initial settings numTargetAggs %d offset_list[%d] with value %ld past fd border %ld with len %ld lwrrentSourceBufferOffset set to %ld intraRoundCollBufsizeOffset set to %ld\n",numTargetAggs,blockIter,offset_list[blockIter],fd_start[lwrrentAggRankListIndex],len_list[blockIter],lwrrentSourceBufferOffset,intraRoundCollBufsizeOffset);
#endif
      }

      /* Replace the last offset block iter with this one.
       */
      targetAggsForMyDataLastOffLenIndex[targetAggsForMyDataLwrrentRoundIter[numTargetAggs]][numTargetAggs] = blockIter;

      /* If this blocks extends into the next file domain advance to the next target aggs and source buffer states.
       */
      if (blockEnd > fd_end[lwrrentAggRankListIndex]) {
#ifdef onesidedtrace
      printf("block extends past current fd, blockEnd %ld >= fd_end[lwrrentAggRankListIndex] %ld total block size is %ld blockStart was %ld\n",blockEnd,fd_end[lwrrentAggRankListIndex], len_list[blockIter],blockStart);
#endif
        ADIO_Offset amountToAdvanceSBOffsetForFD = 0;
        int additionalFDCounter = 0;

        while (blockEnd >= fd_end[lwrrentAggRankListIndex]) {
          ADIO_Offset thisAggBlockEnd = fd_end[lwrrentAggRankListIndex];
          if (thisAggBlockEnd >= intraRoundCollBufsizeOffset) {
            while (thisAggBlockEnd >= intraRoundCollBufsizeOffset) {
              targetAggsForMyDataLwrrentRoundIter[numTargetAggs]++;
              intraRoundCollBufsizeOffset += coll_bufsize;
              targetAggsForMyDataFirstOffLenIndex[targetAggsForMyDataLwrrentRoundIter[numTargetAggs]][numTargetAggs] = blockIter;
              targetAggsForMyDataLastOffLenIndex[targetAggsForMyDataLwrrentRoundIter[numTargetAggs]][numTargetAggs] = blockIter;
#ifdef onesidedtrace
              printf("targetAggsForMyDataLwrrentRoundI%d] is now %d intraRoundCollBufsizeOffset is now %ld\n",numTargetAggs,targetAggsForMyDataLwrrentRoundIter[numTargetAggs],intraRoundCollBufsizeOffset);
#endif
            } // while (thisAggBlockEnd >= intraRoundCollBufsizeOffset)
          } // if (thisAggBlockEnd >= intraRoundCollBufsizeOffset)

          int prevAggRankListIndex = lwrrentAggRankListIndex;
          lwrrentAggRankListIndex++;

          /* Skip over unused aggs.
           */
          if (fd_start[lwrrentAggRankListIndex] > fd_end[lwrrentAggRankListIndex]) {
            while (fd_start[lwrrentAggRankListIndex] > fd_end[lwrrentAggRankListIndex])
              lwrrentAggRankListIndex++;

          } // (fd_start[lwrrentAggRankListIndex] > fd_end[lwrrentAggRankListIndex])

          /* Start new target agg.
           */
          if (blockEnd >= fd_start[lwrrentAggRankListIndex]) {
            numTargetAggs++;
            targetAggsForMyData[numTargetAggs] = fd->hints->ranklist[lwrrentAggRankListIndex];
            targetAggsForMyDataFDStart[numTargetAggs] = fd_start[lwrrentAggRankListIndex];
            /* Round up file domain to the first actual offset used if this is the first file domain.
             */
            if (lwrrentAggRankListIndex == smallestFileDomainAggRank) {
              if (targetAggsForMyDataFDStart[numTargetAggs] < firstFileOffset)
                targetAggsForMyDataFDStart[numTargetAggs] = firstFileOffset;
            }
            targetAggsForMyDataFDEnd[numTargetAggs] = fd_end[lwrrentAggRankListIndex];
            /* Round down file domain to the last actual offset used if this is the last file domain.
             */
            if (lwrrentAggRankListIndex == greatestFileDomainAggRank) {
              if (targetAggsForMyDataFDEnd[numTargetAggs] > lastFileOffset)
                targetAggsForMyDataFDEnd[numTargetAggs] = lastFileOffset;
            }
            targetAggsForMyDataFirstOffLenIndex[targetAggsForMyDataLwrrentRoundIter[numTargetAggs]][numTargetAggs] = blockIter;
	    /* For the first additonal file domain the source buffer offset
	     * will be incremented relative to the state of this first main
	     * loop but for subsequent full file domains the offset will be
	     * incremented by the size
             * of the file domain.
             */
            if (additionalFDCounter == 0)
              amountToAdvanceSBOffsetForFD = (fd_end[prevAggRankListIndex]
		      - blockStart) + (ADIO_Offset)1;
            else
              amountToAdvanceSBOffsetForFD = (fd_end[prevAggRankListIndex]
		      -fd_start[prevAggRankListIndex]) +(ADIO_Offset)1;

            if (bufTypeIsContig) {
              ADIOI_Assert(numTargetAggs > 0);
              if (lwrrentFDSourceBufferState[numTargetAggs].sourceBufferOffset == -1) {
                if (additionalFDCounter == 0) { // first file domain, still use the current data counter
                  lwrrentFDSourceBufferState[numTargetAggs].sourceBufferOffset =
		      lwrrentSourceBufferOffset+amountToAdvanceSBOffsetForFD;
                }
                else { // 2nd file domain, advance full file domain from last source buffer state
                  lwrrentFDSourceBufferState[numTargetAggs].sourceBufferOffset =
		      lwrrentFDSourceBufferState[numTargetAggs-1].sourceBufferOffset+amountToAdvanceSBOffsetForFD;
                }

#ifdef onesidedtrace
            printf("Crossed into new FD - for agg %d sourceBufferOffset initialized to %ld amountToAdvanceSBOffsetForFD is %ld\n",numTargetAggs,lwrrentFDSourceBufferState[numTargetAggs].sourceBufferOffset,amountToAdvanceSBOffsetForFD);
#endif
              }
            }
            else if (lwrrentFDSourceBufferState[numTargetAggs].indiceOffset == -1) {
		// non-contiguos source buffer
              ADIOI_Assert(numTargetAggs > 0);

	      /* Initialize the source buffer state appropriately and then
	       * advance it with the
               * nonContigSourceDataBufferAdvance function.
               */
              if (additionalFDCounter == 0) {
		  // first file domain, still use the current data counter
                lwrrentFDSourceBufferState[numTargetAggs].indiceOffset =
		    lwrrentIndiceOffset;
                lwrrentFDSourceBufferState[numTargetAggs].bufTypeExtent = bufTypeExtent;
                lwrrentFDSourceBufferState[numTargetAggs].dataTypeExtent =
		    lwrrentDataTypeExtent;
                lwrrentFDSourceBufferState[numTargetAggs].flatBufIndice =
		    lwrrentFlatBufIndice;
              }
              else {
		  // 2nd file domain, advance full file domain from last source buffer state
                lwrrentFDSourceBufferState[numTargetAggs].indiceOffset =
		    lwrrentFDSourceBufferState[numTargetAggs-1].indiceOffset;
                lwrrentFDSourceBufferState[numTargetAggs].bufTypeExtent =
		    lwrrentFDSourceBufferState[numTargetAggs-1].bufTypeExtent;
                lwrrentFDSourceBufferState[numTargetAggs].dataTypeExtent =
		    lwrrentFDSourceBufferState[numTargetAggs-1].dataTypeExtent;
                lwrrentFDSourceBufferState[numTargetAggs].flatBufIndice =
		    lwrrentFDSourceBufferState[numTargetAggs-1].flatBufIndice;
              }
              nonContigSourceDataBufferAdvance(((char*)buf), flatBuf,
		      (int)amountToAdvanceSBOffsetForFD, 1,
		      &lwrrentFDSourceBufferState[numTargetAggs], NULL);
#ifdef onesidedtrace
              printf("Crossed into new FD - for agg %d dataTypeExtent initialized to %d flatBufIndice to %d indiceOffset to %ld amountToAdvanceSBOffsetForFD is %d\n",numTargetAggs,lwrrentFDSourceBufferState[numTargetAggs].dataTypeExtent,lwrrentFDSourceBufferState[numTargetAggs].flatBufIndice,lwrrentFDSourceBufferState[numTargetAggs].indiceOffset,amountToAdvanceSBOffsetForFD);
#endif
	    }
	    additionalFDCounter++;

#ifdef onesidedtrace
            printf("block extended beyond fd init settings numTargetAggs %d offset_list[%d] with value %ld past fd border %ld with len %ld\n",numTargetAggs,i,offset_list[blockIter],fd_start[lwrrentAggRankListIndex],len_list[blockIter]);
#endif
            intraRoundCollBufsizeOffset = fd_start[lwrrentAggRankListIndex] + coll_bufsize;
            targetAggsForMyDataLastOffLenIndex[targetAggsForMyDataLwrrentRoundIter[numTargetAggs]][numTargetAggs] = blockIter;
          } // if (blockEnd >= fd_start[lwrrentAggRankListIndex])
        } // while (blockEnd >= fd_end[lwrrentAggRankListIndex])
      } // if (blockEnd > fd_end[lwrrentAggRankListIndex])

      /* If we are still in the same file domain / target agg but have gone
       * past the coll_bufsize and need to advance to the next round -
       * initialize tracking data appropriately.
       */
      if (blockEnd >= intraRoundCollBufsizeOffset) {
        ADIO_Offset lwrrentBlockEnd = blockEnd;
        while (lwrrentBlockEnd >= intraRoundCollBufsizeOffset) {
          targetAggsForMyDataLwrrentRoundIter[numTargetAggs]++;
          intraRoundCollBufsizeOffset += coll_bufsize;
          targetAggsForMyDataFirstOffLenIndex[targetAggsForMyDataLwrrentRoundIter[numTargetAggs]][numTargetAggs] = blockIter;
          targetAggsForMyDataLastOffLenIndex[targetAggsForMyDataLwrrentRoundIter[numTargetAggs]][numTargetAggs] = blockIter;
#ifdef onesidedtrace
        printf("smaller than fd lwrrentBlockEnd is now %ld intraRoundCollBufsizeOffset is now %ld targetAggsForMyDataLwrrentRoundIter[%d] is now %d\n",lwrrentBlockEnd, intraRoundCollBufsizeOffset, numTargetAggs,targetAggsForMyDataLwrrentRoundIter[numTargetAggs]);
#endif
        } // while (lwrrentBlockEnd >= intraRoundCollBufsizeOffset)
      } // if (blockEnd >= intraRoundCollBufsizeOffset)

      /* Need to advance numTargetAggs if this is the last target offset to
       * include this one.
       */
      if (blockIter == (contig_access_count-1))
        numTargetAggs++;
    }

#ifdef onesidedtrace
    printf("numTargetAggs is %d\n",numTargetAggs);
    for (i=0;i<numTargetAggs;i++) {
      for (j=0;j<=targetAggsForMyDataLwrrentRoundIter[i];j++)
        printf("targetAggsForMyData[%d] is %d targetAggsForMyDataFDStart[%d] is %ld targetAggsForMyDataFDEnd is %ld targetAggsForMyDataFirstOffLenIndex is %d with value %ld targetAggsForMyDataLastOffLenIndex is %d with value %ld\n",i,targetAggsForMyData[i],i,targetAggsForMyDataFDStart[i],targetAggsForMyDataFDEnd[i],targetAggsForMyDataFirstOffLenIndex[j][i],offset_list[targetAggsForMyDataFirstOffLenIndex[j][i]],targetAggsForMyDataLastOffLenIndex[j][i],offset_list[targetAggsForMyDataLastOffLenIndex[j][i]]);

    }
#endif

    } // if ((contig_access_count > 0) && (buf != NULL) && lenListOverZero)

    ADIOI_Free(targetAggsForMyDataLwrrentRoundIter);

    int lwrrentWriteBuf = 0;
    int useIOBuffer = 0;
#ifdef ROMIO_GPFS
    if (gpfsmpio_pthreadio && (numberOfRounds>1)) {
    useIOBuffer = 1;
    io_thread = pthread_self();
    }
#endif

    /* use the write buffer allocated in the file_open */
    char *write_buf0 = fd->io_buf;
    char *write_buf1 = fd->io_buf + coll_bufsize;

    /* start off pointing to the first buffer. If we use the 2nd buffer (threaded
     * case) we'll swap later */
    char *write_buf = write_buf0;
    MPI_Win write_buf_window = fd->io_buf_window;

    int *write_buf_put_amounts = fd->io_buf_put_amounts;
    if(!gpfsmpio_onesided_no_rmw) {
      *hole_found = 0;
      for (i=0;i<nprocs;i++)
        write_buf_put_amounts[i] = 0;
    }

    /* Counters to track the offset range being written by the used aggs.
     */
    ADIO_Offset lwrrentRoundFDStart = 0;
    ADIO_Offset lwrrentRoundFDEnd = 0;

    if (iAmUsedAgg) {
      lwrrentRoundFDStart = fd_start[myAggRank];
      if (myAggRank == smallestFileDomainAggRank) {
        if (lwrrentRoundFDStart < firstFileOffset)
          lwrrentRoundFDStart = firstFileOffset;
      }
      else if (myAggRank == greatestFileDomainAggRank) {
        if (lwrrentRoundFDEnd > lastFileOffset)
          lwrrentRoundFDEnd = lastFileOffset;
      }
#ifdef onesidedtrace
printf("iAmUsedAgg - lwrrentRoundFDStart initialized to %ld lwrrentRoundFDEnd to %ld\n",lwrrentRoundFDStart,lwrrentRoundFDEnd);
#endif
      if (gpfsmpio_onesided_always_rmw) { // read in the first buffer
        ADIO_Offset tmpLwrrentRoundFDEnd = 0;
        if ((fd_end[myAggRank] - lwrrentRoundFDStart) < coll_bufsize) {
          if (myAggRank == greatestFileDomainAggRank) {
            if (fd_end[myAggRank] > lastFileOffset)
              tmpLwrrentRoundFDEnd = lastFileOffset;
            else
              tmpLwrrentRoundFDEnd = fd_end[myAggRank];
          }
          else
            tmpLwrrentRoundFDEnd = fd_end[myAggRank];
        }
        else
        tmpLwrrentRoundFDEnd = lwrrentRoundFDStart + coll_bufsize - (ADIO_Offset)1;
#ifdef onesidedtrace
printf("gpfsmpio_onesided_always_rmw - first buffer pre-read for file offsets %ld to %ld total is %d\n",lwrrentRoundFDStart,tmpLwrrentRoundFDEnd,(int)(tmpLwrrentRoundFDEnd - lwrrentRoundFDStart)+1);
#endif
        ADIO_ReadContig(fd, write_buf, (int)(tmpLwrrentRoundFDEnd - lwrrentRoundFDStart)+1,
          MPI_BYTE, ADIO_EXPLICIT_OFFSET,lwrrentRoundFDStart, &status, error_code);

      }
    }
    if (gpfsmpio_onesided_always_rmw) // wait until the first buffer is read
      MPI_Barrier(fd->comm);

#ifdef ROMIO_GPFS
    endTimeBase = MPI_Wtime();
    gpfsmpio_prof_cw[GPFSMPIO_CIO_T_DEXCH_SETUP] += (endTimeBase-startTimeBase);
    startTimeBase = MPI_Wtime();
#endif

    /* This is the second main loop of the algorithm, actually nested loop of target aggs within rounds.  There are 2 flavors of this.
     * For gpfsmpio_write_aggmethod of 1 each nested iteration for the target
     * agg does an mpi_put on a contiguous chunk using a primative datatype
     * determined using the data structures from the first main loop.  For
     * gpfsmpio_write_aggmethod of 2 each nested iteration for the target agg
     * builds up data to use in created a derived data type for 1 mpi_put that is done for the target agg for each round.
     * To support lustre there will need to be an additional layer of nesting
     * for the multiple file domains within target aggs.
     */
    int roundIter;

    for (roundIter=0;roundIter<numberOfRounds;roundIter++) {
	if ((contig_access_count > 0) && (buf != NULL) && lenListOverZero) {


    int aggIter;
    for (aggIter=0;aggIter<numTargetAggs;aggIter++) {

    int numBytesPutThisAggRound = 0;
    /* If we have data for the round/agg process it.
     */
    if (targetAggsForMyDataFirstOffLenIndex[roundIter][aggIter] != -1) {
      ADIO_Offset lwrrentRoundFDStartForMyTargetAgg = (ADIO_Offset)((ADIO_Offset)targetAggsForMyDataFDStart[aggIter] + (ADIO_Offset)((ADIO_Offset)roundIter*coll_bufsize));
      ADIO_Offset lwrrentRoundFDEndForMyTargetAgg = (ADIO_Offset)((ADIO_Offset)targetAggsForMyDataFDStart[aggIter] + (ADIO_Offset)((ADIO_Offset)(roundIter+1)*coll_bufsize) - (ADIO_Offset)1);

      int targetAggContigAccessCount = 0;

      /* These data structures are used for the derived datatype mpi_put
       * in the gpfsmpio_write_aggmethod of 2 case.
       */
      int *targetAggBlockLengths=NULL;
      MPI_Aint *targetAggDisplacements=NULL, *sourceBufferDisplacements=NULL;
      MPI_Datatype *targetAggDataTypes=NULL;

      char *derivedTypePackedSourceBuffer=NULL;
      int derivedTypePackedSourceBufferOffset = 0;
      int allocatedDerivedTypeArrays = 0;
      ADIO_Offset amountOfDataWrittenThisRoundAgg = 0;

#ifdef onesidedtrace
      printf("roundIter %d processing targetAggsForMyData %d \n",roundIter,targetAggsForMyData[aggIter]);
#endif

      /* Process the range of offsets for this target agg.
       */
      int offsetIter;
      int startingOffLenIndex = targetAggsForMyDataFirstOffLenIndex[roundIter][aggIter], endingOffLenIndex = targetAggsForMyDataLastOffLenIndex[roundIter][aggIter];
      for (offsetIter=startingOffLenIndex;offsetIter<=endingOffLenIndex;offsetIter++) {
        if (lwrrentRoundFDEndForMyTargetAgg > targetAggsForMyDataFDEnd[aggIter])
            lwrrentRoundFDEndForMyTargetAgg = targetAggsForMyDataFDEnd[aggIter];

        ADIO_Offset offsetStart = offset_list[offsetIter], offsetEnd = (offset_list[offsetIter]+len_list[offsetIter]-(ADIO_Offset)1);

#ifdef onesidedtrace
        printf("roundIter %d target iter %d targetAggsForMyData is %d offset_list[%d] is %ld len_list[%d] is %ld targetAggsForMyDataFDStart is %ld targetAggsForMyDataFDEnd is %ld lwrrentRoundFDStartForMyTargetAgg is %ld lwrrentRoundFDEndForMyTargetAgg is %ld targetAggsForMyDataFirstOffLenIndex is %ld\n",
            roundIter,aggIter,targetAggsForMyData[aggIter],offsetIter,offset_list[offsetIter],offsetIter,len_list[offsetIter],
            targetAggsForMyDataFDStart[aggIter],targetAggsForMyDataFDEnd[aggIter],
            lwrrentRoundFDStartForMyTargetAgg,lwrrentRoundFDEndForMyTargetAgg, targetAggsForMyDataFirstOffLenIndex[roundIter][aggIter]);
#endif

        /* Determine the amount of data and exact source buffer offsets to use.
         */
        int bufferAmountToSend = 0;

        if ((offsetStart >= lwrrentRoundFDStartForMyTargetAgg) && (offsetStart <= lwrrentRoundFDEndForMyTargetAgg)) {
            if (offsetEnd > lwrrentRoundFDEndForMyTargetAgg)
            bufferAmountToSend = (lwrrentRoundFDEndForMyTargetAgg - offsetStart) +1;
            else
            bufferAmountToSend = (offsetEnd - offsetStart) +1;
        }
        else if ((offsetEnd >= lwrrentRoundFDStartForMyTargetAgg) && (offsetEnd <= lwrrentRoundFDEndForMyTargetAgg)) {
            if (offsetEnd > lwrrentRoundFDEndForMyTargetAgg)
            bufferAmountToSend = (lwrrentRoundFDEndForMyTargetAgg - lwrrentRoundFDStartForMyTargetAgg) +1;
            else
            bufferAmountToSend = (offsetEnd - lwrrentRoundFDStartForMyTargetAgg) +1;
            if (offsetStart < lwrrentRoundFDStartForMyTargetAgg) {
              offsetStart = lwrrentRoundFDStartForMyTargetAgg;
            }
        }
        else if ((offsetStart <= lwrrentRoundFDStartForMyTargetAgg) && (offsetEnd >= lwrrentRoundFDEndForMyTargetAgg)) {
            bufferAmountToSend = (lwrrentRoundFDEndForMyTargetAgg - lwrrentRoundFDStartForMyTargetAgg) +1;
            offsetStart = lwrrentRoundFDStartForMyTargetAgg;
        }

        numBytesPutThisAggRound += bufferAmountToSend;
#ifdef onesidedtrace
        printf("bufferAmountToSend is %d\n",bufferAmountToSend);
#endif
        if (bufferAmountToSend > 0) { /* we have data to send this round */
          if (gpfsmpio_write_aggmethod == 2) {
            /* Only allocate these arrays if we are using method 2 and only do it once for this round/target agg.
             */
            if (!allocatedDerivedTypeArrays) {
              targetAggBlockLengths = (int *)ADIOI_Malloc(maxNumContigOperations * sizeof(int));
              targetAggDisplacements = (MPI_Aint *)ADIOI_Malloc(maxNumContigOperations * sizeof(MPI_Aint));
              sourceBufferDisplacements = (MPI_Aint *)ADIOI_Malloc(maxNumContigOperations * sizeof(MPI_Aint));
              targetAggDataTypes = (MPI_Datatype *)ADIOI_Malloc(maxNumContigOperations * sizeof(MPI_Datatype));
              if (!bufTypeIsContig) {
                int k;
                for (k=targetAggsForMyDataFirstOffLenIndex[roundIter][aggIter];k<=targetAggsForMyDataLastOffLenIndex[roundIter][aggIter];k++)
                  amountOfDataWrittenThisRoundAgg += len_list[k];

#ifdef onesidedtrace
                printf("derivedTypePackedSourceBuffer mallocing %ld\n",amountOfDataWrittenThisRoundAgg);
#endif
                if (amountOfDataWrittenThisRoundAgg > 0)
                  derivedTypePackedSourceBuffer = (char *)ADIOI_Malloc(amountOfDataWrittenThisRoundAgg * sizeof(char));
                else
                  derivedTypePackedSourceBuffer = NULL;
              }
              allocatedDerivedTypeArrays = 1;
            }
          }

          /* Determine the offset into the target window.
           */
          MPI_Aint targetDisplacementToUseThisRound = (MPI_Aint) (offsetStart - lwrrentRoundFDStartForMyTargetAgg);

          /* If using the thread writer select the appropriate side of the split window.
           */
          if (useIOBuffer && (write_buf == write_buf1)) {
            targetDisplacementToUseThisRound += (MPI_Aint) coll_bufsize;
          }

          /* For gpfsmpio_write_aggmethod of 1 do the mpi_put using the primitive MPI_BYTE type for each contiguous
           * chunk in the target, of source data is non-contiguous then pack the data first.
           */

          if (gpfsmpio_write_aggmethod == 1) {
            MPI_Win_lock(MPI_LOCK_SHARED, targetAggsForMyData[aggIter], 0, write_buf_window);
            if (bufTypeIsContig) {
              MPI_Put(((char*)buf) + lwrrentFDSourceBufferState[aggIter].sourceBufferOffset,bufferAmountToSend, MPI_BYTE,targetAggsForMyData[aggIter],targetDisplacementToUseThisRound, bufferAmountToSend,MPI_BYTE,write_buf_window);
              lwrrentFDSourceBufferState[aggIter].sourceBufferOffset += (ADIO_Offset)bufferAmountToSend;
            }
            else {
              char *putSourceData = (char *) ADIOI_Malloc(bufferAmountToSend*sizeof(char));
              nonContigSourceDataBufferAdvance(((char*)buf), flatBuf, bufferAmountToSend, 1, &lwrrentFDSourceBufferState[aggIter], putSourceData);
              MPI_Put(putSourceData,bufferAmountToSend, MPI_BYTE,targetAggsForMyData[aggIter],targetDisplacementToUseThisRound, bufferAmountToSend,MPI_BYTE,write_buf_window);
              ADIOI_Free(putSourceData);
            }
            MPI_Win_unlock(targetAggsForMyData[aggIter], write_buf_window);
          }

          /* For gpfsmpio_write_aggmethod of 2 populate the data structures for this round/agg for this offset iter
           * to be used subsequently when building the derived type for 1 mpi_put for all the data for this
           * round/agg.
           */
          else if (gpfsmpio_write_aggmethod == 2) {

            if (bufTypeIsContig) {
              targetAggBlockLengths[targetAggContigAccessCount]= bufferAmountToSend;
              targetAggDataTypes[targetAggContigAccessCount] = MPI_BYTE;
              targetAggDisplacements[targetAggContigAccessCount] = targetDisplacementToUseThisRound;
              sourceBufferDisplacements[targetAggContigAccessCount] = (MPI_Aint)lwrrentFDSourceBufferState[aggIter].sourceBufferOffset;
              lwrrentFDSourceBufferState[aggIter].sourceBufferOffset += (ADIO_Offset)bufferAmountToSend;
              targetAggContigAccessCount++;
            }
            else {
              nonContigSourceDataBufferAdvance(((char*)buf), flatBuf, bufferAmountToSend, 1, &lwrrentFDSourceBufferState[aggIter], &derivedTypePackedSourceBuffer[derivedTypePackedSourceBufferOffset]);
              targetAggBlockLengths[targetAggContigAccessCount]= bufferAmountToSend;
              targetAggDataTypes[targetAggContigAccessCount] = MPI_BYTE;
              targetAggDisplacements[targetAggContigAccessCount] = targetDisplacementToUseThisRound;
              sourceBufferDisplacements[targetAggContigAccessCount] = (MPI_Aint)derivedTypePackedSourceBufferOffset;
              targetAggContigAccessCount++;
              derivedTypePackedSourceBufferOffset += (ADIO_Offset)bufferAmountToSend;
            }
          }
#ifdef onesidedtrace
        printf("roundIter %d bufferAmountToSend is %d offsetStart is %ld lwrrentRoundFDStartForMyTargetAgg is %ld targetDisplacementToUseThisRound is %ld targetAggsForMyDataFDStart[aggIter] is %ld\n",roundIter, bufferAmountToSend, offsetStart,lwrrentRoundFDStartForMyTargetAgg,targetDisplacementToUseThisRound,targetAggsForMyDataFDStart[aggIter]);
#endif

        } // bufferAmountToSend > 0
      } // contig list

      /* For gpfsmpio_write_aggmethod of 2 now build the derived type using the data from this round/agg and do 1 single mpi_put.
       */
      if (gpfsmpio_write_aggmethod == 2) {

        MPI_Datatype sourceBufferDerivedDataType, targetBufferDerivedDataType;
        MPI_Type_create_struct(targetAggContigAccessCount, targetAggBlockLengths, sourceBufferDisplacements, targetAggDataTypes, &sourceBufferDerivedDataType);
        MPI_Type_commit(&sourceBufferDerivedDataType);
        MPI_Type_create_struct(targetAggContigAccessCount, targetAggBlockLengths, targetAggDisplacements, targetAggDataTypes, &targetBufferDerivedDataType);
        MPI_Type_commit(&targetBufferDerivedDataType);

#ifdef onesidedtrace
        printf("mpi_put of derived type to agg %d targetAggContigAccessCount is %d\n",targetAggsForMyData[aggIter],targetAggContigAccessCount);
#endif
        if (targetAggContigAccessCount > 0) {
        MPI_Win_lock(MPI_LOCK_SHARED, targetAggsForMyData[aggIter], 0, write_buf_window);
        if (bufTypeIsContig) {
          MPI_Put(((char*)buf),1, sourceBufferDerivedDataType,targetAggsForMyData[aggIter],0, 1,targetBufferDerivedDataType,write_buf_window);
        }
        else {
          MPI_Put(derivedTypePackedSourceBuffer,1, sourceBufferDerivedDataType,targetAggsForMyData[aggIter],0, 1,targetBufferDerivedDataType,write_buf_window);
        }


        MPI_Win_unlock(targetAggsForMyData[aggIter], write_buf_window);
        }

        if (allocatedDerivedTypeArrays) {
          ADIOI_Free(targetAggBlockLengths);
          ADIOI_Free(targetAggDisplacements);
          ADIOI_Free(targetAggDataTypes);
          ADIOI_Free(sourceBufferDisplacements);
          if (!bufTypeIsContig)
            if (derivedTypePackedSourceBuffer != NULL)
              ADIOI_Free(derivedTypePackedSourceBuffer);
        }
        if (targetAggContigAccessCount > 0) {
        MPI_Type_free(&sourceBufferDerivedDataType);
        MPI_Type_free(&targetBufferDerivedDataType);
        }
      }
      if (!gpfsmpio_onesided_no_rmw) {
        MPI_Win_lock(MPI_LOCK_SHARED, targetAggsForMyData[aggIter], 0, fd->io_buf_put_amounts_window);
        MPI_Put(&numBytesPutThisAggRound,1, MPI_INT,targetAggsForMyData[aggIter],myrank, 1,MPI_INT,fd->io_buf_put_amounts_window);
        MPI_Win_unlock(targetAggsForMyData[aggIter], fd->io_buf_put_amounts_window);
      }
      } // baseoffset != -1
    } // target aggs
	} /// contig_access_count > 0

#ifdef onesidedtrace
printf("first barrier roundIter %d\n",roundIter);
#endif
    MPI_Barrier(fd->comm);

    if (iAmUsedAgg) {
    /* Determine what offsets define the portion of the file domain the agg is writing this round.
     */
        if ((fd_end[myAggRank] - lwrrentRoundFDStart) < coll_bufsize) {
          if (myAggRank == greatestFileDomainAggRank) {
            if (fd_end[myAggRank] > lastFileOffset)
              lwrrentRoundFDEnd = lastFileOffset;
            else
              lwrrentRoundFDEnd = fd_end[myAggRank];
          }
          else
            lwrrentRoundFDEnd = fd_end[myAggRank];
        }
        else
        lwrrentRoundFDEnd = lwrrentRoundFDStart + coll_bufsize - (ADIO_Offset)1;

#ifdef onesidedtrace
        printf("used agg about to writecontig - lwrrentRoundFDStart is %ld lwrrentRoundFDEnd is %ld within file domeain %ld to %ld\n",lwrrentRoundFDStart,lwrrentRoundFDEnd,fd_start[myAggRank],fd_end[myAggRank]);
#endif

        int doWriteContig = 1;
        if (!gpfsmpio_onesided_no_rmw) {
          int numBytesPutIntoBuf = 0;
          for (i=0;i<nprocs;i++) {
            numBytesPutIntoBuf += write_buf_put_amounts[i];
            write_buf_put_amounts[i] = 0;
          }
          if (numBytesPutIntoBuf != ((int)(lwrrentRoundFDEnd - lwrrentRoundFDStart)+1)) {
            doWriteContig = 0;
            *hole_found = 1;
          }
        }

        if (!useIOBuffer) {
          if (doWriteContig)
            ADIO_WriteContig(fd, write_buf, (int)(lwrrentRoundFDEnd - lwrrentRoundFDStart)+1,
              MPI_BYTE, ADIO_EXPLICIT_OFFSET,lwrrentRoundFDStart, &status, error_code);

        } else { /* use the thread writer */

        if(!pthread_equal(io_thread, pthread_self())) {
            pthread_join(io_thread, &thread_ret);
            *error_code = *(int *)thread_ret;
            if (*error_code != MPI_SUCCESS) return;
            io_thread = pthread_self();

        }
        io_thread_args.fd = fd;
        /* do a little pointer shuffling: background I/O works from one
         * buffer while two-phase machinery fills up another */

        if (lwrrentWriteBuf == 0) {
            io_thread_args.buf = write_buf0;
            lwrrentWriteBuf = 1;
            write_buf = write_buf1;
        }
        else {
            io_thread_args.buf = write_buf1;
            lwrrentWriteBuf = 0;
            write_buf = write_buf0;
        }
        if (doWriteContig) {
        io_thread_args.io_kind = ADIOI_WRITE;
        io_thread_args.size = (lwrrentRoundFDEnd-lwrrentRoundFDStart) + 1;
        io_thread_args.offset = lwrrentRoundFDStart;
        io_thread_args.status = &status;
        io_thread_args.error_code = *error_code;

        if ( (pthread_create(&io_thread, NULL,
                ADIOI_IO_Thread_Func, &(io_thread_args))) != 0)
            io_thread = pthread_self();
        }
        } // useIOBuffer

    } // iAmUsedAgg

    if (!iAmUsedAgg && useIOBuffer) {
        if (lwrrentWriteBuf == 0) {
            lwrrentWriteBuf = 1;
            write_buf = write_buf1;
        }
        else {
            lwrrentWriteBuf = 0;
            write_buf = write_buf0;
        }
    }

    if (iAmUsedAgg) {
      lwrrentRoundFDStart += coll_bufsize;

      if (gpfsmpio_onesided_always_rmw && (roundIter<(numberOfRounds-1))) { // read in the buffer for the next round unless this is the last round
        ADIO_Offset tmpLwrrentRoundFDEnd = 0;
        if ((fd_end[myAggRank] - lwrrentRoundFDStart) < coll_bufsize) {
          if (myAggRank == greatestFileDomainAggRank) {
            if (fd_end[myAggRank] > lastFileOffset)
              tmpLwrrentRoundFDEnd = lastFileOffset;
            else
              tmpLwrrentRoundFDEnd = fd_end[myAggRank];
          }
          else
            tmpLwrrentRoundFDEnd = fd_end[myAggRank];
        }
        else
        tmpLwrrentRoundFDEnd = lwrrentRoundFDStart + coll_bufsize - (ADIO_Offset)1;
#ifdef onesidedtrace
printf("gpfsmpio_onesided_always_rmw - round %d buffer pre-read for file offsets %ld to %ld total is %d\n",roundIter, lwrrentRoundFDStart,tmpLwrrentRoundFDEnd,(int)(tmpLwrrentRoundFDEnd - lwrrentRoundFDStart)+1);
#endif
        ADIO_ReadContig(fd, write_buf, (int)(tmpLwrrentRoundFDEnd - lwrrentRoundFDStart)+1,
          MPI_BYTE, ADIO_EXPLICIT_OFFSET,lwrrentRoundFDStart, &status, error_code);

      }
    }

    if (roundIter<(numberOfRounds-1)) {
#ifdef onesidedtrace
printf("second barrier roundIter %d\n",roundIter);
#endif
      MPI_Barrier(fd->comm);
    }

    } /* for-loop roundIter */


#ifdef ROMIO_GPFS
    endTimeBase = MPI_Wtime();
    gpfsmpio_prof_cw[GPFSMPIO_CIO_T_DEXCH] += (endTimeBase-startTimeBase);
#endif

    if (useIOBuffer) { /* thread writer cleanup */

    if ( !pthread_equal(io_thread, pthread_self()) ) {
        pthread_join(io_thread, &thread_ret);
        *error_code = *(int *)thread_ret;
    }

    }

#ifdef onesidedtrace
printf("freeing datastructures\n");
#endif
    ADIOI_Free(targetAggsForMyData);
    ADIOI_Free(targetAggsForMyDataFDStart);
    ADIOI_Free(targetAggsForMyDataFDEnd);

    for (i=0;i<numberOfRounds;i++) {
      ADIOI_Free(targetAggsForMyDataFirstOffLenIndex[i]);
      ADIOI_Free(targetAggsForMyDataLastOffLenIndex[i]);
    }
    ADIOI_Free(targetAggsForMyDataFirstOffLenIndex);
    ADIOI_Free(targetAggsForMyDataLastOffLenIndex);

    ADIOI_Free(lwrrentFDSourceBufferState);

    if (!bufTypeIsContig)
      ADIOI_Delete_flattened(datatype);
    return;
}


void ADIOI_OneSidedReadAggregation(ADIO_File fd,
    ADIO_Offset *offset_list,
    ADIO_Offset *len_list,
    int contig_access_count,
    const void *buf,
    MPI_Datatype datatype,
    int *error_code,
    ADIO_Offset *st_offsets,
    ADIO_Offset *end_offsets,
    int numNonZeroDataOffsets,
    ADIO_Offset *fd_start,
    ADIO_Offset* fd_end)
{
    int i,j; /* generic iterators */

#ifdef onesidedtrace
    if (buf == NULL) {
      printf("ADIOI_OneSidedWriteAggregation - buf is NULL contig_access_count is %d\n",contig_access_count);
      for (i=0;i<contig_access_count;i++)
        printf("offset_list[%d] is %ld len_list[%d] is %ld\n",i,offset_list[i],i,len_list[i]);
    }
    if (contig_access_count < 0)
      printf("ADIOI_OneSidedWriteAggregation - contig_access_count of %d is less than 0\n",contig_access_count);
#endif

    int lenListOverZero = 0;
    for (i=0;((i<contig_access_count) && (!lenListOverZero));i++)
      if (len_list[i] > 0)
        lenListOverZero = 1;

    *error_code = MPI_SUCCESS; /* initialize to success */

#ifdef ROMIO_GPFS
    double startTimeBase,endTimeBase;
    startTimeBase = MPI_Wtime();
#endif

    MPI_Status status;
    pthread_t io_thread;
    void *thread_ret;
    ADIOI_IO_ThreadFuncData io_thread_args;

    int nprocs,myrank;
    MPI_Comm_size(fd->comm, &nprocs);
    MPI_Comm_rank(fd->comm, &myrank);

#ifdef onesidedtrace
printf("ADIOI_OneSidedReadAggregation started on rank %d\n",myrank);
#endif

    if (fd->io_buf_window == MPI_WIN_NULL ||
	    fd->io_buf_put_amounts_window == MPI_WIN_NULL)
    {
	ADIOI_OneSidedSetup(fd, nprocs);
    }
    /* This flag denotes whether the source datatype is contiguus, which is referenced throughout the algorithm
     * and defines how the source buffer offsets and data chunks are determined.  If the value is 1 (true - contiguous data)
     * things are profoundly simpler in that the source buffer offset for a given source offset simply linearly increases
     * by the chunk sizes being read.  If the value is 0 (non-contiguous) then these values are based on callwlations
     * from the flattened source datatype.
     */
    int bufTypeIsContig;

    MPI_Aint bufTypeExtent, lb;
    ADIOI_Flatlist_node *flatBuf=NULL;
    ADIOI_Datatype_iscontig(datatype, &bufTypeIsContig);

    if (!bufTypeIsContig) {
    /* Flatten the non-contiguous source datatype.
     */
      flatBuf = ADIOI_Flatten_and_find(datatype);
      MPI_Type_get_extent(datatype, &lb, &bufTypeExtent);
#ifdef onesidedtrace
      printf("flatBuf->count is %d bufTypeExtent is %d\n", flatBuf->count,bufTypeExtent);
      for (i=0;i<flatBuf->count;i++)
        printf("flatBuf->blocklens[%d] is %d flatBuf->indices[%d] is %ld\n",i,flatBuf->blocklens[i],i,flatBuf->indices[i]);
#endif
    }
#ifdef onesidedtrace
      printf("ADIOI_OneSidedReadAggregation bufTypeIsContig is %d contig_access_count is %d\n",bufTypeIsContig,contig_access_count);
#endif

    int naggs = fd->hints->cb_nodes;

    /* Track the state of the source buffer for feeding the target data blocks.
     * For GPFS the number of file domains per agg is always 1 so we just need 1 agg
     * dimension to track the data, in the case of lustre we will need 2 dimensions
     * agg and file domain since aggs write to multiple file domains in the
     * case of lustre.
     * This structure will be modified as the data is written to reflect the
     * current state of the offset.
     */

#ifdef onesidedtrace
    printf("sizeof(FDSourceBufferState) is %d - make sure is 32 for 32-byte memalign optimal\n",sizeof(FDSourceBufferState));
#endif
    FDSourceBufferState *lwrrentFDSourceBufferState;

    lwrrentFDSourceBufferState = (FDSourceBufferState *) ADIOI_Malloc(naggs * sizeof(FDSourceBufferState));
    for (i=0;i<naggs;i++) {
      /* initialize based on the bufType to indicate that it is unset.
       */
      if (bufTypeIsContig) {
        lwrrentFDSourceBufferState[i].sourceBufferOffset = -1;
      }
      else {
        lwrrentFDSourceBufferState[i].indiceOffset = -1;
      }
    }

#ifdef onesidedtrace
    printf(" ADIOI_OneSidedReadAggregation bufTypeIsContig is %d contig_access_count is %d\n",bufTypeIsContig,contig_access_count);
#endif

    /* maxNumContigOperations keeps track of how many different chunks we will
     * need to recv for the purpose of pre-allocating the data structures to
     * hold them.
     */
    int maxNumContigOperations = contig_access_count;


    ADIO_Offset lastFileOffset = 0, firstFileOffset = -1;

    /* Get the total range being read.
     */
    for (j=0;j<numNonZeroDataOffsets;j++) {
#ifdef onesidedtrace
printf("end_offsets[%d] is %ld st_offsets[%d] is %ld\n",j,end_offsets[j],j,st_offsets[j]);
#endif
        lastFileOffset = ADIOI_MAX(lastFileOffset,end_offsets[j]);
        if (firstFileOffset == -1)
          firstFileOffset = st_offsets[j];
        else
          firstFileOffset = ADIOI_MIN(firstFileOffset,st_offsets[j]);
    }

    int myAggRank = -1; /* if I am an aggregor this is my index into fd->hints->ranklist */
    int iAmUsedAgg = 0; /* whether or not this rank is used as an aggregator. */

    int coll_bufsize = fd->hints->cb_buffer_size;
#ifdef ROMIO_GPFS
    if (gpfsmpio_pthreadio == 1) {
    /* split buffer in half for a kind of double buffering with the threads*/
    coll_bufsize = fd->hints->cb_buffer_size/2;
    }
#endif

    /* This logic defines values that are used later to determine what offsets define the portion
     * of the file domain the agg is reading this round.
     */
    int greatestFileDomainAggRank = -1,smallestFileDomainAggRank = -1;
    ADIO_Offset greatestFileDomainOffset = 0;
    ADIO_Offset smallestFileDomainOffset = lastFileOffset;
    for (j=0;j<naggs;j++) {
      if (fd_end[j] > greatestFileDomainOffset) {
        greatestFileDomainOffset = fd_end[j];
        greatestFileDomainAggRank = j;
      }
      if (fd_start[j] < smallestFileDomainOffset) {
        smallestFileDomainOffset = fd_start[j];
        smallestFileDomainAggRank = j;
      }
      if (fd->hints->ranklist[j] == myrank) {
        myAggRank = j;
        if (fd_end[j] > fd_start[j]) {
          iAmUsedAgg = 1;
        }
      }
    }

#ifdef onesidedtrace
    printf("contig_access_count is %d lastFileOffset is %ld firstFileOffset is %ld\n",contig_access_count,lastFileOffset,firstFileOffset);
    for (j=0;j<contig_access_count;j++) {
      printf("offset_list[%d]: %ld , len_list[%d]: %ld\n",j,offset_list[j],j,len_list[j]);
    }
#endif

    /* Compute number of rounds.
     */
    int numberOfRounds = 0;
    for (j=0;j<naggs;j++) {
	  int lwrrentNumberOfRounds = (int)(((fd_end[j] - fd_start[j])+(ADIO_Offset)1)/coll_bufsize);
      if (((ADIO_Offset)lwrrentNumberOfRounds*coll_bufsize) < ((fd_end[j] - fd_start[j])+(ADIO_Offset)1))
        lwrrentNumberOfRounds++;
	  if (lwrrentNumberOfRounds > numberOfRounds)
	    numberOfRounds = lwrrentNumberOfRounds;
    }

    /* Data structures to track what data this compute needs to receive from whom.
     * For lustre they will all need another dimension for the file domain.
     */    int *sourceAggsForMyData = (int *)ADIOI_Malloc(naggs * sizeof(int));
    ADIO_Offset *sourceAggsForMyDataFDStart = (ADIO_Offset *)ADIOI_Malloc(naggs * sizeof(ADIO_Offset));
    ADIO_Offset *sourceAggsForMyDataFDEnd = (ADIO_Offset *)ADIOI_Malloc(naggs * sizeof(ADIO_Offset));
    int numSourceAggs = 0;

    /* This data structure holds the beginning offset and len list index for the range to be read
     * coresponding to the round and source agg. Initialize to -1 to denote being unset.
    */
    int **sourceAggsForMyDataFirstOffLenIndex = (int **)ADIOI_Malloc(numberOfRounds * sizeof(int *));
    for (i=0;i<numberOfRounds;i++) {
      sourceAggsForMyDataFirstOffLenIndex[i] = (int *)ADIOI_Malloc(naggs * sizeof(int));
      for (j=0;j<naggs;j++)
        sourceAggsForMyDataFirstOffLenIndex[i][j] = -1;
    }

    /* This data structure holds the ending offset and len list index for the range to be read
     * coresponding to the round and source agg.
    */
    int **sourceAggsForMyDataLastOffLenIndex = (int **)ADIOI_Malloc(numberOfRounds * sizeof(int *));
    for (i=0;i<numberOfRounds;i++)
      sourceAggsForMyDataLastOffLenIndex[i] = (int *)ADIOI_Malloc(naggs * sizeof(int));

#ifdef onesidedtrace
    printf("NumberOfRounds is %d\n",numberOfRounds);
    for (i=0;i<naggs;i++)
      printf("fd->hints->ranklist[%d]is %d fd_start is %ld fd_end is %ld\n",i,fd->hints->ranklist[i],fd_start[i],fd_end[i]);
    for (j=0;j<contig_access_count;j++)
      printf("offset_list[%d] is %ld len_list is %ld\n",j,offset_list[j],len_list[j]);
#endif

    int lwrrentAggRankListIndex = 0;
    int maxNumNonContigSourceChunks = 0;

    ADIO_Offset lwrrentRecvBufferOffset = 0;
    int lwrrentDataTypeExtent = 0;
    int lwrrentFlatBufIndice=0;
    ADIO_Offset lwrrentIndiceOffset = 0;

    /* This denotes the coll_bufsize boundaries within the source buffer for reading for 1 round.
     */
    ADIO_Offset intraRoundCollBufsizeOffset = 0;

    /* This data structure tracks what source aggs need to be read to on what rounds.
     */
    int *sourceAggsForMyDataLwrrentRoundIter = (int *)ADIOI_Malloc(naggs * sizeof(int));
    for (i=0;i<naggs;i++)
      sourceAggsForMyDataLwrrentRoundIter[i] = 0;

    /* This is the first of the two main loops in this algorithm.  The purpose of this loop is essentially to populate
     * the data structures defined above for what read data blocks needs to go where (source agg and file domain) and when
     * (round iter).  For lustre essentially an additional layer of nesting will be required for the multiple file domains
     * within the source agg.
     */
     if ((contig_access_count > 0) && (buf != NULL) && lenListOverZero) {
    int blockIter;
    for (blockIter=0;blockIter<contig_access_count;blockIter++) {

      /* Determine the starting source buffer offset for this block - for iter 0 skip it since that value is 0.
       */
      if (blockIter>0) {
        if (bufTypeIsContig) {
          lwrrentRecvBufferOffset += len_list[blockIter-1];
        }
        else {
          /* Non-contiguous source datatype, count up the extents and indices to this point
           * in the blocks.
           */
          ADIO_Offset sourceBlockTotal = 0;
          int lastIndiceUsed = lwrrentFlatBufIndice;
          int numNonContigSourceChunks = 0;
          while (sourceBlockTotal < len_list[blockIter-1]) {
            numNonContigSourceChunks++;
            sourceBlockTotal += (flatBuf->blocklens[lwrrentFlatBufIndice] - lwrrentIndiceOffset);
            lastIndiceUsed = lwrrentFlatBufIndice;
            lwrrentFlatBufIndice++;
            if (lwrrentFlatBufIndice == flatBuf->count) {
              lwrrentFlatBufIndice = 0;
              lwrrentDataTypeExtent++;
            }
            lwrrentIndiceOffset = (ADIO_Offset)0;
          }
          if (sourceBlockTotal > len_list[blockIter-1]) {
            lwrrentFlatBufIndice--;
            if (lwrrentFlatBufIndice < 0 ) {
              lwrrentDataTypeExtent--;
              lwrrentFlatBufIndice = flatBuf->count-1;
            }
            lwrrentIndiceOffset =  len_list[blockIter-1] - (sourceBlockTotal - flatBuf->blocklens[lastIndiceUsed]);
            // ADIOI_Assert((lwrrentIndiceOffset >= 0) && (lwrrentIndiceOffset < flatBuf->blocklens[lwrrentFlatBufIndice]));
          }
          else
            lwrrentIndiceOffset = 0;
          maxNumContigOperations += (numNonContigSourceChunks+2);
          if (numNonContigSourceChunks > maxNumNonContigSourceChunks)
            maxNumNonContigSourceChunks = numNonContigSourceChunks;

#ifdef onesidedtrace
          printf("block iter %d lwrrentFlatBufIndice is now %d lwrrentDataTypeExtent is now %d lwrrentIndiceOffset is now %ld maxNumContigOperations is now %d\n",blockIter,lwrrentFlatBufIndice,lwrrentDataTypeExtent,lwrrentIndiceOffset,maxNumContigOperations);
#endif
        } // !bufTypeIsContig
      } // blockIter > 0

      /* For the last iteration we need to include these maxNumContigOperations and maxNumNonContigSourceChunks
       * for non-contig case even though we did not need to compute the next starting offset.
       */
      if ((blockIter == (contig_access_count-1)) && (!bufTypeIsContig)) {
        ADIO_Offset sourceBlockTotal = 0;
        int tmpLwrrentFlatBufIndice = lwrrentFlatBufIndice;
        int  lastNumNonContigSourceChunks = 0;
        while (sourceBlockTotal < len_list[blockIter]) {
          lastNumNonContigSourceChunks++;
          sourceBlockTotal += flatBuf->blocklens[tmpLwrrentFlatBufIndice];
          tmpLwrrentFlatBufIndice++;
          if (tmpLwrrentFlatBufIndice == flatBuf->count) {
            tmpLwrrentFlatBufIndice = 0;
          }
        }
        maxNumContigOperations += (lastNumNonContigSourceChunks+2);
        if (lastNumNonContigSourceChunks > maxNumNonContigSourceChunks)
          maxNumNonContigSourceChunks = lastNumNonContigSourceChunks;

      }

      ADIO_Offset blockStart = offset_list[blockIter], blockEnd = offset_list[blockIter]+len_list[blockIter]-(ADIO_Offset)1;

      /* Find the starting source agg for this block - normally it will be the current agg so guard the expensive
       * while loop with a cheap if-check which for large numbers of small blocks will usually be false.
       */
      if (!((blockStart >= fd_start[lwrrentAggRankListIndex]) && (blockStart <= fd_end[lwrrentAggRankListIndex]))) {
        while (!((blockStart >= fd_start[lwrrentAggRankListIndex]) && (blockStart <= fd_end[lwrrentAggRankListIndex])))
          lwrrentAggRankListIndex++;
      };

      /* Determine if this is a new source agg.
       */
      if (blockIter>0) {
        if ((offset_list[blockIter-1]+len_list[blockIter-1]-(ADIO_Offset)1) < fd_start[lwrrentAggRankListIndex])
          numSourceAggs++;
      }

       /* Determine which round to start reading.
        */
      if ((blockStart - fd_start[lwrrentAggRankListIndex]) >= coll_bufsize) {
        ADIO_Offset lwrrentRoundBlockStart = fd_start[lwrrentAggRankListIndex];
        int startingRound = 0;
        while (blockStart > (lwrrentRoundBlockStart + coll_bufsize - (ADIO_Offset)1)) {
          lwrrentRoundBlockStart+=coll_bufsize;
          startingRound++;
        }
        sourceAggsForMyDataLwrrentRoundIter[numSourceAggs] = startingRound;
      }

      /* Initialize the data structures if this is the first offset in the round/source agg.
       */
      if (sourceAggsForMyDataFirstOffLenIndex[sourceAggsForMyDataLwrrentRoundIter[numSourceAggs]][numSourceAggs] == -1) {
        sourceAggsForMyData[numSourceAggs] = fd->hints->ranklist[lwrrentAggRankListIndex];
        sourceAggsForMyDataFDStart[numSourceAggs] = fd_start[lwrrentAggRankListIndex];
        /* Round up file domain to the first actual offset used if this is the first file domain.
         */
        if (lwrrentAggRankListIndex == smallestFileDomainAggRank) {
          if (sourceAggsForMyDataFDStart[numSourceAggs] < firstFileOffset)
            sourceAggsForMyDataFDStart[numSourceAggs] = firstFileOffset;
        }
        sourceAggsForMyDataFDEnd[numSourceAggs] = fd_end[lwrrentAggRankListIndex];
        /* Round down file domain to the last actual offset used if this is the last file domain.
         */
        if (lwrrentAggRankListIndex == greatestFileDomainAggRank) {
          if (sourceAggsForMyDataFDEnd[numSourceAggs] > lastFileOffset)
            sourceAggsForMyDataFDEnd[numSourceAggs] = lastFileOffset;
        }
        sourceAggsForMyDataFirstOffLenIndex[sourceAggsForMyDataLwrrentRoundIter[numSourceAggs]][numSourceAggs] = blockIter;

        /* Set the source buffer state starting point for data access for this agg and file domain.
         */
        if (bufTypeIsContig) {
          if (lwrrentFDSourceBufferState[numSourceAggs].sourceBufferOffset == -1) {

            lwrrentFDSourceBufferState[numSourceAggs].sourceBufferOffset = lwrrentRecvBufferOffset;
#ifdef onesidedtrace
            printf("For agg %d sourceBufferOffset initialized to %ld\n",lwrrentAggRankListIndex,lwrrentRecvBufferOffset);
#endif
          }
        }
	else {
          if (lwrrentFDSourceBufferState[numSourceAggs].indiceOffset == -1) {
            lwrrentFDSourceBufferState[numSourceAggs].indiceOffset = lwrrentIndiceOffset;
            lwrrentFDSourceBufferState[numSourceAggs].bufTypeExtent = bufTypeExtent;
            lwrrentFDSourceBufferState[numSourceAggs].dataTypeExtent = lwrrentDataTypeExtent;
            lwrrentFDSourceBufferState[numSourceAggs].flatBufIndice = lwrrentFlatBufIndice;
#ifdef onesidedtrace
            printf("For agg %d dataTypeExtent initialized to %d flatBufIndice to %d indiceOffset to %ld\n",numSourceAggs,lwrrentDataTypeExtent,lwrrentFlatBufIndice,lwrrentIndiceOffset);
#endif
          }
	}
        intraRoundCollBufsizeOffset = fd_start[lwrrentAggRankListIndex] + ((ADIO_Offset)(sourceAggsForMyDataLwrrentRoundIter[numSourceAggs]+1) * coll_bufsize);
#ifdef onesidedtrace
        printf("init settings numSourceAggs %d offset_list[%d] with value %ld past fd border %ld with len %ld lwrrentRecvBufferOffset set to %ld intraRoundCollBufsizeOffset set to %ld\n",numSourceAggs,blockIter,offset_list[blockIter],fd_start[lwrrentAggRankListIndex],len_list[blockIter],lwrrentRecvBufferOffset,intraRoundCollBufsizeOffset);
#endif

      }

      /* Replace the last offset block iter with this one.
       */
      sourceAggsForMyDataLastOffLenIndex[sourceAggsForMyDataLwrrentRoundIter[numSourceAggs]][numSourceAggs] = blockIter;

      /* If this blocks extends into the next file domain advance to the next source aggs and source buffer states.
       */
      if (blockEnd > fd_end[lwrrentAggRankListIndex]) {
#ifdef onesidedtrace
      printf("block extends past current fd, blockEnd %ld >= fd_end[lwrrentAggRankListIndex] %ld total block size is %ld blockStart was %ld\n",blockEnd,fd_end[lwrrentAggRankListIndex], len_list[blockIter],blockStart);
#endif
        ADIO_Offset amountToAdvanceSBOffsetForFD = 0;
        int additionalFDCounter = 0;
        while (blockEnd >= fd_end[lwrrentAggRankListIndex]) {
          ADIO_Offset thisAggBlockEnd = fd_end[lwrrentAggRankListIndex];
          if (thisAggBlockEnd >= intraRoundCollBufsizeOffset) {
            while (thisAggBlockEnd >= intraRoundCollBufsizeOffset) {
              sourceAggsForMyDataLwrrentRoundIter[numSourceAggs]++;
              intraRoundCollBufsizeOffset += coll_bufsize;
              sourceAggsForMyDataFirstOffLenIndex[sourceAggsForMyDataLwrrentRoundIter[numSourceAggs]][numSourceAggs] = blockIter;
              sourceAggsForMyDataLastOffLenIndex[sourceAggsForMyDataLwrrentRoundIter[numSourceAggs]][numSourceAggs] = blockIter;
#ifdef onesidedtrace
              printf("sourceAggsForMyDataLwrrentRoundI%d] is now %d intraRoundCollBufsizeOffset is now %ld\n",numSourceAggs,sourceAggsForMyDataLwrrentRoundIter[numSourceAggs],intraRoundCollBufsizeOffset);
#endif
            } // while (thisAggBlockEnd >= intraRoundCollBufsizeOffset)
          } // if (thisAggBlockEnd >= intraRoundCollBufsizeOffset)

          int prevAggRankListIndex = lwrrentAggRankListIndex;
          lwrrentAggRankListIndex++;

          /* Skip over unused aggs.
           */
          if (fd_start[lwrrentAggRankListIndex] > fd_end[lwrrentAggRankListIndex]) {
            while (fd_start[lwrrentAggRankListIndex] > fd_end[lwrrentAggRankListIndex])
              lwrrentAggRankListIndex++;

          } // (fd_start[lwrrentAggRankListIndex] > fd_end[lwrrentAggRankListIndex])

          /* Start new source agg.
           */
          if (blockEnd >= fd_start[lwrrentAggRankListIndex]) {
            numSourceAggs++;
            sourceAggsForMyData[numSourceAggs] = fd->hints->ranklist[lwrrentAggRankListIndex];
            sourceAggsForMyDataFDStart[numSourceAggs] = fd_start[lwrrentAggRankListIndex];
            /* Round up file domain to the first actual offset used if this is the first file domain.
             */
            if (lwrrentAggRankListIndex == smallestFileDomainAggRank) {
              if (sourceAggsForMyDataFDStart[numSourceAggs] < firstFileOffset)
                sourceAggsForMyDataFDStart[numSourceAggs] = firstFileOffset;
            }
            sourceAggsForMyDataFDEnd[numSourceAggs] = fd_end[lwrrentAggRankListIndex];
            /* Round down file domain to the last actual offset used if this is the last file domain.
             */
            if (lwrrentAggRankListIndex == greatestFileDomainAggRank) {
              if (sourceAggsForMyDataFDEnd[numSourceAggs] > lastFileOffset)
                sourceAggsForMyDataFDEnd[numSourceAggs] = lastFileOffset;
            }
            sourceAggsForMyDataFirstOffLenIndex[sourceAggsForMyDataLwrrentRoundIter[numSourceAggs]][numSourceAggs] = blockIter;


	    /* For the first additonal file domain the source buffer offset
	     * will be incremented relative to the state of this first main
	     * loop but for subsequent full file domains the offset will be
	     * incremented by the size of the file domain.
             */
            if (additionalFDCounter == 0)
              amountToAdvanceSBOffsetForFD = (fd_end[prevAggRankListIndex] - blockStart) + (ADIO_Offset)1;
            else
              amountToAdvanceSBOffsetForFD = (fd_end[prevAggRankListIndex]-fd_start[prevAggRankListIndex]) +(ADIO_Offset)1;

            if (bufTypeIsContig) {
              ADIOI_Assert(numSourceAggs > 0);
              if (lwrrentFDSourceBufferState[numSourceAggs].sourceBufferOffset == -1) {
                if (additionalFDCounter == 0) { // first file domain, still use the current data counter
                  lwrrentFDSourceBufferState[numSourceAggs].sourceBufferOffset = lwrrentRecvBufferOffset+amountToAdvanceSBOffsetForFD;
                }
                else { // 2nd file domain, advance full file domain from last source buffer state
                  lwrrentFDSourceBufferState[numSourceAggs].sourceBufferOffset = lwrrentFDSourceBufferState[numSourceAggs-1].sourceBufferOffset+amountToAdvanceSBOffsetForFD;
                }

#ifdef onesidedtrace
            printf("Crossed into new FD - for agg %d sourceBufferOffset initialized to %ld amountToAdvanceSBOffsetForFD is %ld\n",numSourceAggs,lwrrentFDSourceBufferState[numSourceAggs].sourceBufferOffset,amountToAdvanceSBOffsetForFD);
#endif
              }
            }
            else if (lwrrentFDSourceBufferState[numSourceAggs].indiceOffset == -1) {
		// non-contiguos source buffer
              ADIOI_Assert(numSourceAggs > 0);

	      /* Initialize the source buffer state appropriately and then
	       * advance it with the nonContigSourceDataBufferAdvance function.
               */
              if (additionalFDCounter == 0) {
		  // first file domain, still use the current data counter
                lwrrentFDSourceBufferState[numSourceAggs].indiceOffset = lwrrentIndiceOffset;
                lwrrentFDSourceBufferState[numSourceAggs].bufTypeExtent = bufTypeExtent;
                lwrrentFDSourceBufferState[numSourceAggs].dataTypeExtent = lwrrentDataTypeExtent;
                lwrrentFDSourceBufferState[numSourceAggs].flatBufIndice = lwrrentFlatBufIndice;
              }
              else {
		  // 2nd file domain, advance full file domain from last source
		  // buffer state
                lwrrentFDSourceBufferState[numSourceAggs].indiceOffset = lwrrentFDSourceBufferState[numSourceAggs-1].indiceOffset;
                lwrrentFDSourceBufferState[numSourceAggs].bufTypeExtent = lwrrentFDSourceBufferState[numSourceAggs-1].bufTypeExtent;
                lwrrentFDSourceBufferState[numSourceAggs].dataTypeExtent = lwrrentFDSourceBufferState[numSourceAggs-1].dataTypeExtent;
                lwrrentFDSourceBufferState[numSourceAggs].flatBufIndice = lwrrentFDSourceBufferState[numSourceAggs-1].flatBufIndice;
              }
              nonContigSourceDataBufferAdvance(((char*)buf), flatBuf, (int)amountToAdvanceSBOffsetForFD, 0, &lwrrentFDSourceBufferState[numSourceAggs], NULL);
#ifdef onesidedtrace
              printf("Crossed into new FD - for agg %d dataTypeExtent initialized to %d flatBufIndice to %d indiceOffset to %ld amountToAdvanceSBOffsetForFD is %d\n",numSourceAggs,lwrrentFDSourceBufferState[numSourceAggs].dataTypeExtent,lwrrentFDSourceBufferState[numSourceAggs].flatBufIndice,lwrrentFDSourceBufferState[numSourceAggs].indiceOffset,amountToAdvanceSBOffsetForFD);
#endif
            }

            additionalFDCounter++;
 

#ifdef onesidedtrace
            printf("block extended beyond fd init settings numSourceAggs %d offset_list[%d] with value %ld past fd border %ld with len %ld\n",numSourceAggs,i,offset_list[blockIter],fd_start[lwrrentAggRankListIndex],len_list[blockIter]);
#endif
            intraRoundCollBufsizeOffset = fd_start[lwrrentAggRankListIndex] + coll_bufsize;
            sourceAggsForMyDataLastOffLenIndex[sourceAggsForMyDataLwrrentRoundIter[numSourceAggs]][numSourceAggs] = blockIter;
          } // if (blockEnd >= fd_start[lwrrentAggRankListIndex])
        } // while (blockEnd >= fd_end[lwrrentAggRankListIndex])
      } // if (blockEnd > fd_end[lwrrentAggRankListIndex])

      /* If we are still in the same file domain / source agg but have gone past the coll_bufsize and need
       * to advance to the next round handle this situation.
       */
      if (blockEnd >= intraRoundCollBufsizeOffset) {
        ADIO_Offset lwrrentBlockEnd = blockEnd;
        while (lwrrentBlockEnd >= intraRoundCollBufsizeOffset) {
          sourceAggsForMyDataLwrrentRoundIter[numSourceAggs]++;
          intraRoundCollBufsizeOffset += coll_bufsize;
          sourceAggsForMyDataFirstOffLenIndex[sourceAggsForMyDataLwrrentRoundIter[numSourceAggs]][numSourceAggs] = blockIter;
          sourceAggsForMyDataLastOffLenIndex[sourceAggsForMyDataLwrrentRoundIter[numSourceAggs]][numSourceAggs] = blockIter;
#ifdef onesidedtrace
          printf("block less than fd lwrrentBlockEnd is now %ld intraRoundCollBufsizeOffset is now %ld sourceAggsForMyDataLwrrentRoundIter[%d] is now %d\n",lwrrentBlockEnd, intraRoundCollBufsizeOffset, numSourceAggs,sourceAggsForMyDataLwrrentRoundIter[numSourceAggs]);
#endif
        } // while (lwrrentBlockEnd >= intraRoundCollBufsizeOffset)
      } // if (blockEnd >= intraRoundCollBufsizeOffset)

      /* Need to advance numSourceAggs if this is the last source offset to
       * include this one.
       */
      if (blockIter == (contig_access_count-1))
        numSourceAggs++;
    }

#ifdef onesidedtrace
    printf("numSourceAggs is %d\n",numSourceAggs);
    for (i=0;i<numSourceAggs;i++) {
      for (j=0;j<=sourceAggsForMyDataLwrrentRoundIter[i];j++)
        printf("sourceAggsForMyData[%d] is %d sourceAggsForMyDataFDStart[%d] is %ld sourceAggsForMyDataFDEnd is %ld sourceAggsForMyDataFirstOffLenIndex is %d with value %ld sourceAggsForMyDataLastOffLenIndex is %d with value %ld\n",i,sourceAggsForMyData[i],i,sourceAggsForMyDataFDStart[i],sourceAggsForMyDataFDEnd[i],sourceAggsForMyDataFirstOffLenIndex[j][i],offset_list[sourceAggsForMyDataFirstOffLenIndex[j][i]],sourceAggsForMyDataLastOffLenIndex[j][i],offset_list[sourceAggsForMyDataLastOffLenIndex[j][i]]);

    }
#endif

    } // if ((contig_access_count > 0) && (buf != NULL) && lenListOverZero)

    ADIOI_Free(sourceAggsForMyDataLwrrentRoundIter);

    /* use the two-phase buffer allocated in the file_open - no app should ever
     * be both reading and reading at the same time */
    char *read_buf0 = fd->io_buf;
    char *read_buf1 = fd->io_buf + coll_bufsize;
    /* if threaded i/o selected, we'll do a kind of double buffering */
    char *read_buf = read_buf0;

    int lwrrentReadBuf = 0;
    int useIOBuffer = 0;
#ifdef ROMIO_GPFS
    if (gpfsmpio_pthreadio && (numberOfRounds>1)) {
    useIOBuffer = 1;
    io_thread = pthread_self();
    }
#endif

    MPI_Win read_buf_window = fd->io_buf_window;

    ADIO_Offset lwrrentRoundFDStart = 0, nextRoundFDStart = 0;
    ADIO_Offset lwrrentRoundFDEnd = 0, nextRoundFDEnd = 0;

    if (iAmUsedAgg) {
      lwrrentRoundFDStart = fd_start[myAggRank];
      nextRoundFDStart = fd_start[myAggRank];
      if (myAggRank == smallestFileDomainAggRank) {
        if (lwrrentRoundFDStart < firstFileOffset)
          lwrrentRoundFDStart = firstFileOffset;
        if (nextRoundFDStart < firstFileOffset)
          nextRoundFDStart = firstFileOffset;
      }
      else if (myAggRank == greatestFileDomainAggRank) {
        if (lwrrentRoundFDEnd > lastFileOffset)
          lwrrentRoundFDEnd = lastFileOffset;
        if (nextRoundFDEnd > lastFileOffset)
          nextRoundFDEnd = lastFileOffset;
      }
#ifdef onesidedtrace
printf("iAmUsedAgg - lwrrentRoundFDStart initialized "
	"to %ld lwrrentRoundFDEnd to %ld\n",
	lwrrentRoundFDStart,lwrrentRoundFDEnd);
#endif


    }

#ifdef ROMIO_GPFS
    endTimeBase = MPI_Wtime();
    gpfsmpio_prof_cw[GPFSMPIO_CIO_T_DEXCH_SETUP] += (endTimeBase-startTimeBase);
    startTimeBase = MPI_Wtime();
#endif


    /* This is the second main loop of the algorithm, actually nested loop of target aggs within rounds.  There are 2 flavors of this.
     * For gpfsmpio_read_aggmethod of 1 each nested iteration for the source agg does an mpi_put on a contiguous chunk using a primative datatype
     * determined using the data structures from the first main loop.  For gpfsmpio_read_aggmethod of 2 each nested iteration for the source agg
     * builds up data to use in created a derived data type for 1 mpi_put that is done for the target agg for each round.
     * To support lustre there will need to be an additional layer of nesting for the multiple file domains
     * within target aggs.
     */
    int roundIter;
    for (roundIter=0;roundIter<numberOfRounds;roundIter++) {

    if ((contig_access_count > 0) && (buf != NULL) && lenListOverZero)
    {
    /* determine what offsets define the portion of the file domain the agg is reading this round */
    if (iAmUsedAgg) {

        lwrrentRoundFDStart = nextRoundFDStart;

        if (!useIOBuffer || (roundIter == 0)) {
        int amountDataToReadThisRound;
        if ((fd_end[myAggRank] - lwrrentRoundFDStart) < coll_bufsize) {
            lwrrentRoundFDEnd = fd_end[myAggRank];
            amountDataToReadThisRound = ((lwrrentRoundFDEnd-lwrrentRoundFDStart)+1);
        }
        else {
            lwrrentRoundFDEnd = lwrrentRoundFDStart + coll_bufsize - (ADIO_Offset)1;
            amountDataToReadThisRound = coll_bufsize;
        }

        /* read lwrrentRoundFDEnd bytes */
        ADIO_ReadContig(fd, read_buf,amountDataToReadThisRound,
            MPI_BYTE, ADIO_EXPLICIT_OFFSET, lwrrentRoundFDStart,
            &status, error_code);
        lwrrentReadBuf = 1;

        }
        if (useIOBuffer) { /* use the thread reader for the next round */
        /* switch back and forth between the read buffers so that the data aggregation code is diseminating 1 buffer while the thread is reading into the other */

        if (roundIter > 0)
            lwrrentRoundFDEnd = nextRoundFDEnd;

        if (roundIter < (numberOfRounds-1)) {
            nextRoundFDStart += coll_bufsize;
            int amountDataToReadNextRound;
            if ((fd_end[myAggRank] - nextRoundFDStart) < coll_bufsize) {
            nextRoundFDEnd = fd_end[myAggRank];
            amountDataToReadNextRound = ((nextRoundFDEnd-nextRoundFDStart)+1);
            }
            else {
            nextRoundFDEnd = nextRoundFDStart + coll_bufsize - (ADIO_Offset)1;
            amountDataToReadNextRound = coll_bufsize;
            }

            if(!pthread_equal(io_thread, pthread_self())) {
            pthread_join(io_thread, &thread_ret);
            *error_code = *(int *)thread_ret;
            if (*error_code != MPI_SUCCESS) return;
            io_thread = pthread_self();

            }
            io_thread_args.fd = fd;
            /* do a little pointer shuffling: background I/O works from one
             * buffer while two-phase machinery fills up another */

            if (lwrrentReadBuf == 0) {
            io_thread_args.buf = read_buf0;
            lwrrentReadBuf = 1;
            read_buf = read_buf1;
            }
            else {
            io_thread_args.buf = read_buf1;
            lwrrentReadBuf = 0;
            read_buf = read_buf0;
            }
            io_thread_args.io_kind = ADIOI_READ;
            io_thread_args.size = amountDataToReadNextRound;
            io_thread_args.offset = nextRoundFDStart;
            io_thread_args.status = &status;
            io_thread_args.error_code = *error_code;
            if ( (pthread_create(&io_thread, NULL,
                    ADIOI_IO_Thread_Func, &(io_thread_args))) != 0)
            io_thread = pthread_self();

        }
        else { /* last round */

            if(!pthread_equal(io_thread, pthread_self())) {
            pthread_join(io_thread, &thread_ret);
            *error_code = *(int *)thread_ret;
            if (*error_code != MPI_SUCCESS) return;
            io_thread = pthread_self();

            }
            if (lwrrentReadBuf == 0) {
            read_buf = read_buf0;
            }
            else {
            read_buf = read_buf1;
            }

        }
        } /* useIOBuffer */
    } /* IAmUsedAgg */
    else if (useIOBuffer) {
      if (roundIter < (numberOfRounds-1)) {
            if (lwrrentReadBuf == 0) {
            lwrrentReadBuf = 1;
            read_buf = read_buf1;
            }
            else {
            lwrrentReadBuf = 0;
            read_buf = read_buf0;
            }
      }
      else {
            if (lwrrentReadBuf == 0) {
            read_buf = read_buf0;
            }
            else {
            read_buf = read_buf1;
            }
      }

    }
    // wait until the read buffers are full before we start pulling from the source procs
    MPI_Barrier(fd->comm);

    int aggIter;
    for (aggIter=0;aggIter<numSourceAggs;aggIter++) {

    /* If we have data for the round/agg process it.
     */
    if (sourceAggsForMyDataFirstOffLenIndex[roundIter][aggIter] != -1)  {

      ADIO_Offset lwrrentRoundFDStartForMySourceAgg = (ADIO_Offset)((ADIO_Offset)sourceAggsForMyDataFDStart[aggIter] + (ADIO_Offset)((ADIO_Offset)roundIter*coll_bufsize));
      ADIO_Offset lwrrentRoundFDEndForMySourceAgg = (ADIO_Offset)((ADIO_Offset)sourceAggsForMyDataFDStart[aggIter] + (ADIO_Offset)((ADIO_Offset)(roundIter+1)*coll_bufsize) - (ADIO_Offset)1);

      int sourceAggContigAccessCount = 0;

      /* These data structures are used for the derived datatype mpi_get
       * in the gpfsmpio_read_aggmethod of 2 case.
       */
      int *sourceAggBlockLengths=NULL;
      MPI_Aint *sourceAggDisplacements=NULL, *recvBufferDisplacements=NULL;
      MPI_Datatype *sourceAggDataTypes=NULL;
      char *derivedTypePackedSourceBuffer=NULL;
      int derivedTypePackedSourceBufferOffset = 0;
      int allocatedDerivedTypeArrays = 0;
      ADIO_Offset amountOfDataReadThisRoundAgg = 0;

      /* Process the range of offsets for this source agg.
       */
      int offsetIter;
      int startingOffLenIndex = sourceAggsForMyDataFirstOffLenIndex[roundIter][aggIter], endingOffLenIndex = sourceAggsForMyDataLastOffLenIndex[roundIter][aggIter];
      for (offsetIter=startingOffLenIndex;offsetIter<=endingOffLenIndex;offsetIter++) {
        if (lwrrentRoundFDEndForMySourceAgg > sourceAggsForMyDataFDEnd[aggIter])
            lwrrentRoundFDEndForMySourceAgg = sourceAggsForMyDataFDEnd[aggIter];

        ADIO_Offset offsetStart = offset_list[offsetIter], offsetEnd = (offset_list[offsetIter]+len_list[offsetIter]-(ADIO_Offset)1);

        /* Determine the amount of data and exact source buffer offsets to use.
         */
        int bufferAmountToRecv = 0;

        if ((offsetStart >= lwrrentRoundFDStartForMySourceAgg) && (offsetStart <= lwrrentRoundFDEndForMySourceAgg)) {
            if (offsetEnd > lwrrentRoundFDEndForMySourceAgg)
            bufferAmountToRecv = (lwrrentRoundFDEndForMySourceAgg - offsetStart) +1;
            else
            bufferAmountToRecv = (offsetEnd - offsetStart) +1;
        }
        else if ((offsetEnd >= lwrrentRoundFDStartForMySourceAgg) && (offsetEnd <= lwrrentRoundFDEndForMySourceAgg)) {
            if (offsetEnd > lwrrentRoundFDEndForMySourceAgg)
            bufferAmountToRecv = (lwrrentRoundFDEndForMySourceAgg - lwrrentRoundFDStartForMySourceAgg) +1;
            else
            bufferAmountToRecv = (offsetEnd - lwrrentRoundFDStartForMySourceAgg) +1;
            if (offsetStart < lwrrentRoundFDStartForMySourceAgg) {
              offsetStart = lwrrentRoundFDStartForMySourceAgg;
            }
        }
        else if ((offsetStart <= lwrrentRoundFDStartForMySourceAgg) && (offsetEnd >= lwrrentRoundFDEndForMySourceAgg)) {
            bufferAmountToRecv = (lwrrentRoundFDEndForMySourceAgg - lwrrentRoundFDStartForMySourceAgg) +1;
            offsetStart = lwrrentRoundFDStartForMySourceAgg;
        }

        if (bufferAmountToRecv > 0) { /* we have data to recv this round */
          if (gpfsmpio_read_aggmethod == 2) {
            /* Only allocate these arrays if we are using method 2 and only do it once for this round/source agg.
             */
            if (!allocatedDerivedTypeArrays) {
              sourceAggBlockLengths = (int *)ADIOI_Malloc(maxNumContigOperations * sizeof(int));
              sourceAggDisplacements = (MPI_Aint *)ADIOI_Malloc(maxNumContigOperations * sizeof(MPI_Aint));
              recvBufferDisplacements = (MPI_Aint *)ADIOI_Malloc(maxNumContigOperations * sizeof(MPI_Aint));
              sourceAggDataTypes = (MPI_Datatype *)ADIOI_Malloc(maxNumContigOperations * sizeof(MPI_Datatype));
              if (!bufTypeIsContig) {
                int k;
                for (k=sourceAggsForMyDataFirstOffLenIndex[roundIter][aggIter];k<=sourceAggsForMyDataLastOffLenIndex[roundIter][aggIter];k++)
                  amountOfDataReadThisRoundAgg += len_list[k];

#ifdef onesidedtrace
                printf("derivedTypePackedSourceBuffer mallocing %ld\n",amountOfDataReadThisRoundAgg);
#endif
                if (amountOfDataReadThisRoundAgg > 0)
                  derivedTypePackedSourceBuffer = (char *)ADIOI_Malloc(amountOfDataReadThisRoundAgg * sizeof(char));
                else
                  derivedTypePackedSourceBuffer = NULL;
              }
              allocatedDerivedTypeArrays = 1;
            }
          }

          /* Determine the offset into the source window.
           */
          MPI_Aint sourceDisplacementToUseThisRound = (MPI_Aint) (offsetStart - lwrrentRoundFDStartForMySourceAgg);

          /* If using the thread reader select the appropriate side of the split window.
           */
          if (useIOBuffer && (read_buf == read_buf1)) {
            sourceDisplacementToUseThisRound += (MPI_Aint)coll_bufsize;
          }

          /* For gpfsmpio_read_aggmethod of 1 do the mpi_get using the primitive MPI_BYTE type from each
           * contiguous chunk from the target, if the source is non-contiguous then unpack the data after
           * the MPI_Win_unlock is done to make sure the data has arrived first.
           */
          if (gpfsmpio_read_aggmethod == 1) {
            MPI_Win_lock(MPI_LOCK_SHARED, sourceAggsForMyData[aggIter], 0, read_buf_window);
            char *getSourceData = NULL;
            if (bufTypeIsContig) {
              MPI_Get(((char*)buf) + lwrrentFDSourceBufferState[aggIter].sourceBufferOffset,bufferAmountToRecv, MPI_BYTE,sourceAggsForMyData[aggIter],sourceDisplacementToUseThisRound, bufferAmountToRecv,MPI_BYTE,read_buf_window);
              lwrrentFDSourceBufferState[aggIter].sourceBufferOffset += (ADIO_Offset)bufferAmountToRecv;

            }
            else {
              getSourceData = (char *) ADIOI_Malloc(bufferAmountToRecv*sizeof(char));
              MPI_Get(getSourceData,bufferAmountToRecv, MPI_BYTE,sourceAggsForMyData[aggIter],sourceDisplacementToUseThisRound, bufferAmountToRecv,MPI_BYTE,read_buf_window);

            }
            MPI_Win_unlock(sourceAggsForMyData[aggIter], read_buf_window);
            if (!bufTypeIsContig) {
              nonContigSourceDataBufferAdvance(((char*)buf), flatBuf, bufferAmountToRecv, 0, &lwrrentFDSourceBufferState[aggIter], getSourceData);
              ADIOI_Free(getSourceData);
            }
          }

          /* For gpfsmpio_read_aggmethod of 2 populate the data structures for this round/agg for this offset iter
           * to be used subsequently when building the derived type for 1 mpi_put for all the data for this
           * round/agg.
           */
          else if (gpfsmpio_read_aggmethod == 2) {
            if (bufTypeIsContig) {
              sourceAggBlockLengths[sourceAggContigAccessCount]= bufferAmountToRecv;
              sourceAggDataTypes[sourceAggContigAccessCount] = MPI_BYTE;
              sourceAggDisplacements[sourceAggContigAccessCount] = sourceDisplacementToUseThisRound;
              recvBufferDisplacements[sourceAggContigAccessCount] = (MPI_Aint)lwrrentFDSourceBufferState[aggIter].sourceBufferOffset;
              lwrrentFDSourceBufferState[aggIter].sourceBufferOffset += (ADIO_Offset)bufferAmountToRecv;
              sourceAggContigAccessCount++;
            }
            else {
              sourceAggBlockLengths[sourceAggContigAccessCount]= bufferAmountToRecv;
              sourceAggDataTypes[sourceAggContigAccessCount] = MPI_BYTE;
              sourceAggDisplacements[sourceAggContigAccessCount] = sourceDisplacementToUseThisRound;
              recvBufferDisplacements[sourceAggContigAccessCount] = (MPI_Aint)derivedTypePackedSourceBufferOffset;
              derivedTypePackedSourceBufferOffset += (ADIO_Offset)bufferAmountToRecv;
              sourceAggContigAccessCount++;
            }
          }
          } // bufferAmountToRecv > 0
      } // contig list

      /* For gpfsmpio_read_aggmethod of 2 now build the derived type using the data from this round/agg and do 1 single mpi_put.
       */
      if (gpfsmpio_read_aggmethod == 2) {
        MPI_Datatype recvBufferDerivedDataType, sourceBufferDerivedDataType;

        MPI_Type_create_struct(sourceAggContigAccessCount, sourceAggBlockLengths, recvBufferDisplacements, sourceAggDataTypes, &recvBufferDerivedDataType);
        MPI_Type_commit(&recvBufferDerivedDataType);
        MPI_Type_create_struct(sourceAggContigAccessCount, sourceAggBlockLengths, sourceAggDisplacements, sourceAggDataTypes, &sourceBufferDerivedDataType);
        MPI_Type_commit(&sourceBufferDerivedDataType);

        if (sourceAggContigAccessCount > 0) {

        MPI_Win_lock(MPI_LOCK_SHARED, sourceAggsForMyData[aggIter], 0, read_buf_window);
        if (bufTypeIsContig) {
          MPI_Get(((char*)buf),1, recvBufferDerivedDataType,sourceAggsForMyData[aggIter],0, 1,sourceBufferDerivedDataType,read_buf_window);
        }
        else {
          MPI_Get(derivedTypePackedSourceBuffer,1, recvBufferDerivedDataType,sourceAggsForMyData[aggIter],0, 1,sourceBufferDerivedDataType,read_buf_window);
        }

        MPI_Win_unlock(sourceAggsForMyData[aggIter], read_buf_window);
        if (!bufTypeIsContig) {
          nonContigSourceDataBufferAdvance(((char*)buf), flatBuf, derivedTypePackedSourceBufferOffset, 0, &lwrrentFDSourceBufferState[aggIter], derivedTypePackedSourceBuffer);
        }
        }

        if (allocatedDerivedTypeArrays) {
          ADIOI_Free(sourceAggBlockLengths);
          ADIOI_Free(sourceAggDisplacements);
          ADIOI_Free(sourceAggDataTypes);
          ADIOI_Free(recvBufferDisplacements);
          if (!bufTypeIsContig)
            if (derivedTypePackedSourceBuffer != NULL)
              ADIOI_Free(derivedTypePackedSourceBuffer);
        }
        if (sourceAggContigAccessCount > 0) {
        MPI_Type_free(&recvBufferDerivedDataType);
        MPI_Type_free(&sourceBufferDerivedDataType);
        }
      }
      } // baseoffset != -1
    } // source aggs
    } // contig_access_count > 0
    /* the source procs recv the requested data to the aggs */

    MPI_Barrier(fd->comm);

    nextRoundFDStart = lwrrentRoundFDStart + coll_bufsize;

    } /* for-loop roundIter */

#ifdef ROMIO_GPFS
    endTimeBase = MPI_Wtime();
    gpfsmpio_prof_cw[GPFSMPIO_CIO_T_DEXCH] += (endTimeBase-startTimeBase);
#endif

    if (useIOBuffer) { /* thread readr cleanup */

    if ( !pthread_equal(io_thread, pthread_self()) ) {
        pthread_join(io_thread, &thread_ret);
        *error_code = *(int *)thread_ret;
    }

    }

    ADIOI_Free(sourceAggsForMyData);
    ADIOI_Free(sourceAggsForMyDataFDStart);
    ADIOI_Free(sourceAggsForMyDataFDEnd);

    for (i=0;i<numberOfRounds;i++) {
      ADIOI_Free(sourceAggsForMyDataFirstOffLenIndex[i]);
      ADIOI_Free(sourceAggsForMyDataLastOffLenIndex[i]);
    }
    ADIOI_Free(sourceAggsForMyDataFirstOffLenIndex);
    ADIOI_Free(sourceAggsForMyDataLastOffLenIndex);

    ADIOI_Free(lwrrentFDSourceBufferState);

    if (!bufTypeIsContig)
      ADIOI_Delete_flattened(datatype);
    return;
}
