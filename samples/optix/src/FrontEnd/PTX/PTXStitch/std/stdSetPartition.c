/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2011-2017, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */
/*
 *  Module name              : stdSetPartition.c
 *
 *  Description              :
 */

/*--------------------------------- Includes ---------------------------------*/

#include "stdSetPartition.h"

/*--------------------------------- Functions --------------------------------*/

typedef struct {
    stdPartitionInfoRec      info;
    stdSet_t                 set;
    stdSet_t                 partitions;
    stdHashTableParameters   eltPars;
    stdHashTableParameters   setPars;
} PartitionRec;


        static void STD_CDECL mapElement( Pointer element, PartitionRec *rec )
        {
            stdSet_t sets= mapApply(rec->info.setMap,element);

            if (!sets) { 
                sets= setNEW(Pointer,16);
                mapDefine(rec->info.setMap,element,sets);
            }

            setInsert(sets, rec->set);
        }

    static void STD_CDECL mapElements( stdSet_t set, PartitionRec *rec )
    {
        rec->set = set;
        setTraverse( set, (stdEltFun)mapElement, rec );
    }

    static void STD_CDECL addElementPartition( Pointer element, PartitionRec *rec )
    {
        stdSet_t sets      = mapApply(rec->info.setMap,         element );
        stdSet_t partition = mapApply(rec->info.setToPartition, sets    );

        if (partition == Nil) {
           // partition capacity == nrof elements, which may be too liberal
           // TODO: resize all partitions at the end.
            partition = setCreate(rec->eltPars.hash, rec->eltPars.equal, rec->eltPars.nrofBuckets);
            
            mapDefine(rec->info.setToPartition, sets, partition);
        }

        mapDefine(rec->info.elementToPartition, element, partition);
        setInsert(partition, element);
    }

        static void STD_CDECL addPartition( Pointer element, PartitionRec *rec )
        {
            stdSet_t partition = mapApply(rec->info.elementToPartition,element);
            
            stdASSERT( partition, ("allElements does not contain all elements") );
            
            setInsert( rec->partitions, partition );
        }

    static void STD_CDECL mapElementSet(stdSet_t set, PartitionRec *rec)
    {
        rec->partitions= mapApply(rec->info.setToPartitions, set);
        
        if (!rec->partitions) {
            rec->partitions= setNEW(Set,rec->setPars.nrofBuckets);
            setTraverse(set, (stdEltFun)addPartition, rec);
            mapDefine(rec->info.setToPartitions, set, rec->partitions);
        }
    }

stdPartitionInfo STD_CDECL stdPartitionSets( stdSet_t allElements, stdSet_t sets)
{
    PartitionRec rec;
    
   /*
    * Get hash table parameters:
    */
    setGetHashTableParameters( allElements, &rec.eltPars );
    setGetHashTableParameters( sets,        &rec.setPars );


   /*
    * Construct for each element the subset of 'sets' 
    * in which this element oclwrs:
    */
    rec.info.setMap  = mapCreate(rec.eltPars.hash, rec.eltPars.equal, rec.eltPars.nrofBuckets);
    
    mapElements(allElements, &rec);
    setTraverse(sets, (stdEltFun)mapElements, &rec);
    

   /*
    * Each of these unique subsets constructed above
    * corresponds with a set of the requested partitioning.
    * So construct a mapping from such subsets to the set of elements
    * that have such subsets in common (and hence are 'equivalent'):
    */
    rec.info.elementToPartition = mapCreate(rec.eltPars.hash, rec.eltPars.equal, rec.eltPars.nrofBuckets);
    rec.info.setToPartition     = mapNEW(Set, rec.setPars.nrofBuckets);
  
    setTraverse(allElements, (stdEltFun)addElementPartition, &rec);
    
    
   /*
    * Construct a mapping from set of elements to all 
    * of these elements' partitions:
    */
    rec.info.setToPartitions = mapNEW(Set, rec.setPars.nrofBuckets);
    
    mapElementSet( allElements, &rec );
    setTraverse(sets, (stdEltFun)mapElementSet, &rec);


   /*
    * Construct the set of all partitions:
    */
    rec.info.allPartitions= mapRangeToSet(rec.info.elementToPartition, (stdHashFun)stdSetHash, (stdEqualFun)stdSetEqual);


   /*
    * Clean up and return:
    */
    
    return stdCOPY(&rec.info);
}
