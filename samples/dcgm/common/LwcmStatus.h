/* 
 * File:   LwcmStatus.h
 */

#ifndef DCGMSTATUS_H
#define	DCGMSTATUS_H

#include <iostream>
#include <list>
#include "dcgm_structs.h"
#include "lwos.h"

using namespace std;

class LwcmStatus {
public:
    LwcmStatus();
    
    virtual ~LwcmStatus();
    
    /*****************************************************************************
     * This method checks if the status list is empty or not
     *****************************************************************************/
    bool IsEmpty();
    
    /*****************************************************************************
     * This method is used to get the number of errors in the status list
     *****************************************************************************/
    unsigned int GetNumErrors();
    
    /*****************************************************************************
     * Add status to the list
     * @param gpuId         IN  : Represents GPU ID
     * @param fieldId       IN  : Field ID corresponding to which error is reported
     * @param errorCode     IN  : Error code corresponding to the GPU ID and Field ID
     * @return 
     *****************************************************************************/
    int Enqueue(unsigned int gpuId, short fieldId, int errorCode);
    
    /*****************************************************************************
     * Removes status from the list
     * @param pLwcmStatus   OUT :   Status to be returned to the caller
     * @return 
     *****************************************************************************/
    int Dequeue(dcgmErrorInfo_t *pLwcmStatus);
    
    /*****************************************************************************
     * Clears all the status from the list
     *****************************************************************************/
    int RemoveAll();
    
private:

    /*****************************************************************************
     * Lock/Unlock methods
     *****************************************************************************/
    int Lock();
    int UnLock();

    LWOSCriticalSection mLock; /* Lock used for accessing status list */    
    list<dcgmErrorInfo_t> mStatusList;
};

#endif	/* DCGMSTATUS_H */

