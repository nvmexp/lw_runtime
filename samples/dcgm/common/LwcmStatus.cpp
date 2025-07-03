/* 
 * File:   LwcmStatus.cpp
 */

#include "LwcmStatus.h"

LwcmStatus::LwcmStatus() {
    lwosInitializeCriticalSection(&mLock);    
}

LwcmStatus::~LwcmStatus() {
    RemoveAll();
    lwosDeleteCriticalSection(&mLock);
}

/*****************************************************************************/
int LwcmStatus::Lock() 
{
    lwosEnterCriticalSection(&mLock);
    return DCGM_ST_OK;
}

/*****************************************************************************/
int LwcmStatus::UnLock() 
{
    lwosLeaveCriticalSection(&mLock);
    return DCGM_ST_OK;
}

/*****************************************************************************/
int LwcmStatus::Enqueue(unsigned int gpuId, short fieldId, int errorCode) 
{
    Lock();
    dcgmErrorInfo_t st;
    st.gpuId = gpuId;
    st.fieldId = fieldId;
    st.status = errorCode;
    mStatusList.push_back(st);
    UnLock();
    return 0;
}

/*****************************************************************************/
int LwcmStatus::Dequeue(dcgmErrorInfo_t* pLwcmStatus) {
    Lock();

    if (NULL == pLwcmStatus) {
        UnLock();
        return -1;
    }

    if (mStatusList.empty()) {
        UnLock();
        return -1;
    }

    *pLwcmStatus = mStatusList.front();
    mStatusList.pop_front();

    UnLock();
    return 0;
}

/*****************************************************************************/
int LwcmStatus::RemoveAll() {
    Lock();
    mStatusList.clear();
    UnLock();
    
    return 0;
}

/*****************************************************************************/
bool LwcmStatus::IsEmpty()
{
    Lock();
    
    if (mStatusList.size()) {
        UnLock();
        return false;
    }
    
    UnLock();
    return true;
}

/*****************************************************************************/
unsigned int LwcmStatus::GetNumErrors()
{
    unsigned int size;
    
    Lock();
    size = mStatusList.size();
    UnLock();
    
    return size;
}
