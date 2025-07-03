#ifndef _LWCM_THREAD_SAFE_H
#define _LWCM_THREAD_SAFE_H

#include <list>
#include <iterator>
#include "lwos.h"

/* a thread safe version of the std::list object */
template<class T>
class LwcmThreadSafeList : private std::list<T>
{
public:
    LwcmThreadSafeList()
    { lwosInitializeCriticalSection(&mLock); }
    ~LwcmThreadSafeList()
    { lwosDeleteCriticalSection(&mLock); }

    T Front()
    {
        T temp;
        Lock();
        temp = this->front();
        Unlock();
        return temp;
    }

    void PushBack(T val)
    {
        Lock();
        this->push_back(val);
        Unlock();
    }

    void PopFront()
    {
        Lock();
        this->pop_front();
        Unlock();
    }

    int Size()
    {
        int temp;
        Lock();
        temp = this->size();
        Unlock();
        return temp;
    }

    using std::list<T>::iterator;

    // these functions *must* be protected by a Lock/Unlock when used
    // as they are meant for iterating through a list
    using std::list<T>::begin;
    using std::list<T>::end;
    using std::list<T>::erase;

    void Lock() { lwosEnterCriticalSection(&mLock); }
    void Unlock() { lwosLeaveCriticalSection(&mLock); }
private:
    LWOSCriticalSection mLock;
};

#endif //_LWCM_THREAD_SAFE_H
