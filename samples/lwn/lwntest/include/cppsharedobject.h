/*
 * Copyright (c) 2009 - 2011 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef __CPPSHAREDOBJECT_H__
#define __CPPSHAREDOBJECT_H__

//
// cppsharedobject.h
//
// Provides a general template class, lwSharedObject<T>, that supports objects
// of type <T> that have dynamically-allocated data stores that can be shared
// between multiple object instances.  The data stores themselves are
// reference counted and freed when the last reference is removed.
//
// By default, editing the contents of an object's data store will update all
// other objects referring to the same data store.  However, this class also
// provides an edit() method that can be used to support on copy-on-write
// semantics.  The edit() method will create a deep copy of the object if
// called on an object whose data store has multiple pending references.  
//
// The implementation of this class is not thread-safe; the reference counters
// are not updated atomically and no mutual exclusion is provided for data
// store edits or references.
//
// This class is used for the lwString class to avoid duplicating the data
// store on string assignments.
//

//
// lwSharedObject is a template class used to encapsulate data of class <T> in
// a reference-counted dynamically-allocated object.
//
template <class T> class lwSharedObject
{

    // lwSharedObjectData:  Helper class corresponding to the dynamic
    // allocation, which includes an object of type <T> plus a reference
    // count.
    class lwSharedObjectData
    {
        T m_data;
        int m_counter;
    public:
        lwSharedObjectData() :              m_data(), m_counter(1) {}
        lwSharedObjectData(const T &data) : m_data(data), m_counter(1) {}
        lwSharedObjectData(const T *data) : m_data(*data), m_counter(1) {}
        ~lwSharedObjectData() {}

        int reference() { return ++m_counter; }
        int unreference() { return --m_counter; }
        int refcount() { return m_counter; }

        T *data() { return &m_data; }
        const T *data() const { return &m_data; }
    };

    lwSharedObjectData *m_object;

public:
    // Basic constructors will create a new dynamic allocation.
    lwSharedObject() : m_object(new lwSharedObjectData) {}
    lwSharedObject(const T &data) : m_object(new lwSharedObjectData(data)) {}

    // Constructing a new shared object from an existing one will share its
    // data store and increment the reference count.
    lwSharedObject(const lwSharedObject &ptr) : m_object(ptr.m_object)
    {
        m_object->reference();
    }

    // The destructor decrements its allocation's reference count and deletes
    // the data store if unused.
    ~lwSharedObject() 
    {
        if (m_object->unreference() == 0) delete m_object;
    }

    // The assignment operator copies (by reference) the allocation from the
    // RHS and then unreferences its old allocation.
    lwSharedObject &operator= (const lwSharedObject &ptr)
    {
        lwSharedObjectData *old = m_object;
        m_object = ptr.m_object;
        m_object->reference();
        if (old->unreference() == 0) delete old;
        return *this;
    }

    // The "->" operator returns a pointer to the data allocated.
    T * operator->() { return m_object ? m_object->data() : NULL; }
    const T* operator->() const { return m_object ? m_object->data() : NULL; }

    // The "*" operator returns a reference to the data allocated.
    T & operator*() { return *m_object->data(); }
    const T & operator*() const { return *m_object->data(); }

    // The edit() method is used for copy-on-edit semantics.  If the
    // allocation's reference count is 1, we can edit the object directly.
    // Otherwise, we make a new copy of the allocation and edit that.  The
    // method returns a pointer to the object's allocation.
    T *edit()
    {
        if (m_object->refcount() > 1) {
            lwSharedObjectData *old = m_object;
            m_object = new lwSharedObjectData(old->data());
            old->unreference();
        }
        return m_object->data();
    }

    // The refcount() method returns the allocation's reference count, and is
    // probably only useful for debugging.
    int refcount() const { return m_object->refcount(); }
};

#endif // #ifndef __CPPSHAREDOBJECT_H__
