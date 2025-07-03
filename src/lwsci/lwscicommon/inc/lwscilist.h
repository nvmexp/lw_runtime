/*
 * Copyright © 2010 Intel Corporation
 * Copyright © 2010 Francisco Jerez <lwrrojerez@riseup.net>
 * Copyright © 2012-2022 LWPU Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *
 */

/*
 * This file was copied from the X.Org X server source at commit
 * 5884e7dedecdd82ddbb037360cf9c85143e094b5 and modified to match LWPU's X
 * driver code style.
 */

/*
 * This file was copied from commit 7c90bc88b959c9b54c4d7bdd1f473589b01809f2
 * and modified to be used for lwsci*
 */
#ifndef INCLUDED_LW_LIST_H
#define INCLUDED_LW_LIST_H

#include <stdbool.h>
#include <stddef.h>
#include "lwscicommon_os.h"
#include "lwscilog.h"

#ifdef __cplusplus
extern "C" {
#endif //__cplusplus

/**
 * @defgroup lwscicommon_blanket_statements LwSciCommon blanket statements.
 * Generic statements applicable for LwSciCommon interfaces.
 * @{
 */

/**
 * \page page_blanket_statements LwSciCommon blanket statements
 * \section in_out_params Input parameters
 * - LWListRec* passed as an input parameter to an API is a valid input if it is
 * previously returned by a successful call to lwListInit or lwListAppend or
 * lwListDel.
 */

/**
 * @}
 */

/**
 * @defgroup lwscicommon_platformutils_api LwSciCommon APIs for platform utils.
 * List of APIs exposed at Inter-Element level.
 * @{
 */

#if defined(LW_WINDOWS) || defined(__cplusplus)
    #undef HAVE_TYPEOF
#else
    #define HAVE_TYPEOF 1
#endif

/** Macro for determining the offset of "member" in "type". */
#if !defined(LW_OFFSETOF)
    #if defined(__GNUC__)
        #define LW_OFFSETOF(type, member)   __builtin_offsetof(type, member)
    #else
        #define LW_OFFSETOF(type, member)   ((size_t)(&(((type *)0)->member)))
    #endif
#endif

/**
 * @file Classic doubly-link cirlwlar list implementation.
 * For real usage examples of the linked list, see the file test/list.c
 *
 * Example:
 * We need to keep a list of struct foo in the parent struct bar, i.e. what
 * we want is something like this.
 *
 *     struct bar {
 *          ...
 *          struct foo *list_of_foos; -----> struct foo {}, struct foo {}, struct foo{}
 *          ...
 *     }
 *
 * We need one list head in bar and a list element in all list_of_foos (both are of
 * data type 'LWListRec').
 *
 *     struct bar {
 *          ...
 *          LWListRec list_of_foos;
 *          ...
 *     }
 *
 *     struct foo {
 *          ...
 *          LWListRec entry;
 *          ...
 *     }
 *
 * Now we initialize the list head:
 *
 *     struct bar bar;
 *     ...
 *     lwListInit(&bar.list_of_foos);
 *
 * Then we create the first element and add it to this list:
 *
 *     struct foo *foo = malloc(...);
 *     ....
 *     lwListAdd(&foo->entry, &bar.list_of_foos);
 *
 * Repeat the above for each element you want to add to the list. Deleting
 * works with the element itself.
 *      lwListDel(&foo->entry);
 *      free(foo);
 *
 * Note: calling lwListDel(&bar.list_of_foos) will set bar.list_of_foos to an empty
 * list again.
 *
 * Looping through the list requires a 'struct foo' as iterator and the
 * name of the field the subnodes use.
 *
 * struct foo *iterator;
 * lwListForEachEntry(iterator, &bar.list_of_foos, entry) {
 *      if (iterator->something == ...)
 *             ...
 * }
 *
 * Note: You must not call lwListDel() on the iterator if you continue the
 * loop. You need to run the safe for-each loop instead:
 *
 * struct foo *iterator, *next;
 * lwListForEachEntry_safe(iterator, next, &bar.list_of_foos, entry) {
 *      if (...)
 *              lwListDel(&iterator->entry);
 * }
 *
 */

/**
 * \brief Structure acts as linkage between list objects.
 *  LWListRec holds pointer to next list object and previous list object.
 *  The structure must be part of user to-be-linked structure.
 *  User to-be-linked structure is user specified structure for which
 *  linked list needs to be created using LWListRec as linkage between
 *  two user specified structures.
 *
 */
struct LWList {
    /** Pointer to next list object. */
    struct LWList *next;
    /** Pointer to previous list object. */
    struct LWList *prev;
};
typedef struct LWList LWListRec;

/**
 * \brief Initialize the list as an empty list. Empty list implies a list
 *  with a single LWListRec. The LWListRec is designated as head of the
 *  list being initialized.
 *
 * \param[in] list pointer to the head LWListRec of the list that needs
 *  to be initialized. Valid value: @a list is not NULL
 *
 * \return void
 * - Panics if any of the following oclwrs:
 *      - @a list is NULL
 *
 * \conlwrrency:
 *  - Thread-safe: No
 *  - Synchronization
 *      - The user must ensure that the list is not accessed by multiple
 *        threads at the same time.
 *
 * \implements{18851253}
 * \implements{21750002}
 * \implements{18850740}
 */
static inline void
lwListInit(LWListRec* list)
{
    if (NULL == list) {
        LWSCI_ERR_STR("Invalid Head list object as input parameter");
        LwSciCommonPanic();
    }

    list->prev = list;
    list->next = list;
}

static inline void
lwListAddPriv(LWListRec* entry, LWListRec* prev, LWListRec* next)
{
    next->prev = entry;
    entry->next = next;
    entry->prev = prev;
    prev->next = entry;
}

/**
 * \brief Appends a new LWListRec pointed by input LWListRec* to the tail of the
 * list represented by input head LWListRec*.
 *
 * \param[in] entry The new LWListRec* to be added in the list.
 *  Valid value: @a entry is not NULL.
 * \param[in,out] head Head LWListRec* of existing list.
 *  Valid value: @a head is not NULL.
 *
 * \return void
 * - Panics if any of the following oclwrs:
 *      - @a entry is NULL
 *      - @a head is NULL
 *
 * \conlwrrency:
 *  - Thread-safe: No
 *  - Synchronization
 *      - The user must ensure that the list is not accessed by multiple
 *        threads at the same time.
 *
 * \implements{18851256}
 * \implements{21750003}
 * \implements{18850737}
 */
static inline void
lwListAppend(LWListRec* entry, LWListRec* head)
{
    if ((NULL == entry) || (NULL == head)) {
        LWSCI_ERR_STR("Invalid list object as input parameter");
        LwSciCommonPanic();
    }

    lwListAddPriv(entry, head->prev, head);
}

static inline void
lwListDelPriv(LWListRec* prev, LWListRec* next)
{
    next->prev = prev;
    prev->next = next;
}

/**
 * \brief Removes the LWListRec pointed by input LWListRec* from the list it
 *  is in. Using this function will reset the pointers to/from this
 *  LWListRec* such that it is removed from the list. It does not free the
 *  LWListRec pointed by the input LWListRec* itself. Using lwListDel
 *  on a head LWListRec* will not remove the first LWListRec from the list
 *  but rather reset the list as an empty list.
 *
 * \param[in] entry The LWListRec* to be removed.
 *  Valid value: @a entry is not NULL.
 *
 * \return void
 * - Panics if any of the following oclwrs:
 *      - @a entry is NULL
 *
 * \conlwrrency:
 *  - Thread-safe: No
 *  - Synchronization
 *      - The user must ensure that the list is not accessed by multiple
 *        threads at the same time.
 *
 * \implements{18851259}
 * \implements{21750004}
 * \implements{18850734}
 */
static inline void
lwListDel(LWListRec* entry)
{
    if (NULL == entry) {
        LWSCI_ERR_STR("Invalid list object as input parameter");
        LwSciCommonPanic();
    }

    lwListDelPriv(entry->prev, entry->next);
    lwListInit(entry);
}

/**
 * \brief Returns a pointer to the container of this list object.
 *
 * Example:
 * struct foo* f;
 * f = lw_container_of(&foo->entry, struct foo, entry);
 * assert(f == foo);
 *
 * \param ptr Pointer to the LWListRec.
 * \param type Data type of the list element.
 * \param member Member name of the LWListRec field in the list element.
 * \return A pointer to the data struct containing the list head.
 */
#ifndef lw_container_of
    #ifdef HAVE_TYPEOF
        #define lw_container_of(ptr, type, member) \
            (typeof(type) *)(void *)((char *)(ptr) - \
            LW_OFFSETOF(typeof(type), member))
    #else
        #define lw_container_of(ptr, type, member) \
            (type *)(void *)((char *)(ptr) - \
            LW_OFFSETOF(type, member))
    #endif
#endif

#ifdef HAVE_TYPEOF
#define lw_container_of_priv(ptr, sample, member)  \
    lw_container_of(ptr, typeof(*sample), member)
#else
/* This implementation of lw_container_of_priv has undefined behavior according
 * to the C standard, but it works in many cases.  If your compiler doesn't
 * support typeof() and fails with this implementation, please try a newer
 * compiler.
 */
#define lw_container_of_priv(ptr, sample, member)                      \
    (void *)((char *)(ptr)                                             \
            - ((char *)&(sample)->member - (char *)(sample)))
#endif

/**
 * \brief Loop through the list given by input head LWListRec and returns
 *  a pointer to next user to-be-linked struture in the list.
 *  This iterator is not safe for user to-be-linked structure deletion.
 *  Use lwListForEachEntry_safe instead.
 *
 * \param[in,out] pos iterator variable of the type pointer to
 *  user to-be-linked structure.
 *  Valid value: pos is not NULL.
 * \param[in] head pointer to the head LWListRec of the list
 *  Valid value: @a head is not NULL and previously operated by lwListInit or
 *  lwListAppend or lwListDel.
 * \param[in] member Member name of the LWListRec in the user to-be-linked
 *  structure. Valid value: @a member is valid member name in the
 *  user to-be-linked structure.
 *
 */
#define lwListForEachEntry(pos, head, member)                               \
    for ((pos) = (lw_container_of_priv(((head)->next), (pos), member));     \
         (&(pos)->member) != (head);                                        \
         (pos) = (lw_container_of_priv(((pos)->member.next), (pos), member)))

/**
 * \brief Loop through the list, keeping a backup pointer to the next
 *  user to-be-linked structure.
 *  This allows for the deletion of user to-be-linked structure
 *  while looping through the list.
 *  lwListForEachEntry_safe needs to be used in conjunction with
 *  lwListForEachEntryEnd_safe at the end of the loop.
 *
 * \param[in,out] pos iterator variable of the type pointer to
 *  user to-be-linked structure.
 *  Valid value: pos is not NULL.
 * \param[in] tmp temporary iterator to hold backup pointer to the next
 *  LWListRec.
 *  Valid value: tmp is not NULL.
 * \param[in] head pointer to the head LWListRec of the list.
 *  Valid value: @a head is not NULL and previously operated by lwListInit or
 *  lwListAppend or lwListDel.
 * \param[in] member Member name of the LWListRec in the
 *  user to-be-linked structure. Valid value: @a member is valid member name
 *  in the user to-be-linked structure.
 *
 */
#define lwListForEachEntry_safe(pos, tmp, head, member)                 \
    (pos) = (lw_container_of_priv(((head)->next), (pos), member));      \
    (tmp) = (lw_container_of_priv(((pos)->member.next), (pos), member));\
    while ((&(pos)->member) != (head))

/**
 * \brief Increment condition for lwListForEachEntry_safe looping construct.
 *  This assigns pointer to the next user to-be-linked structure in the list
 *  to the iterator.
 *  lwListForEachEntry_safe needs to be used in conjunction with
 *  lwListForEachEntryEnd_safe at the end of the loop.
 *
 * \param[in,out] pos iterator variable of the type pointer to
 *  user to-be-linked structure.
 *  Valid value: pos is not NULL.
 * \param[in] tmp temporary iterator to hold backup pointer to the
 *  next user to-be-linked structure in the list.
 *  Valid value: tmp is not NULL.
 * \param[in] member Member name of the LWListRec in the
 *  user to-be-linked structure.
 *  Valid value: @a member is valid member name in the
 *  user to-be-linked structure.
 *
 */
#define lwListForEachEntryEnd_safe(pos, tmp, member)                    \
    (pos) = (tmp);                                                      \
    (tmp) = (lw_container_of_priv(((pos)->member.next), (tmp), member));

#if (LW_IS_SAFETY == 0)
/**
 * Alias of lw_container_of
 */
#define lwListEntry(ptr, type, member) \
    lw_container_of(ptr, type, member)

/**
 * Retrieve the last list entry for the given listpointer.
 *
 * Example:
 * struct foo *first;
 * first = lwListLastEntry(&bar->list_of_foos, struct foo, entry);
 *
 * \param ptr The list head
 * \param type Data type of the list element to retrieve
 * \param member Member name of the LWListRec field in the list element.
 * \return A pointer to the last list element.
 */
#define lwListLastEntry(ptr, type, member) \
    lwListEntry((ptr)->prev, type, member)

/**
 * Retrieve the next list entry for the given listpointer.
 *
 * Example:
 * struct foo *next;
 * next = lwListNextEntry(&bar->list_of_foos, struct foo, entry);
 *
 * \param ptr The list head
 * \param type Data type of the list element to retrieve
 * \param member Member name of the LWListRec field in the list element.
 * \return A pointer to the last list element.
 */
#define lwListNextEntry(ptr, type, member) \
    lwListEntry((ptr)->next, type, member)



/**
 * Retrieve the first list entry for the given list pointer.
 *
 * Example:
 * struct foo *first;
 * first = lwListFirstEntry(&bar->list_of_foos, struct foo, entry);
 *
 * \param ptr The list head
 * \param type Data type of the list element to retrieve
 * \param member Member name of the LWListRec field in the list element.
 * \return A pointer to the first list element.
 */
#define lwListFirstEntry(ptr, type, member) \
    lwListEntry((ptr)->next, type, member)


/**
 * Check if the list is empty.
 *
 * Example:
 * lwListIsEmpty(&bar->list_of_foos);
 *
 * \return True if the list contains one or more elements or False otherwise.
 */
static inline bool
lwListIsEmpty(LWListRec* head)
{
    return head->next == head;
}

static inline int
lwListCount(const LWListRec *head)
{
    LWListRec* next;
    int count = 0;

    for (next = head->next; next != head; next = next->next) {
        count++;
    }

    return count;
}

/**
 * Check if entry is present in the list.
 *
 * Example:
 * lwListPresent(&foo->entry, &bar->list_of_foos);
 *
 * \return 1 if the list contains the specified entry; otherwise, return 0.
 */
static inline bool
lwListPresent(const LWListRec *entry, const LWListRec *head)
{
    const LWListRec *next;

    for (next = head->next; next != head; next = next->next) {
        if (next == entry) {
            return true;
        }
    }

    return false;
}


/**
 * Insert a new element after the given list head. The new element does not
 * need to be initialised as empty list.
 * The list changes from:
 *      head → some element → ...
 * to
 *      head → new element → older element → ...
 *
 * Example:
 * struct foo *newfoo = malloc(...);
 * lwListAdd(&newfoo->entry, &bar->list_of_foos);
 *
 * \param[in] entry : The new element to add to the list.
 * \param[in] head : The existing list.
 *
 * \return void
 */
static inline void
lwListAdd(LWListRec* entry, LWListRec* head)
{
    lwListAddPriv(entry, head, head->next);
}

/**
 * Initialize the list as an empty list.
 *
 * This is functionally the same as lwListInit, but can be used for
 * initialization of global variables.
 *
 * Example:
 * static LWListRec list_of_foos = LW_LIST_INIT(&list_of_foos);
 *
 * \param The list to initialized.
 */
#define LW_LIST_INIT(head) { .prev = (head), .next = (head) }


/* NULL-Terminated List Interface
 *
 * The interface below does _not_ use the LWListRec as described above.
 * It is mainly for legacy structures that cannot easily be switched to
 * LWListRec.
 *
 * This interface is for structs like
 *      struct foo {
 *          [...]
 *          struct foo *next;
 *           [...]
 *      };
 *
 * The position and field name of "next" are arbitrary.
 */

/**
 * Init the element as null-terminated list.
 *
 * Example:
 * struct foo *list = malloc();
 * lwNTListInit(list, next);
 *
 * \param list The list element that will be the start of the list
 * \param member Member name of the field pointing to next struct
 */
#define lwNTListInit(_list, _member) \
        (_list)->_member = NULL

/**
 * Returns the next element in the list or NULL on termination.
 *
 * Example:
 * struct foo *element = list;
 * while ((element = lwNTListNext(element, next)) { }
 *
 * This macro is not safe for node deletion. Use lwListForEachEntry_safe
 * instead.
 *
 * \param list The list or current element.
 * \param member Member name of the field pointing to next struct.
 */
#define lwNTListNext(_list, _member) \
        (_list)->_member

/**
 * Iterate through each element in the list.
 *
 * Example:
 * struct foo *iterator;
 * lwNTListForEachEntry(iterator, list, next) {
 *      [modify iterator]
 * }
 *
 * \param entry Assigned to the current list element
 * \param list The list to iterate through.
 * \param member Member name of the field pointing to next struct.
 */
#define lwNTListForEachEntry(_entry, _list, _member)            \
        for (_entry = _list; _entry; _entry = (_entry)->_member)

/**
 * Iterate through each element in the list, keeping a backup pointer to the
 * element. This macro allows for the deletion of a list element while
 * looping through the list.
 *
 * See lwNTListForEachEntry for more details.
 *
 * \param entry Assigned to the current list element
 * \param tmp The pointer to the next element
 * \param list The list to iterate through.
 * \param member Member name of the field pointing to next struct.
 */
#define lwNTListForEachEntrySafe(_entry, _tmp, _list, _member)          \
        for (_entry = _list, _tmp = (_entry) ? (_entry)->_member : NULL;\
                _entry;                                                 \
                _entry = _tmp, _tmp = (_tmp) ? (_tmp)->_member: NULL)

/**
 * Append the element to the end of the list. This macro may be used to
 * merge two lists.
 *
 * Example:
 * struct foo *elem = malloc(...);
 * lwNTListInit(elem, next)
 * lwNTListAppend(elem, list, struct foo, next);
 *
 * Resulting list order:
 * list_item_0 -> list_item_1 -> ... -> elem_item_0 -> elem_item_1 ...
 *
 * \param entry An entry (or list) to append to the list
 * \param list The list to append to. This list must be a valid list, not
 * NULL.
 * \param type The list type
 * \param member Member name of the field pointing to next struct
 */
#define lwNTListAppend(_entry, _list, _type, _member)                   \
    do {                                                                \
        _type *__iterator = _list;                                      \
        while (__iterator->_member) { __iterator = __iterator->_member;}\
        __iterator->_member = _entry;                                   \
    } while (0)

/**
 * Insert the element at the next position in the list. This macro may be
 * used to insert a list into a list.
 *
 * struct foo *elem = malloc(...);
 * lwNTListInit(elem, next)
 * lwNTListInsert(elem, list, struct foo, next);
 *
 * Resulting list order:
 * list_item_0 -> elem_item_0 -> elem_item_1 ... -> list_item_1 -> ...
 *
 * \param entry An entry (or list) to append to the list
 * \param list The list to insert to. This list must be a valid list, not
 * NULL.
 * \param type The list type
 * \param member Member name of the field pointing to next struct
 */
#define lwNTListInsert(_entry, _list, _type, _member)                   \
    do {                                                                \
        lwNTListAppend((_list)->_member, _entry, _type, _member);       \
        (_list)->_member = _entry;                                      \
    } while (0)

/**
 * Delete the entry from the list by iterating through the list and
 * removing any reference from the list to the entry.
 *
 * Example:
 * struct foo *elem = <assign to right element>
 * lwNTListDel(elem, list, struct foo, next);
 *
 * \param entry The entry to delete from the list. entry is always
 * re-initialized as a null-terminated list.
 * \param list The list containing the entry, set to the new list without
 * the removed entry.
 * \param type The list type
 * \param member Member name of the field pointing to the next entry
 */
#define lwNTListDel(_entry, _list, _type, _member)              \
        do {                                                    \
                _type *__e = _entry;                            \
                if (__e == NULL || _list == NULL) break;        \
                if ((_list) == __e) {                           \
                    _list = __e->_member;                       \
                } else {                                        \
                    _type *__prev = _list;                      \
                    while (__prev->_member && __prev->_member != __e)   \
                        __prev = lwNTListNext(__prev, _member); \
                    if (__prev->_member)                        \
                        __prev->_member = __e->_member;         \
                }                                               \
                lwNTListInit(__e, _member);                     \
        } while(0)

#endif // LW_IS_SAFETY

/**
 * @}
 */

#ifdef __cplusplus
}
#endif //__cplusplus

#endif /* INCLUDED_LW_LIST_H */
