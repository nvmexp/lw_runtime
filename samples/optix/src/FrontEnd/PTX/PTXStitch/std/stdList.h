/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2006-2021, LWPU CORPORATION.  All rights reserved.
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
 *  Module name              : stdList.h
 *
 *  Description              :
 *     
 *         This module defines a data type 'list',
 *
 *         The element type of the list is represented  
 *         by the generic type 'Pointer'. Obviously, lists can hold 
 *         (pointers to) memory objects, but as special exception also 
 *         objects of integer type are allowed.
 *
 *         Several common list operations are defined, including a sorting
 *         procedure.
 */

#ifndef stdList_INCLUDED
#define stdList_INCLUDED

/*------------------------------- Includes -----------------------------------*/

#include "stdTypes.h"
#include "stdStdFun.h"
#include "stdWriter.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------- Types ------------------------------------*/

typedef struct stdList *stdList_t;

typedef struct stdList {
    stdList_t  tail;
    Pointer    head;
} stdListRec;

/*------------------------------- Includes -----------------------------------*/

#include "stdSet.h"

/*---------------------------------- Macros ----------------------------------*/

/*
 * The following macros support tail insertion into lists:
 */

#define listXList(x) \
         stdList_t x, *x##ToEnd

#define listXInit(x) \
         (x)       = Nil; \
         (x##ToEnd)= &(x)

#define listXSave(x, y) \
         (x)       = (y); \
         (x##ToEnd)= (y##ToEnd)

#define listXPutAfter(x,y) \
        { stdList_t ____x= listCons(y,Nil);  \
          *(x##ToEnd)= ____x;                \
           (x##ToEnd)= &____x->tail;         \
        }

#define listXPutInFront(x,y) \
        { if (x) { x= listCons(y,x);   } \
            else { listXPutAfter(x,y); } \
        }

#define listXPopFront(x,y) \
        { if (x) { stdList_t ____x;                \
                   y= x->head;                     \
                   ____x= x->tail;                 \
                   stdFREE(x);                     \
                   x= ____x;                       \
                   if (!(x)) { (x##ToEnd)= &(x); } \
          } else { y= Nil; }                       \
        }

#define listXConcat(x,y) \
        { if (y) {  \
             *(x##ToEnd)= (y);        \
              (x##ToEnd)= (y##ToEnd); \
        } }

#define listXRepair(x) \
        { if (x) { stdList_t ___x = x;                 \
                   while (___x->tail) ___x=___x->tail; \
                   x##ToEnd = &(___x->tail);           \
          } else { x##ToEnd = &(x); }                  \
        }

#define listXLast(x) \
        (x ? (((stdList_t)(x##ToEnd))->head) : Nil )

/*
 * Additional list functions:
 */

#define listAppend(list, element) \
        listConcat((list), listCons((element), Nil))

#define listFirst(list) \
        listIndex((list), 0)

#define listLast(list) \
        (list ? listIndex((list), listSize(list)-1) : Nil)
        
        
/*--------------------------------- Functions --------------------------------*/

/*
 * Function        : Construct extended list by combining specified element 
 *                   and tail list.
 * Parameters      : element  (I) Element to serve as head of returned list.
 *                   tail     (I) List to serve as tail of returned list.
 * Function Result : Element o tail.
 */
stdList_t STD_CDECL listCons( Pointer element, stdList_t tail );



/*
 * Function        : Add specified element to front of specified list.
 * Parameters      : element  (I)  Element to add.
 *                   list     (IO) Pointer to list to modify.
 * Function Result : 
 * NB              : This function is an analogon of listCons,
 *                   intended as traversal function.
 */
void STD_CDECL listAddTo( Pointer element, stdList_t *list );



/*
 * Function        : Discard list.
 * Parameters      : list  (I) List to discard.
 * Function Result :
 */
void STD_CDECL  listDelete( stdList_t  list );



/*
 * Function        : Delete list and free all members.
 * Parameters      : list       (I) List to discard.
 *                   deleteFun  (I) Function used to delete list member.
 * Function Result :
 */
void STD_CDECL  listObliterate( stdList_t list, stdDataFun deleteFun );



/*
 * Function        : Return element at specified position in list.
 * Parameters      : list   (I) List to inspect.
 *                   index  (I) Position in list (index origin 0).
 * Function Result : Specified element in list, 
 *                   or Nil if index >= size(list).
 */
Pointer STD_CDECL listIndex( stdList_t  list, SizeT index );



/*
 * Function        : Remove specified element from list using Pointer equality.
 * Parameters      : list    (IO) List to affect.
 *                   element (I)  Element to remove.
 * Function Result : 
 */
void STD_CDECL  listDeleteFrom( stdList_t *list, Pointer element );


/*
 * Function        : Remove-and-return head of list
 * Parameters      : list    (IO) List to pop from.
 * Function Result : (former) head of list
 */
Pointer STD_CDECL listPop( stdList_t *list );
#define listPush(list,element)    listAddTo(element,list)


/*
 * Function        : Remove specified element from list using Pointer equality.
 * Parameters      : element (I)  Element to remove.
 *                   list    (IO) List to affect.
 * Function Result : -
 * NB              : This function is an analogon of listDeleteFrom,
 *                   intended as traversal function.
 */
void STD_CDECL  listRemoveElement( Pointer element, stdList_t *list );



/*
 * Function        : Test oclwrrence in list using Pointer equality.
 * Parameters      : list     (I) List to test.
 *                   element  (I) Element to test for oclwrrence.
 * Function Result : True if and only if element is a member of list.
 */
Bool STD_CDECL listContains( stdList_t list, Pointer element );



/*
 * Function        : Test oclwrrence in list, according to specified equality function
 * Parameters      : list     (I) list to test
 *                   element  (I) element to test for oclwrrence
 *                   equal    (I) equality function for element type
 * Function Result : True if and only if element is a member of list
 */
Bool STD_CDECL listContainsObject( stdList_t list, Pointer element, stdEqualFun equal);

#define listContainsString(l,s)    listContainsObject(l,s,(stdEqualFun)stdStringEqual)



/*
 * Function        : Remove element at specified position in list
 * Parameters      : list    (IO) List to affect.
 *                   index   (I)  Index of element to remove.
 * Function Result : 
 */
void STD_CDECL listRemoveIndex( stdList_t *list, SizeT index );



/*
 * Function        : Apply specified function to all elements of the specified list,
 *                   with specified generic data element as additional parameter.
 *                   The list is traversed from head to tail. 
 * Parameters      : list       (I) List to traverse.
 *                   traverse   (I) Function to apply to all elements.
 *                   data       (I) Generic data element passed as additional
 *                                  parameter to every invocation of 'traverse'.
 * Function Result : 
 */
void STD_CDECL  listTraverse( stdList_t  list, stdEltFun traverse, Pointer data );



/*
 * Function        : Apply specified function to all elements of the specified list,
 *                   with specified generic data element as additional parameter.
 *                   The list is traversed from head to tail. 
 * Parameters      : list       (I) List to traverse.
 *                   traverse   (I) Function to apply to all elements.
 *                                  Contrary to parameter 'traverse' in listTraverse
 *                                  which is applied to the list's elements,
 *                                  the traversal function is here applied to the list
 *                                  blocks that hold these elements.
 *                   data       (I) Generic data element passed as additional
 *                                  parameter to every invocation of 'traverse'.
 * Function Result : 
 */
void STD_CDECL  listRawTraverse( stdList_t  list, stdEltFun traverse, Pointer data );



/*
 * Function        : Return number of elements in list.
 * Parameters      : list  (I) List to inspect.
 * Function Result : Number of elements in  list.
 */
SizeT STD_CDECL listSize( stdList_t  list );



/*
 * Function        : Copy a list.
 * Parameters      : list    (I) List to copy.
 * Function Result : Copy of list. The head objects are not copied! .
 */
stdList_t STD_CDECL listCopy( stdList_t list );



/*
 * Function        : Concatenate two lists.
 * Parameters      : head    (I) Head of the new list.
 *                   tail    (I) Tail of the new list.
 * Function Result : Concatenate two lists. No copies are made, i.e.
 *                   this function may cause tail sharing.
 */
stdList_t STD_CDECL listConcat( stdList_t head, stdList_t tail );



/*
 * Function        : Reverse a list (destructive).
 * Parameters      : list    (I) List to reverse.
 * Function Result : Reversed list.
 */
stdList_t STD_CDECL listReverse( stdList_t list );



/*
 * Function        : List hash function
 * Parameters      : l       (I) list to hash
 *                   hash    (I) Hash function for element type.
 * Function Result : List hash value
 */
uInt STD_CDECL listHash( stdList_t l, stdHashFun hash);



/*
 * Function        : List equality function
 * Parameters      : l,r     (I)  list to compare
 *                   equal   (I)  Equality function for element type.
 * Function Result : True iff l and r are equal according to specified
 *                   list element equality function
 */
Bool STD_CDECL listEqual( stdList_t l, stdList_t r, stdEqualFun equal);



/*
 * Function        : Test if list is a prefix of another list,
 *                   according to specified equality function
 * Parameters      : prefix  (I) prefix
 *                   list    (I) list to test
 *                   equal   (I) Equality function for element type.
 * Function Result : True iff 'prefix' is a prefix of 'list'
 */
Bool STD_CDECL listIsPrefix( stdList_t prefix, stdList_t list, stdEqualFun equal);



/*
 * Function        : Sort list into increasing element order.
 * Parameters      : list    (IO) Pointer to list to sort.
 *                   lessEq  (I)  Comparison function defining some total
 *                                ordering of the list's elements.
 * Function Result :
 */
void STD_CDECL listSort( stdList_t *list, stdLessEqFun lessEq);

/* same as listSort but als use generic 'data' pointer when comparing */
void STD_CDECL listSortD( stdList_t *list, stdLessEqDFun lessEqD, Pointer data);


/*
 * Function        : Sort list into increasing element order.
 * Parameters      : list     (IO) Pointer to list to sort.
 *                   rawLessEq (I) Comparison function defining some total
 *                                 ordering of the list's elements.
 *                                 Contrary to parameter lessEq in listSort
 *                                 which is applied to the list's elements,
 *                                 parameter rawLessEq is applied to the list
 *                                 blocks that hold these elements.
 * Function Result :
 */
void STD_CDECL listRawSort( stdList_t *list, stdLessEqFun rawLessEq);

/* same as listRawSort but als use generic 'data' pointer when comparing */
void STD_CDECL listRawSortD( stdList_t *list, stdLessEqDFun rawLessEqD, Pointer data);


/*
 * Function        : Create a set form of the list.
 * Parameters      : list    (I) List to colwert.
 *                   hash    (I) Hash function, mapping the set element
 *                               type to an arbitrary integer.
 *                   equal   (I) Equality function for element type.
 * Function Result : Colwerted list.
 */
stdSet_t STD_CDECL listToSet( stdList_t list, stdHashFun hash, stdEqualFun equal);


/*
 * Function        : Print list via writer object.
 * Parameters      : wr      (I) Writer to print to
 *                   list    (I) List to print.
 * Function Result : 
 */
void STD_CDECL listPrint( stdWriter_t wr, stdList_t list );


/*
 * Function        : Insert the new element before specified element in the
 *                   list using Pointer equality.
 * Parameters      : element      (I)  Element to add.
 *                   matchElement (I)  Element to match
 *                   list         (IO) List to affect.
 * Function Result : True iff matchElement is found
 */
Bool STD_CDECL listPutBefore( Pointer element, Pointer matchElement, stdList_t *list );

/*--------------------------------- Iterator --------------------------------*/
// Create iterator type and functions, so can write code like:
// stdListIterator_t it;
// FOREACH_LIST_VALUE(list,it) {
//   d = listValue(it);
// }
typedef stdList_t stdListIterator_t;

stdListIterator_t listBegin (stdList_t list);
Bool listAtEnd (stdListIterator_t it);
stdListIterator_t listNext (stdListIterator_t it);
Pointer listValue (stdListIterator_t it);
// no need to delete iterator since it didn't allocate new memory

#define FOREACH_LIST_VALUE(list,it) \
    for (it = listBegin(list); !listAtEnd(it); it = listNext(it))


#define LIST_TRAVERSE_MULTI_ARGS(list, traverse, ...)\
{   stdListIterator_t listIter;\
    FOREACH_LIST_VALUE(list, listIter) {\
        traverse(listValue(listIter), ##__VA_ARGS__);\
    }\
}

#ifdef __cplusplus
}
#endif

#endif
