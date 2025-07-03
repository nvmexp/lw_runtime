/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2006-2020, LWPU CORPORATION.  All rights reserved.
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
 *  Module name              : stdList.c
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

/*------------------------------- Includes -----------------------------------*/

#include "stdList.h"
#include "stdLocal.h"

/*--------------------------------- Functions --------------------------------*/


/*
 * Function        : Construct extended list by combining specified element 
 *                   and tail list.
 * Parameters      : head  (I) Element to serve as head of returned list
 *                   tail  (I) List to serve as tail of returned list
 * Function Result : head o tail
 */
stdList_t STD_CDECL listCons ( Pointer element, stdList_t tail )
{
    stdList_t result;
    
    stdNEW(result);
    result->head= element;
    result->tail= tail;
    
    return result;
}



/*
 * Function        : Add specified element to front of specified list.
 * Parameters      : element  (I)  Element to add
 *                   list     (IO) Pointer to list to modify
 * Function Result : -
 * NB              : This function is an analogon of listCons,
 *                   intended as traversal function.
 */
void STD_CDECL listAddTo( Pointer element, stdList_t *list )
{
    *list= listCons(element,*list);        
}



/*
 * Function        : Discard list
 * Parameters      : list  (I) list to discard
 * Function Result : -
 */
void STD_CDECL listDelete ( stdList_t list )
{
    while (list) {
        stdList_t tail= list->tail;
        stdFREE(list);
        list= tail;
    }
}



/*
 * Function        : Delete a list and all of its members
 * Parameters      : list       (I) list to discard
 *                   deleteFun  (I) function used to delete list member
 * Function Result : -
 */
void STD_CDECL listObliterate ( stdList_t list, stdDataFun deleteFun )
{
    while (list) {
        stdList_t tail= list->tail;

        if (deleteFun == Nil) { stdFREE(list->head);   }
                          else { deleteFun(list->head); }

        stdFREE(list);
        list= tail;
    }
}



/*
 * Function        : Return element at specified position in list
 * Parameters      : list   (I) List to inspect
 *                   index  (I) Position in list (index origin 0)
 * Function Result : Specified element in list, 
 *                   or Nil if index >= size(list)
 */
Pointer STD_CDECL listIndex ( stdList_t  list, SizeT index )
{
    while (list && index) {
        list= list->tail;
        index--;
    }
    
    if (list == Nil) { return Nil;       } 
                 else { return list->head; }
}

/*
 * Function        : Remove specified element from list using Pointer equality.
 * Parameters      : list    (IO) List to affect.
 *                   element (I)  Element to remove.
 * Function Result : 
 */
void STD_CDECL  listDeleteFrom( stdList_t *list, Pointer element )
{
    while (*list) {
        stdList_t l= *list;
        
        if (l->head == element) {
            *list = l->tail;
            stdFREE(l);
            return;
        }
        list = &l->tail;
    } 
}


/*
 * Function        : Remove-and-return head of list
 * Parameters      : list    (IO) List to pop from.
 * Function Result : (former) head of list
 */
Pointer STD_CDECL listPop( stdList_t *list )
{
    if (!*list) {
        return Nil;
    } else {
        stdList_t  tmp    = *list;
        Pointer    result = tmp->head;
       *list = tmp->tail;
        stdFREE(tmp);
        
        return result;
    }
}


/*
 * Function        : Remove specified element from list using Pointer equality.
 * Parameters      : element (I)  Element to remove.
 *                   list    (IO) List to affect.
 * Function Result : -
 * NB              : This function is an analogon of listDeleteFrom,
 *                   intended as traversal function.
 */
void STD_CDECL  listRemoveElement( Pointer element, stdList_t *list )
{
    while (*list) {
        stdList_t l= *list;
        
        if (l->head == element) {
            *list = l->tail;
            stdFREE(l);
            return;
        }
        list = &l->tail;
    } 
}



/*
 * Function        : Test oclwrrence in list using Pointer equality.
 * Parameters      : list     (I) list to test
 *                   element  (I) element to test for oclwrrence
 * Function Result : True if and only if element is a member of list
 */
Bool STD_CDECL  listContains( stdList_t list, Pointer element )
{
    while (list != Nil) {
        if (list->head == element) { return True;       }
                              else { list = list->tail; }
    }

    return False;
}



/*
 * Function        : Test oclwrrence in list, according to specified equality function
 * Parameters      : list     (I) list to test
 *                   element  (I) element to test for oclwrrence
 *                   equal    (I) equality function for element type
 * Function Result : True if and only if element is a member of list
 */
Bool STD_CDECL  listContainsObject( stdList_t list, Pointer element, stdEqualFun equal)
{
    while (list != Nil) {
        if (equal(list->head,element)) { return True;       }
                                       else { list = list->tail; }
    }

    return False;
}



/*
 * Function        : Remove element at specified position in list
 * Parameters      : list    (IO) List to affect.
 *                   index   (I)  Index of element to remove.
 * Function Result : 
 */
void STD_CDECL listRemoveIndex( stdList_t *list, SizeT index )
{
    while (*list) {
        stdList_t l= *list;
        
        if (index-- == 0) {
            *list = l->tail;
            stdFREE(l);
            return;
        }
        list = &l->tail;
    } 
}



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
void STD_CDECL  listRawTraverse( stdList_t  list, stdEltFun traverse, Pointer data )
{
    while (list) {
        stdList_t tail= list->tail;
        
        traverse(list, data);
        
        list= tail;
    }
}



/*
 * Function        : Apply specified function to all elements of the specified list,
 *                   with specified generic data element as additional parameter.
 *                   The list is traversed from head to tail. 
 * Parameters      : list       (I) list to traverse
 *                   traverse   (I) function to apply to all elements
 *                   data       (I) generic data element passed as additional
 *                                  parameter to every invocation of 'traverse'
 * Function Result : -
 */
void STD_CDECL listTraverse ( stdList_t  list, stdEltFun traverse, Pointer data )
{
    while (list) {
        stdList_t tail= list->tail;
        
        traverse(list->head, data);
        
        list= tail;
    }
}



/*
 * Function        : Return number of elements in list.
 * Parameters      : list  (I) List to inspect
 * Function Result : number of elements in  list
 */
SizeT STD_CDECL listSize ( stdList_t  list )
{
    SizeT result= 0;
    
    while (list) {
        list= list->tail;
        result++;
    }
    
    return result;        
}




/*
 * Function        : List hash function
 * Parameters      : l       (I) list to hash
 *                   hash    (I) Hash function for element type.
 * Function Result : List hash value
 */
uInt STD_CDECL listHash( stdList_t l, stdHashFun hash)
{
    uInt result= 0;

    while (l) {
        result ^= hash(l->head);
        l = l->tail;
    }
    
    return result; 
}



/*
 * Function        : List equality function
 * Parameters      : l,r     (I)  list to compare
 *                   equal   (I)  Equality function for element type.
 * Function Result : True iff l and r are equal according to specified
 *                   list element equality function
 */
Bool STD_CDECL  listEqual( stdList_t l, stdList_t r, stdEqualFun equal)
{
    while (l && r) {
        if (equal(l->head,r->head)) {
            l = l->tail;
            r = r->tail;
        } else {
            return False;
        }
    }
    
    return !l && !r; 
}



/*
 * Function        : Test if list is a prefix of another list,
 *                   according to specified equality function
 * Parameters      : prefix  (I) prefix
 *                   list    (I) list to test
 *                   equal   (I) Equality function for element type.
 * Function Result : True iff 'prefix' is a prefix of 'list'
 */
Bool STD_CDECL  listIsPrefix( stdList_t prefix, stdList_t list, stdEqualFun equal)
{
    while (prefix && list) {
        if (equal(prefix->head,list->head)) {
            prefix = prefix->tail;
            list   = list  ->tail;
        } else {
            return False;
        }
    }
    
    return !prefix; 
}



/*
 * Function        : Sort list into increasing element order.
 * Parameters      : list    (IO) Pointer to list to sort.
 *                   lessEq  (I)  Comparison function defining some total
 *                                ordering of the list's elements.
 * Function Result :
 */
void STD_CDECL  listSort( stdList_t *list, stdLessEqFun lessEq)
{
    stdList_t  h,tl;
    stdList_t  l    = *list;
    stdList_t  half1= Nil;
    stdList_t  half2= Nil;
    
   /*
    * Lists of size <=1 are already 
    * sorted, so stop here:
    */
    if ( l == Nil 
      || l->tail == Nil 
       ) { return; }

   /*
    * Split list into two halves:
    */
    while (l != Nil) {
        tl = l->tail;
    
        l->tail = half1;
        half1   = l;
        
        l  = tl;

        stdSWAP(half1,half2,Pointer);
    }

   /*
    * Sort each of the halves; 
    * use temporary variables to not spoil
    * register allocation for half1,2:
    */
    h= half1; listSort(&h, lessEq); half1= h;
    h= half2; listSort(&h, lessEq); half2= h;

   /*
    * And merge the two halves into
    * the sorted result:
    */
    while ( half1 && half2 ) {
        if (lessEq( half1->head, half2->head)) {
           *list  = half1;
            half1 = half1->tail;        
        } else {
           *list  = half2;
            half2 = half2->tail;
        }
    
        list = &((*list)->tail);
    }

    if (half1) { *list = half1; }
          else { *list = half2; }
}

void STD_CDECL  listSortD( stdList_t *list, stdLessEqDFun lessEqD, Pointer data)
{
    stdList_t  h,tl;
    stdList_t  l    = *list;
    stdList_t  half1= Nil;
    stdList_t  half2= Nil;
    
   /*
    * Lists of size <=1 are already 
    * sorted, so stop here:
    */
    if ( l == Nil 
      || l->tail == Nil 
       ) { return; }

   /*
    * Split list into two halves:
    */
    while (l != Nil) {
        tl = l->tail;
    
        l->tail = half1;
        half1   = l;
        
        l  = tl;

        stdSWAP(half1,half2,Pointer);
    }

   /*
    * Sort each of the halves; 
    * use temporary variables to not spoil
    * register allocation for half1,2:
    */
    h= half1; listSortD(&h, lessEqD, data); half1= h;
    h= half2; listSortD(&h, lessEqD, data); half2= h;

   /*
    * And merge the two halves into
    * the sorted result:
    */
    while ( half1 && half2 ) {
        if (lessEqD( half1->head, half2->head, data)) {
           *list  = half1;
            half1 = half1->tail;        
        } else {
           *list  = half2;
            half2 = half2->tail;
        }
    
        list = &((*list)->tail);
    }

    if (half1) { *list = half1; }
          else { *list = half2; }
}



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
void STD_CDECL  listRawSort( stdList_t *list, stdLessEqFun rawLessEq)
{
    stdList_t  h,tl;
    stdList_t  l    = *list;
    stdList_t  half1= Nil;
    stdList_t  half2= Nil;
    
   /*
    * Lists of size <=1 are already 
    * sorted, so stop here:
    */
    if ( l == Nil 
      || l->tail == Nil 
       ) { return; }

   /*
    * Split list into two halves:
    */
    while (l != Nil) {
        tl = l->tail;
    
        l->tail = half1;
        half1   = l;
        
        l  = tl;

        stdSWAP(half1,half2,Pointer);
    }

   /*
    * Sort each of the halves; 
    * use temporary variables to not spoil
    * register allocation for half1,2:
    */
    h= half1; listRawSort(&h, rawLessEq); half1= h;
    h= half2; listRawSort(&h, rawLessEq); half2= h;

   /*
    * And merge the two halves into
    * the sorted result:
    */
    while ( half1 && half2 ) {
        if (rawLessEq( half1, half2)) {
           *list  = half1;
            half1 = half1->tail;        
        } else {
           *list  = half2;
            half2 = half2->tail;
        }
    
        list = &((*list)->tail);
    }

    if (half1) { *list = half1; }
          else { *list = half2; }
}

void STD_CDECL  listRawSortD( stdList_t *list, stdLessEqDFun rawLessEqD, Pointer data)
{
    stdList_t  h,tl;
    stdList_t  l    = *list;
    stdList_t  half1= Nil;
    stdList_t  half2= Nil;
    
   /*
    * Lists of size <=1 are already 
    * sorted, so stop here:
    */
    if ( l == Nil 
      || l->tail == Nil 
       ) { return; }

   /*
    * Split list into two halves:
    */
    while (l != Nil) {
        tl = l->tail;
    
        l->tail = half1;
        half1   = l;
        
        l  = tl;

        stdSWAP(half1,half2,Pointer);
    }

   /*
    * Sort each of the halves; 
    * use temporary variables to not spoil
    * register allocation for half1,2:
    */
    h= half1; listRawSortD(&h, rawLessEqD, data); half1= h;
    h= half2; listRawSortD(&h, rawLessEqD, data); half2= h;

   /*
    * And merge the two halves into
    * the sorted result:
    */
    while ( half1 && half2 ) {
        if (rawLessEqD( half1, half2, data)) {
           *list  = half1;
            half1 = half1->tail;        
        } else {
           *list  = half2;
            half2 = half2->tail;
        }
    
        list = &((*list)->tail);
    }

    if (half1) { *list = half1; }
          else { *list = half2; }
}



/*
 * Function        : Copy a list
 * Parameters      : list    (I) list to copy
 * Function Result : copy of list. The head objects are not copied!
 */
stdList_t STD_CDECL listCopy( stdList_t list )
{
    stdList_t result = Nil;

    if (list != Nil) {
        stdList_t l;
        result = listCons(list->head, Nil);
        for (l = result, list = list->tail; list; list = list->tail, l = l->tail) {
            l->tail = listCons(list->head, Nil);
        }
    }      
    return result;
}



/*
 * Function        : Concatenate two lists
 * Parameters      : head    (I) head of the new list
 *                   tail    (I) tail of the new list
 * Function Result : Concatenate two lists. No copies are made, i.e.
 *                   this function may cause tail sharing.
 */
stdList_t STD_CDECL listConcat( stdList_t head, stdList_t tail )
{
    if (!head) {
        return tail;
    } else 
    if (!tail) {
        return head;
    } else {
        stdList_t result = head;

        while (head->tail != Nil) {
            head = head->tail;
        }
        
        head->tail = tail;
        
        return result;
    }
}



/*
 * Function        : Reverse a list
 * Parameters      : list    (I) list to reverse
 * Function Result : Reversed list.
 */
stdList_t STD_CDECL listReverse( stdList_t list )
{
    stdList_t reversed = Nil;

    while (list != Nil) {
        stdList_t tmp;
        
        tmp        = list->tail;
        list->tail = reversed;
        reversed   = list;
        list       = tmp;
    }
    
    return reversed;
}


/*
 * Function        : Create a set form of the list.
 * Parameters      : list    (I) List to colwert.
 *                   hash    (I) Hash function, mapping the set element
 *                               type to an arbitrary integer.
 *                   equal   (I) Equality function for element type.
 * Function Result : Colwerted list.
 */
stdSet_t STD_CDECL listToSet( stdList_t list, stdHashFun hash, stdEqualFun equal)
{
    stdSet_t set = setCreate(hash, equal, listSize(list));

    listTraverse(list, (stdEltFun)setAddTo, set);

    return set;
}


/*
 * Function        : Print list via writer object.
 * Parameters      : wr      (I) Writer to print to
 *                   list    (I) List to print.
 * Function Result : 
 */
    static void STD_CDECL printElt( Pointer e, stdWriter_t wr )
    { wtrPrintf(wr,"%p ",(void*)e); }
    
void STD_CDECL listPrint( stdWriter_t wr, stdList_t list )
{
    wtrPrintf(wr,"< ");
    listTraverse(list,(stdEltFun)printElt,wr);
    wtrPrintf(wr,">\n");
}


/*
 * Function        : Insert the new element before specified element in the 
 *                   list using Pointer equality.
 * Parameters      : element      (I)  Element to add.
 *                   matchElement (I)  Element to match  
 *                   list         (IO) List to affect.
 * Function Result : True iff matchElement is found
 */
Bool STD_CDECL listPutBefore( Pointer element, Pointer matchElement, stdList_t *list )
{
    if (!(*list)) {
        return False;
    } else {  
        stdList_t lwr  = *list;
        stdList_t prev = Nil;

        while (lwr) {
            if (lwr->head == matchElement) {
                if (!prev) {
                    listAddTo(element, list); 
                } else {
                    stdList_t result;
                    stdNEW(result);

                    result->head = element;
                    prev->tail   = result;
                    result->tail = lwr; 
                }
                return True;
            }
            prev = lwr;
            lwr  = lwr->tail;
        }
        return False;
    }
}

/*--------------------------------- Iterator --------------------------------*/

stdListIterator_t listBegin (stdList_t list)
{
  stdListIterator_t it;
  if (list == NULL) {
    return NULL;
  }
  it = list;
  return it;
}

Bool listAtEnd (stdListIterator_t it)
{
  if (it == NULL) return True;
  return False;
}

stdListIterator_t listNext (stdListIterator_t it)
{
  if (it == NULL) return NULL;
  else return it->tail;
}

Pointer listValue (stdListIterator_t it)
{
  if (it == NULL) return NULL;
  return it->head;
}

