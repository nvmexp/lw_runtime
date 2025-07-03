/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2015-2020, LWPU CORPORATION.  All rights reserved.
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
 *  Module name              : stdAVL.c
 *
 *  Description              :
 *     
 *         This module defines an AVL tree over abstract elements.
 */

/*--------------------------------- Includes ---------------------------------*/

#include "stdAVL.h"

#ifdef __cplusplus
extern "C" {
#endif

/*----------------------------------- Types ----------------------------------*/

typedef struct AVLNodeRec  *AVLNode;

struct AVLNodeRec {
    AVLNode    left;
    AVLNode    right;

    Int8       delta;
    Pointer    contents;
};


struct stdAVLRec {
    stdHashFun      hash;
    stdLessEqFun    lesseq;
    
    SizeT           size;
    uInt32          hashValue;
    
    AVLNode         root;
};


/*--------------------------------- Functions --------------------------------*/

static void freeNodes( AVLNode node )
{
    if (node) {
        freeNodes(node->left);
        freeNodes(node->right);
        stdFREE(node);
    }
}
 
static void travNodes( AVLNode node, stdEltTraversalRec *rec )
{
    if (node) {
        travNodes(node->left,rec);
        rec->traverse(node->contents,rec->data);
        travNodes(node->right,rec);
    }
}
 
static AVLNode copyNodes( AVLNode node )
{
    if (!node) {
        return Nil;
    } else {
        node = stdCOPY(node);
        
        node->left  = copyNodes(node->left); 
        node->right = copyNodes(node->right);
        
        return node;
    }
}
 
static void addNodes( AVLNode node, stdList_t *result )
{
    if (node) {
        addNodes (node->left,     result); 
        listAddTo(node->contents, result);
        addNodes (node->right,    result);
    }
}
 
static Bool equalNodes( AVLNode n1, AVLNode n2, stdAVL_t avl )
{
    if (!n1 && !n2) {
        return True;
    } else 
    if (!n1) {
        return False;
    } else 
    if (!n2) {
        return False;
    } else {
        Bool le = avl->lesseq( n1->contents, n2->contents);
        Bool ge = avl->lesseq( n2->contents, n1->contents);
        
        return (le == ge)
            && equalNodes(n1->left, n2->left,  avl) 
            && equalNodes(n1->right,n2->right, avl);
    }
}
 
static void STD_CDECL printElement( Pointer elt, stdWriter_t wr) 
{
    wtrPrintf(wr," %p \n", elt); 
}


/*--------------------------------- Functions --------------------------------*/

static void validate( AVLNode node, uInt *depth, SizeT *size )
{
    if (!node) {
        *depth = *size = 0;
    } else {
        uInt lDepth, rDepth;
        SizeT lSize, rSize;
        
        validate(node->left, &lDepth,&lSize);
        validate(node->right,&rDepth,&rSize);
        
        stdASSERT( (lDepth-rDepth) == node->delta, ("Node balance mismatch") );
        
        *depth = stdMAX(lDepth,rDepth) + 1;
        *size  = lSize + rSize + 1;
    }
}

#ifndef AVL_CHECK
    #define lwalidate(x)
#else
    static void lwalidate( AVLNode node )
    {
        uInt depth;
        SizeT size;

        validate(node,&depth,&size);
    }
#endif


/*--------------------------------- Functions --------------------------------*/

    static Bool rebalance( AVLNode *node, Bool insert )
    {
        AVLNode n = *node;
        
        switch (n->delta) {
        case -1 : return  insert;
        case  0 : return !insert;
        case  1 : return  insert;
        
        case -2 :
            {
                AVLNode r      =  n->right;
                AVLNode rl     =  r->left;
                Int     C      =  r->delta;
                Bool    result;

                switch (C) {
                case  0:  
                    /*
                     * This situation can only 
                     * occur after a deletion, 
                     * never after an insertion;
                     * So test this, and then FALLTHROUGH
                     */
                     stdASSERT( !insert, ("Unexpected case") );

                case -1:
                  {
                    *node      =    r;
                     r->left   =    n;
                     n->right  =   rl;
                     r->delta  =  1+C;
                     n->delta  = -1-C;
                     result    = C && !insert;
                     break;
                  } 

                case  1:
                  {
                     AVLNode x = rl->right;
                     AVLNode y = rl->left;

                     C         = rl->delta;

                    *node      = rl;
                     rl->right =  r;
                     rl->left  =  n;
                     n ->right =  y;
                     r ->left  =  x;
                     rl->delta =  0;
                     r->delta  = -(C== 1);
                     n->delta  =  (C==-1);
                     result    = !insert;
                     break;
                  } 
                default : 
                     stdASSERT(False, ("Case label out of range"));
                     result = False;
                }

                lwalidate(*node);
                return result;
            }
        
        case  2 :
            {
                AVLNode l     =  n->left;
                AVLNode lr    =  l->right;
                Int     C     =  l->delta;
                Bool    result;
                 
                switch (C) {
                case  0:  
                    /*
                     * This situation can only 
                     * occur after a deletion, 
                     * never after an insertion;
                     * So test this, and then FALLTHROUGH
                     */
                     stdASSERT( !insert, ("Unexpected case") );
                     
                case  1:
                  {
                    *node      =      l;
                     l->right  =      n;
                     n->left   =     lr;
                     l->delta  = -1 + C;
                     n->delta  =  1 - C;
                     result    = C && !insert;
                     break;
                  } 

                case -1:
                  {
                     AVLNode x = lr->left;
                     AVLNode y = lr->right;

                     C         = lr->delta;

                    *node      = lr;
                     lr->left  =  l;
                     lr->right =  n;
                     n ->left  =  y;
                     l ->right =  x;
                     lr->delta =  0;
                     l->delta  =  (C==-1);
                     n->delta  = -(C== 1);
                     result    = !insert;
                     break;
                  } 
                default : 
                     stdASSERT(False, ("Case label out of range"));
                     result = False;
                }
                 
                lwalidate(*node);
                return result;
            }
             
        default : 
             stdASSERT(False, ("Case label out of range"));
             return True;
        }
    }


/*--------------------------------- Functions --------------------------------*/

    static Bool insertAvl( stdAVL_t avl, AVLNode *node, Pointer *elt )
    {
        AVLNode n = *node;
        
        if (!n) {
            stdNEW(n);
            stdSWAP(n->contents,*elt,Pointer);
            
            avl->size++;
            avl->hashValue ^= avl->hash( n->contents);
            
           *node = n;
            return True;
            
        } else {
            Bool greatereq = avl->lesseq( n->contents, *elt);
            Bool lesseq    = avl->lesseq( *elt, n->contents);

            if (lesseq && greatereq) {
                stdSWAP(n->contents,*elt,Pointer);
                return False;
            } else
            
            if (lesseq) {
               /*
                * Tree rebalancing; see keepsakes/avl:
                */
                if ( !insertAvl(avl,&n->left,elt) ) {
                    return False;
                } else {
                    n->delta++;
                
                    return rebalance(node,True);
                }
                                
            } else {
               /*
                * Tree rebalancing; mirror of the above:
                */
                if ( !insertAvl(avl,&n->right,elt) ) { 
                    return False;
                } else {
                    n->delta--;
                
                    return rebalance(node,True);
                }
            }
        }
    }

/*
 * Function        : Insert element into avl.
 * Parameters      : avl  (I) avl tree to insert into.
 *                   elt  (I) Element to insert.
 * Function Result : The element x previously in the avl such that avl.equal(x,elt),
 *                   or Nil otherwise. Note that if such an x oclwrred, it is 
 *                   replaced by the new element, since the avl treats them as
 *                   equal.
 */
Pointer STD_CDECL  avlInsert( stdAVL_t avl, Pointer elt )
{
    insertAvl(avl,&avl->root,&elt);
    return elt;
}


/*--------------------------------- Functions --------------------------------*/

    static Bool liftRight( AVLNode *pl, AVLNode n )
    {
        AVLNode l= *pl;
        
        if (l->right) {
            if (!liftRight(&l->right, n)) {
                return False;
            } else {
                l->delta++;
                return rebalance(pl,False);
            }
        } else {
            n->contents = l->contents;
           
           (*pl) = l->left;
            stdFREE(l);
            
            return True;
        }
    } 

    static Bool removeAvl( stdAVL_t avl, AVLNode *node, Pointer *elt )
    {
        AVLNode n = *node;
        
        if (!n) {
           *elt = Nil;
            return False;
        } else {
            AVLNode l         = n->left;
            AVLNode r         = n->right;
            Bool    greatereq = avl->lesseq( n->contents, *elt);
            Bool    lesseq    = avl->lesseq( *elt, n->contents);

            if (lesseq && greatereq) {
               /*
                * Element removal; see keepsakes/avl:
                */
                avl->size--;
                avl->hashValue ^= avl->hash( n->contents);
            
               *elt = n->contents;
                
                if (!r) {
                    *node = l;
                     stdFREE(n);
                     return True;
                } else 
                if (!l) {
                    *node = r;
                     stdFREE(n);
                     return True;
                } else {
                     if (!liftRight(&n->left,n)) {
                         return False;
                     } else {
                         n->delta--;
                         return rebalance(node,False);
                     }
                }
                
            } else
            if (lesseq) {
                if ( !removeAvl(avl,&n->left,elt) ) {
                    return False;
                } else {
                    n->delta--;
                    return rebalance(node,False);
                }
            } else {
               /*
                * Tree rebalancing; mirror of the above:
                */
                if ( !removeAvl(avl,&n->right,elt) ) {
                    return False;
                } else {
                    n->delta++;
                    return rebalance(node,False);
                }
            }
        }
    }

/*
 * Function        : Remove element from avl.
 * Parameters      : avl  (I) avl tree to remove from.
 *                   elt  (I) Element to remove.
 * Function Result : The element x previously in the avl such that avl.equal(x,elt),
 *                   or Nil otherwise. If such an x oclwrred, it is 
 *                   removed from the avl.
 */
Pointer STD_CDECL  avlRemove( stdAVL_t avl, Pointer elt )
{
    removeAvl(avl,&avl->root,&elt);
    return elt;
}


/*--------------------------------- Functions --------------------------------*/

/*
 * Function        : Create new avl.
 * Parameters      : hash         (I) Hash function, mapping the avl element
 *                                    type to an arbitrary integer.
 *                   lesseq       (I) Comparison function for element type.
 * Function Result : Requested avl.
 */
stdAVL_t STD_CDECL avlCreate( stdHashFun hash, stdLessEqFun lesseq)
{
    stdAVL_t    result;
    
    stdNEW(result);
    
    result->hash   = hash;
    result->lesseq = lesseq;
    
    return result;
}


/*
 * Function        : Create new (empty) avl with parameters identical to specified avl
 * Parameters      : avl          (I) Template avl.
 * Function Result : Requested avl.
 */
stdAVL_t STD_CDECL avlCreateLike( stdAVL_t avl )
{ return avlCreate( avl->hash, avl->lesseq); }


/*
 * Function        : Discard avl.
 * Parameters      : avl  (I) avl tree to discard.
 * Function Result :
 */
void STD_CDECL  avlDelete( stdAVL_t avl )
{
    freeNodes(avl->root);
    
    stdFREE(avl);
}


/*
 * Function         : Remove all elements from the avl.
 * Parameters       : avl (O) avl tree to empty.
 * Function Result  : True iff. the avl was non-empty when 
 *                    it was passed to this function.
 */
Bool STD_CDECL  avlEmpty( stdAVL_t avl )
{
    Bool result = avl->size == 0;

    freeNodes(avl->root);
    avl->root      = Nil;
    avl->size      = 0;
    avl->hashValue = 0;
    
    return result;
}


/*
 * Function        : Apply specified function to all elements in the specified avl,
 *                   with specified generic data element as additional parameter.
 *                   Traversal will be performed in low to high element order. 
 *                   The avl is not allowed to change during traversal.
 *                   Note: the special exception for the other ADTs, namely
 *                         that the current element may be removed during traversal,
 *                         does NOT hold for avl trees.
 * Parameters      : avl        (I) avl tree to traverse.
 *                   traverse   (I) Function to apply to all elements.
 *                   data       (I) Generic data element passed as additional
 *                                  parameter to every invocation of 'traverse' .
 * Function Result :
 */
void STD_CDECL  avlTraverse( stdAVL_t avl, stdEltFun traverse, Pointer data )
{
    stdEltTraversalRec  rec;
    
    rec.traverse = traverse;
    rec.data     = data;
    
    travNodes(avl->root,&rec);
}


/*
 * Function        : Test oclwrrence in avl.
 * Parameters      : avl  (I) avl tree to test.
 *                   elt  (I) Element to test for oclwrrence.
 * Function Result : The element x in the avl such that avl.equal(x,elt),
 *                   or Nil otherwise.
 */
Pointer STD_CDECL avlElement( stdAVL_t avl, Pointer elt )
{
    AVLNode node = avl->root;
    
    while (node) {
        Bool greatereq = avl->lesseq( node->contents, elt);
        Bool lesseq    = avl->lesseq( elt, node->contents);
        
        if (lesseq && greatereq) { return node->contents; } else
        if (lesseq             ) { node = node->left;     } else
                                 { node = node->right;    }  
    }
    
    return Nil;
}


/*
 * Function        : Test oclwrrence in avl.
 * Parameters      : avl  (I) avl tree to test.
 *                   elt  (I) Element to test for oclwrrence.
 * Function Result : True if and only if elt is a member of avl.
 */
Bool STD_CDECL avlContains( stdAVL_t avl, Pointer elt )
{
    AVLNode node = avl->root;
    
    while (node) {
        Bool greatereq = avl->lesseq( node->contents, elt);
        Bool lesseq    = avl->lesseq( elt, node->contents);
        
        if (lesseq && greatereq) { return True;        } else
        if (lesseq             ) { node = node->left;  } else
                                 { node = node->right; }  
    }
    
    return False;
}


/*
 * Function        : Add specified element to specified avl.
 * Parameters      : element  (I)  Element to add.
 *                   avl     (IO)  stdAVL to modify.
 * Function Result : 
 * NB              : This function is an analogon of avlInsert,
 *                   intended as traversal function.
 */
void STD_CDECL avlAddTo( Pointer element, stdAVL_t avl )
{ avlInsert( avl, element ); }


/*
 * Function        : Delete specified element from specified avl.
 * Parameters      : element  (I)  Element to delete.
 *                   avl     (IO)  stdAVL to modify.
 * Function Result :
 * NB              : This function is an analogon of avlRemove,
 *                   intended as traversal function.
 */
void STD_CDECL avlDeleteFrom( Pointer element, stdAVL_t avl )
{ avlRemove( avl, element ); }


/*
 * Function        : Return number of elements in avl.
 * Parameters      : avl  (I) avl tree to size.
 * Function Result : Number of elements in avl.
 */
SizeT STD_CDECL avlSize( stdAVL_t avl )
{ return avl->size; }


/*
 * Function        : Copy a avl.
 * Parameters      : avl    (I) avl tree to copy.
 * Function Result : Copy of avl. The elt objects are not copied! .
 */
stdAVL_t STD_CDECL avlCopy( stdAVL_t avl )
{
    stdAVL_t result = stdCOPY(avl);
    
    result->root = copyNodes(result->root);
    
    return result;
}


/*
 * Function        : Return an arbitrary element from avl
 *                   (it is not removed).
 * Parameters      : avl  (I) avl tree to return element from.
 * Function Result : An arbitrary element from the avl, or Nil
 *                   if the avl was empty.
 */
Pointer STD_CDECL  avlAnyElement( stdAVL_t avl )
{
    if (avl->root) { return avl->root->contents; }
              else { return Nil;                 }
}


/*
 * Function        : Hash value of avl.
 * Parameters      : avl  (I) avl tree to return hash value from.
 * Function Result : Hash value.
 */
uInt STD_CDECL  avlHash( stdAVL_t avl )
{
    return avl->hashValue;
}


/*
 * Function        : Compare avls for equality.
 * Parameters      : avl1  (I) avl tree1 to compare.
 *                   avl2  (I) avl tree2 to compare.
 * Function Result : True iff the specified avls contain
 *                   an equal amount of elements, that
 *                   are pairwise 'equal' according to the
 *                   equality function by which the avl
 *                   has been created.
 */
Bool STD_CDECL  avlEqual( stdAVL_t avl1, stdAVL_t avl2 )
{
    return (avl1->size      == avl2->size)
        && (avl1->hashValue == avl2->hashValue)
        && equalNodes(avl1->root,avl2->root,avl1);
}


/*
 * Function        : Create a list form of the avl.
 * Parameters      : avl    (I) avl tree to colwert.
 * Function Result : Colwerted avl.
 */
stdList_t STD_CDECL avlToList( stdAVL_t avl )
{
    stdList_t result = Nil;
    
    addNodes(avl->root,&result);
    
    return result;
}


/*
 * Function        : Print avl tree via writer object.
 * Parameters      : wr      (I) Writer to print to
 *                   avl     (I) avl tree to print.
 * Function Result : 
 */
void STD_CDECL avlPrint( stdWriter_t wr, stdAVL_t avl )
{
    avlTraverse(avl,(stdEltFun)printElement,wr);
}


/*
 * Function        : Validate avl tree's internal representation.
 * Parameters      : avl     (I) avl tree to validate.
 * Function Result : 
 */
void STD_CDECL avlValidate( stdAVL_t avl )
{
    uInt depth;
    SizeT size;
    
    validate(avl->root,&depth,&size);
    
    stdASSERT( avl->size == size, ("AVL size does not match") );
}

#ifdef __cplusplus
}
#endif
