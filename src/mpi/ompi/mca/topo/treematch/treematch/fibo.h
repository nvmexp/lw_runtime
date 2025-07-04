/* Copyright 2010 IPB, INRIA & CNRS
**
** This file originally comes from the Scotch software package for
** static mapping, graph partitioning and sparse matrix ordering.
**
** This software is governed by the CeCILL-B license under French law
** and abiding by the rules of distribution of free software. You can
** use, modify and/or redistribute the software under the terms of the
** CeCILL-B license as cirlwlated by CEA, CNRS and INRIA at the following
** URL: "http://www.cecill.info".
** 
** As a counterpart to the access to the source code and rights to copy,
** modify and redistribute granted by the license, users are provided
** only with a limited warranty and the software's author, the holder of
** the economic rights, and the successive licensors have only limited
** liability.
** 
** In this respect, the user's attention is drawn to the risks associated
** with loading, using, modifying and/or developing or reproducing the
** software by the user in light of its specific status of free software,
** that may mean that it is complicated to manipulate, and that also
** therefore means that it is reserved for developers and experienced
** professionals having in-depth computer knowledge. Users are therefore
** encouraged to load and test the software's suitability as regards
** their requirements in conditions enabling the security of their
** systems and/or data to be ensured and, more generally, to use and
** operate it in the same conditions as regards security.
** 
** The fact that you are presently reading this means that you have had
** knowledge of the CeCILL-B license and that you accept its terms.
*/
/************************************************************/
/**                                                        **/
/**   NAME       : fibo.h                                  **/
/**                                                        **/
/**   AUTHOR     : Francois PELLEGRINI                     **/
/**                                                        **/
/**   FUNCTION   : This module contains the definitions of **/
/**                the generic Fibonacci trees.            **/
/**                                                        **/
/**   DATES      : # Version 1.0  : from : 01 may 2010     **/
/**                                 to     12 may 2010     **/
/**                                                        **/
/**   NOTES      : # Since this module has originally been **/
/**                  designed as a gain keeping data       **/
/**                  structure for local optimization      **/
/**                  algorithms, the computation of the    **/
/**                  best node is only done when actually  **/
/**                  searching for it.                     **/
/**                  This is most useful when many         **/
/**                  insertions and deletions can take     **/
/**                  place in the mean time. This is why   **/
/**                  this data structure does not keep     **/
/**                  track of the best node, unlike most   **/
/**                  implementations do.                   **/
/**                                                        **/
/************************************************************/

/*
**  The type and structure definitions.
*/

/* The doubly linked list structure. */

typedef struct FiboLink_ {
  struct FiboNode_ *        prevptr;              /*+ Pointer to previous sibling element +*/
  struct FiboNode_ *        nextptr;              /*+ Pointer to next sibling element     +*/
} FiboLink;

/* The tree node data structure. The deflval
   variable merges degree and flag variables.
   The degree of a node is smaller than
   "bitsizeof (INT)", so it can hold on an
   "int". The flag value is stored in the
   lowest bit of the value.                   */
   

typedef struct FiboNode_ {
  struct FiboNode_ *        pareptr;              /*+ Pointer to parent element, if any                +*/
  struct FiboNode_ *        chldptr;              /*+ Pointer to first child element, if any           +*/
  FiboLink                  linkdat;              /*+ Pointers to sibling elements                     +*/
  int                       deflval;              /*+ Lowest bit: flag value; other bits: degree value +*/
} FiboNode;

/* The tree data structure. The fake dummy node aims
   at handling root node insertion without any test.
   This is important as many insertions have to be
   performed.                                        */

typedef struct FiboTree_ {
  FiboNode                  rootdat;              /*+ Dummy node for fast root insertion                      +*/
  FiboNode ** restrict      degrtab;              /*+ Consolidation array of size "bitsizeof (INT)"           +*/
  int                    (* cmpfptr) (const FiboNode * const, const FiboNode * const); /*+ Comparison routine +*/
} FiboTree;

/*
**  The marco definitions.
*/

/* This is the core of the module. All of
   the algorithms have been de-relwrsived
   and written as macros.                 */

#define fiboTreeLinkAfter(o,n)      do {                              \
                                      FiboNode *        nextptr;      \
                                      nextptr = (o)->linkdat.nextptr; \
                                      (n)->linkdat.nextptr = nextptr; \
                                      (n)->linkdat.prevptr = (o);     \
                                      nextptr->linkdat.prevptr = (n); \
                                      (o)->linkdat.nextptr = (n);     \
                                    } while (0)

#define fiboTreeUnlink(n)           do {                                                            \
                                      (n)->linkdat.prevptr->linkdat.nextptr = (n)->linkdat.nextptr; \
                                      (n)->linkdat.nextptr->linkdat.prevptr = (n)->linkdat.prevptr; \
                                    } while (0)

#define fiboTreeAddMacro(t,n)       do {                                        \
                                      (n)->pareptr = NULL;                      \
                                      (n)->chldptr = NULL;                      \
                                      (n)->deflval = 0;                         \
                                      fiboTreeLinkAfter (&((t)->rootdat), (n)); \
  } while (0)

#define fiboTreeMinMacro(t)         (fiboTreeConsolidate (t))

#define fiboTreeLwtChildren(t,n)    do {                                                \
                                      FiboNode *        chldptr;                        \
                                      chldptr = (n)->chldptr;                           \
                                      if (chldptr != NULL) {                            \
                                        FiboNode *        cendptr;                      \
                                        cendptr = chldptr;                              \
                                        do {                                            \
                                          FiboNode *        nextptr;                    \
                                          nextptr = chldptr->linkdat.nextptr;           \
                                          chldptr->pareptr = NULL;                      \
                                          fiboTreeLinkAfter (&((t)->rootdat), chldptr); \
                                          chldptr = nextptr;                            \
                                        } while (chldptr != cendptr);                   \
                                      }                                                 \
                                    } while (0)

#define fiboTreeDelMacro(t,n)       do {                                                    \
                                      FiboNode *        pareptr;                            \
                                      FiboNode *        rghtptr;                            \
                                      pareptr = (n)->pareptr;                               \
                                      fiboTreeUnlink (n);                                   \
                                      fiboTreeLwtChildren ((t), (n));                       \
                                      if (pareptr == NULL)                                  \
                                        break;                                              \
                                      rghtptr = (n)->linkdat.nextptr;                       \
                                      while (1) {                                           \
                                        FiboNode *        gdpaptr;                          \
                                        int               deflval;                          \
                                        deflval = pareptr->deflval - 2;                     \
                                        pareptr->deflval = deflval | 1;                     \
                                        gdpaptr = pareptr->pareptr;                         \
                                        pareptr->chldptr = (deflval <= 1) ? NULL : rghtptr; \
                                        if (((deflval & 1) == 0) || (gdpaptr == NULL))      \
                                          break;                                            \
                                        rghtptr = pareptr->linkdat.nextptr;                 \
                                        fiboTreeUnlink (pareptr);                           \
                                        pareptr->pareptr = NULL;                            \
                                        fiboTreeLinkAfter (&((t)->rootdat), pareptr);       \
                                        pareptr = gdpaptr;                                  \
                                      }                                                     \
                                    } while (0)

/*
**  The function prototypes.
*/

/* This set of definitions allows the user
   to specify whether he prefers to use
   the fibonacci routines as macros or as
   regular functions, for instance for
   debugging.                             */

#define fiboTreeAdd                 fiboTreeAddMacro
/* #define fiboTreeDel              fiboTreeDelMacro */
/* #define fiboTreeMin              fiboTreeMinMacro */

#ifndef FIBO
#define static
#endif

int                         fiboTreeInit        (FiboTree * const, int (*) (const FiboNode * const, const FiboNode * const));
void                        fiboTreeExit        (FiboTree * const);
void                        fiboTreeFree        (FiboTree * const);
FiboNode *                  fiboTreeConsolidate (FiboTree * const);
#ifndef fiboTreeAdd
void                        fiboTreeAdd         (FiboTree * const, FiboNode * const);
#endif /* fiboTreeAdd */
#ifndef fiboTreeDel
void                        fiboTreeDel         (FiboTree * const, FiboNode * const);
#endif /* fiboTreeDel */
#ifndef fiboTreeMin
FiboNode *                  fiboTreeMin         (FiboTree * const);
#endif /* fiboTreeMin */
#ifdef FIBO_DEBUG
int                         fiboTreeCheck       (const FiboTree * const);
static int                  fiboTreeCheck2      (const FiboNode * const);
#endif /* FIBO_DEBUG */

#undef static
