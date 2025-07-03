/*
** This file contains copyrighted code for the LWN 3D API provided by a
** partner company.  The contents of this file should not be used for
** any purpose other than LWN API development and testing.
*/

#include <math.h>
#include <stdio.h>
#include <mtx.h>
#include <mtx/mtx44.h>
#include "mtxAssert.h"
#include "mtx44Assert.h"

/*---------------------------------------------------------------------*

                             STACK SECTION

*---------------------------------------------------------------------*/


/*---------------------------------------------------------------------*

Name:           MTXInitStack

Description:    initializes a matrix stack size and stack ptr from
                a previously allocated stack.


Arguments:      sPtr      ptr to MtxStack structure to be initialized.

                numMtx    number of matrices in the stack.

                note:     the stack (array) memory must have been
                          previously allocated.
                          MtxVec.h provides a macro
                          ( MTXAllocStack( sPtr, numMtx ) )
                          to accomplish this using OSAlloc().



Return:         none.

*---------------------------------------------------------------------*/
void MTXInitStack( MtxStack *sPtr, u32 numMtx )
{


    ASSERTMSG( (sPtr != 0),              MTX_INITSTACK_1     );
    ASSERTMSG( (sPtr->stackBase != 0),   MTX_INITSTACK_2     );
    ASSERTMSG( (numMtx != 0),            MTX_INITSTACK_3     );


    sPtr->numMtx   = numMtx;
    sPtr->stackPtr = NULL;

}

/*---------------------------------------------------------------------*

Name:           MTXPush

Description:    copy a matrix to stack pointer + 1.
                increment stack pointer.


Arguments:      sPtr    ptr to MtxStack structure.

                m       matrix to copy into (stack pointer + 1) location.


Return:         stack pointer.

*---------------------------------------------------------------------*/
MtxPtr MTXPush ( MtxStack *sPtr, MTX_CONST Mtx m )
{


    ASSERTMSG( (sPtr != 0),             MTX_PUSH_1    );
    ASSERTMSG( (sPtr->stackBase != 0),  MTX_PUSH_2    );
    ASSERTMSG( (m != 0),                MTX_PUSH_3    );


    if( sPtr->stackPtr == NULL )
    {
        sPtr->stackPtr = sPtr->stackBase;
        MTXCopy( m, sPtr->stackPtr );
    }

    else
    {
        // check for stack overflow
        if( (u32)((sPtr->stackPtr - sPtr->stackBase) / MTX_PTR_OFFSET) >=
            (sPtr->numMtx - 1) )
        {
            ASSERTMSG( 0,  MTX_PUSH_4  );
        }

    MTXCopy( m, (sPtr->stackPtr + MTX_PTR_OFFSET) );
    sPtr->stackPtr += MTX_PTR_OFFSET;
    }


    return sPtr->stackPtr;
}

/*---------------------------------------------------------------------*

Name:           MTXPushFwd

Description:    concatenate a matrix with the current top of the stack,
                increment the stack ptr and push the resultant matrix to
                the new top of stack.

                this is intended for use in building forward transformations,
                so concatenation is post-order
                ( top of stack x mtx ) = ( top of stack + 1 ).


Arguments:      sPtr    ptr to MtxStack structure.

                m        matrix to concatenate with stack ptr and
                         push to stack ptr + 1.


Return:         stack pointer.

*---------------------------------------------------------------------*/
MtxPtr MTXPushFwd ( MtxStack *sPtr, MTX_CONST Mtx m )
{

    ASSERTMSG( (sPtr != 0),             MTX_PUSHFWD_1  );
    ASSERTMSG( (sPtr->stackBase != 0),  MTX_PUSHFWD_2  );
    ASSERTMSG( (m != 0),                MTX_PUSHFWD_3  );


    if( sPtr->stackPtr == NULL )
    {
        sPtr->stackPtr = sPtr->stackBase;
        MTXCopy( m, sPtr->stackPtr );
    }

    else
    {
        // check for stack overflow
        if( (u32)((sPtr->stackPtr - sPtr->stackBase) / MTX_PTR_OFFSET) >=
            (sPtr->numMtx - 1) )
        {
            ASSERTMSG( 0,  MTX_PUSHFWD_4  );
        }

        MTXConcat( sPtr->stackPtr, m, ( sPtr->stackPtr + MTX_PTR_OFFSET ) );
        sPtr->stackPtr += MTX_PTR_OFFSET;
    }


    return sPtr->stackPtr;
}

/*---------------------------------------------------------------------*

Name:           MTXPushIlw

Description:    take a matrix, compute its ilwerse and concatenate that
                ilwerse with the current top of the stack,
                increment the stack ptr and push the resultant matrix to
                the new top of stack.

                this is intended for use in building ilwerse transformations,
                so concatenation is pre-order
                ( mtx x top of stack ) = ( top of stack + 1 ).


Arguments:      sPtr    ptr to MtxStack structure.

                m       matrix to concatenate with stack ptr and
                        push to stack ptr + 1.

                        m is not modified by this function.


Return:         stack pointer.

*---------------------------------------------------------------------*/
MtxPtr    MTXPushIlw ( MtxStack *sPtr, MTX_CONST Mtx m )
{

    Mtx mIlw;


    ASSERTMSG( (sPtr != 0),             MTX_PUSHILW_1  );
    ASSERTMSG( (sPtr->stackBase != 0),  MTX_PUSHILW_2  );
    ASSERTMSG( (m != 0),                MTX_PUSHILW_3  );


    MTXIlwerse( m, mIlw );


    if( sPtr->stackPtr == NULL )
    {
        sPtr->stackPtr = sPtr->stackBase;
        MTXCopy( mIlw, sPtr->stackPtr );
    }

    else
    {
        // check for stack overflow
        if( (u32)((sPtr->stackPtr - sPtr->stackBase) / MTX_PTR_OFFSET) >=
            (sPtr->numMtx - 1) )
        {
            ASSERTMSG( 0,  MTX_PUSHILW_4  );
        }

        MTXConcat( mIlw, sPtr->stackPtr, ( sPtr->stackPtr + MTX_PTR_OFFSET ) );
        sPtr->stackPtr += MTX_PTR_OFFSET;
    }


    return sPtr->stackPtr;
}

/*---------------------------------------------------------------------*

Name:           MTXPushIlwXpose

Description:    take a matrix, compute its ilwerse-transpose and concatenate it
                with the current top of the stack,
                increment the stack ptr and push the resultant matrix to
                the new top of stack.

                this is intended for use in building an ilwerse-transpose
                matrix for forward transformations of normals, so
                concatenation is post-order.
                ( top of stack x mtx ) = ( top of stack + 1 ).


Arguments:      sPtr   ptr to MtxStack structure.

                m      matrix to concatenate with stack ptr and
                       push to stack ptr + 1.

                       m is not modified by this function.


Return:         stack pointer.

*---------------------------------------------------------------------*/
MtxPtr MTXPushIlwXpose ( MtxStack *sPtr, MTX_CONST Mtx m )
{

    Mtx mIT;


    ASSERTMSG( (sPtr != 0),             MTX_PUSHILWXPOSE_1    );
    ASSERTMSG( (sPtr->stackBase != 0),  MTX_PUSHILWXPOSE_2    );
    ASSERTMSG( (m != 0),                MTX_PUSHILWXPOSE_3    );


    MTXIlwerse(     m, mIT );
    MTXTranspose( mIT, mIT );


    if( sPtr->stackPtr == NULL )
    {
        sPtr->stackPtr = sPtr->stackBase;
        MTXCopy( mIT, sPtr->stackPtr );
    }

    else
    {
        // check for stack overflow
        if( (u32)((sPtr->stackPtr - sPtr->stackBase) / MTX_PTR_OFFSET) >=
            (sPtr->numMtx - 1) )
        {
            ASSERTMSG( 0,  MTX_PUSHILWXPOSE_4  );
        }

        MTXConcat( sPtr->stackPtr, mIT, ( sPtr->stackPtr + MTX_PTR_OFFSET ) );
        sPtr->stackPtr += MTX_PTR_OFFSET;
    }


    return sPtr->stackPtr;
}

/*---------------------------------------------------------------------*

Name:           MTXPop

Description:    decrement the stack pointer


Arguments:      sPtr        pointer to stack structure


Return:         stack pointer.

*---------------------------------------------------------------------*/
MtxPtr MTXPop ( MtxStack *sPtr )
{


    ASSERTMSG( (sPtr != 0),               MTX_POP_1  );
    ASSERTMSG( (sPtr->stackBase != 0),    MTX_POP_2  );


    if( sPtr->stackPtr == NULL )
    {
        return NULL;
    }

    else if( sPtr->stackBase == sPtr->stackPtr )
    {
        sPtr->stackPtr = NULL;
        return NULL;
    }
    else
    {
        sPtr->stackPtr -= MTX_PTR_OFFSET;
        return sPtr->stackPtr;
    }

}

/*---------------------------------------------------------------------*

Name:           MTXGetStackPtr

Description:    return the stack pointer


Arguments:      sPtr pointer to stack structure


Return:         stack pointer.

*---------------------------------------------------------------------*/
MtxPtr MTXGetStackPtr( const MtxStack *sPtr )
{

    ASSERTMSG( (sPtr != 0),               MTX_GETSTACKPTR_1  );
    ASSERTMSG( (sPtr->stackBase != 0),    MTX_GETSTACKPTR_2  );

    return sPtr->stackPtr;

}


/*===========================================================================*/
