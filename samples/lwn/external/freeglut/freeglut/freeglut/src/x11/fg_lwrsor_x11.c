/*
 * freeglut_lwrsor_x11.c
 *
 * The Windows-specific mouse cursor related stuff.
 *
 * Copyright (c) 2012 Stephen J. Baker. All Rights Reserved.
 * Written by John F. Fay, <fayjf@sourceforge.net>
 * Creation date: Sun Feb 5, 2012
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * PAWEL W. OLSZTA BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include <GL/freeglut.h>
#include "../fg_internal.h"

/* This code is for Posix/X11, Solaris, and OSX */
#include <X11/lwrsorfont.h>

/*
 * A factory method for an empty cursor
 */
static Cursor getEmptyLwrsor( void )
{
    static Cursor lwrsorNone = None;
    if( lwrsorNone == None ) {
        char lwrsorNoneBits[ 32 ];
        XColor dontCare;
        Pixmap lwrsorNonePixmap;
        memset( lwrsorNoneBits, 0, sizeof( lwrsorNoneBits ) );
        memset( &dontCare, 0, sizeof( dontCare ) );
        lwrsorNonePixmap = XCreateBitmapFromData ( fgDisplay.pDisplay.Display,
                                                   fgDisplay.pDisplay.RootWindow,
                                                   lwrsorNoneBits, 16, 16 );
        if( lwrsorNonePixmap != None ) {
            lwrsorNone = XCreatePixmapLwrsor( fgDisplay.pDisplay.Display,
                                              lwrsorNonePixmap, lwrsorNonePixmap,
                                              &dontCare, &dontCare, 0, 0 );
            XFreePixmap( fgDisplay.pDisplay.Display, lwrsorNonePixmap );
        }
    }
    return lwrsorNone;
}

typedef struct tag_lwrsorCacheEntry lwrsorCacheEntry;
struct tag_lwrsorCacheEntry {
    unsigned int lwrsorShape;    /* an XC_foo value */
    Cursor cachedLwrsor;         /* None if the corresponding cursor has
                                    not been created yet */
};

/*
 * Note: The arrangement of the table below depends on the fact that
 * the "normal" GLUT_LWRSOR_* values start a 0 and are conselwtive.
 */ 
static lwrsorCacheEntry lwrsorCache[] = {
    { XC_arrow,               None }, /* GLUT_LWRSOR_RIGHT_ARROW */
    { XC_top_left_arrow,      None }, /* GLUT_LWRSOR_LEFT_ARROW */
    { XC_hand1,               None }, /* GLUT_LWRSOR_INFO */
    { XC_pirate,              None }, /* GLUT_LWRSOR_DESTROY */
    { XC_question_arrow,      None }, /* GLUT_LWRSOR_HELP */
    { XC_exchange,            None }, /* GLUT_LWRSOR_CYCLE */
    { XC_spraycan,            None }, /* GLUT_LWRSOR_SPRAY */
    { XC_watch,               None }, /* GLUT_LWRSOR_WAIT */
    { XC_xterm,               None }, /* GLUT_LWRSOR_TEXT */
    { XC_crosshair,           None }, /* GLUT_LWRSOR_CROSSHAIR */
    { XC_sb_v_double_arrow,   None }, /* GLUT_LWRSOR_UP_DOWN */
    { XC_sb_h_double_arrow,   None }, /* GLUT_LWRSOR_LEFT_RIGHT */
    { XC_top_side,            None }, /* GLUT_LWRSOR_TOP_SIDE */
    { XC_bottom_side,         None }, /* GLUT_LWRSOR_BOTTOM_SIDE */
    { XC_left_side,           None }, /* GLUT_LWRSOR_LEFT_SIDE */
    { XC_right_side,          None }, /* GLUT_LWRSOR_RIGHT_SIDE */
    { XC_top_left_corner,     None }, /* GLUT_LWRSOR_TOP_LEFT_CORNER */
    { XC_top_right_corner,    None }, /* GLUT_LWRSOR_TOP_RIGHT_CORNER */
    { XC_bottom_right_corner, None }, /* GLUT_LWRSOR_BOTTOM_RIGHT_CORNER */
    { XC_bottom_left_corner,  None }  /* GLUT_LWRSOR_BOTTOM_LEFT_CORNER */
};

void fgPlatformSetLwrsor ( SFG_Window *window, int lwrsorID )
{
    Cursor cursor;
    /*
     * XXX FULL_CROSSHAIR demotes to plain CROSSHAIR. Old GLUT allows
     * for this, but if there is a system that easily supports a full-
     * window (or full-screen) crosshair, we might consider it.
     */
    int lwrsorIDToUse =
        ( lwrsorID == GLUT_LWRSOR_FULL_CROSSHAIR ) ? GLUT_LWRSOR_CROSSHAIR : lwrsorID;

    if( ( lwrsorIDToUse >= 0 ) &&
        ( lwrsorIDToUse < sizeof( lwrsorCache ) / sizeof( lwrsorCache[0] ) ) ) {
        lwrsorCacheEntry *entry = &lwrsorCache[ lwrsorIDToUse ];
        if( entry->cachedLwrsor == None ) {
            entry->cachedLwrsor =
                XCreateFontLwrsor( fgDisplay.pDisplay.Display, entry->lwrsorShape );
        }
        cursor = entry->cachedLwrsor;
    } else {
        switch( lwrsorIDToUse )
        {
        case GLUT_LWRSOR_NONE:
            cursor = getEmptyLwrsor( );
            break;

        case GLUT_LWRSOR_INHERIT:
            cursor = None;
            break;

        default:
            fgError( "Unknown cursor type: %d", lwrsorIDToUse );
            return;
        }
    }

    if ( lwrsorIDToUse == GLUT_LWRSOR_INHERIT ) {
        XUndefineLwrsor( fgDisplay.pDisplay.Display, window->Window.Handle );
    } else if ( cursor != None ) {
        XDefineLwrsor( fgDisplay.pDisplay.Display, window->Window.Handle, cursor );
    } else if ( lwrsorIDToUse != GLUT_LWRSOR_NONE ) {
        fgError( "Failed to create cursor" );
    }
}


void fgPlatformWarpPointer ( int x, int y )
{
    XWarpPointer(
        fgDisplay.pDisplay.Display,
        None,
        fgStructure.LwrrentWindow->Window.Handle,
        0, 0, 0, 0,
        x, y
    );
    /* Make the warp visible immediately. */
    XFlush( fgDisplay.pDisplay.Display );
}

void fghPlatformGetLwrsorPos(const SFG_Window *window, GLboolean client, SFG_XYUse *mouse_pos)
{
    /* Get current pointer location in screen coordinates (if client is false or window is NULL), else
     * Get current pointer location relative to top-left of client area of window (if client is true and window is not NULL)
     */
    Window w = (client && window && window->Window.Handle)? window->Window.Handle: fgDisplay.pDisplay.RootWindow;
    Window junk_window;
    unsigned int junk_mask;
    int clientX, clientY;

    XQueryPointer(fgDisplay.pDisplay.Display, w,
            &junk_window, &junk_window,
            &mouse_pos->X, &mouse_pos->Y, /* Screen coords relative to root window's top-left */
            &clientX, &clientY,           /* Client coords relative to window's top-left */
            &junk_mask);

    if (client && window && window->Window.Handle)
    {
        mouse_pos->X = clientX;
        mouse_pos->Y = clientY;
    }

    mouse_pos->Use = GL_TRUE;
}
