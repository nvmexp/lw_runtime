/* Spaceball support for Linux.
 * Written by John Tsiombikas <nuclear@member.fsf.org>
 * Copied for Platform code by Evan Felix <karcaw at gmail.com>
 * Creation date: Thur Feb 2 2012
 *
 * This code supports 3Dconnexion's 6-dof space-whatever devices.
 * It can communicate with either the proprietary 3Dconnexion daemon (3dxsrv)
 * free spacenavd (http://spacenav.sourceforge.net), through the "standard"
 * magellan X-based protocol.
 */

#include <GL/freeglut.h>
#include "../fg_internal.h"

#include <X11/Xlib.h>

extern int sball_initialized;

enum {
    SPNAV_EVENT_ANY,  /* used by spnav_remove_events() */
    SPNAV_EVENT_MOTION,
    SPNAV_EVENT_BUTTON  /* includes both press and release */
};

struct spnav_event_motion {
    int type;
    int x, y, z;
    int rx, ry, rz;
    unsigned int period;
    int *data;
};

struct spnav_event_button {
    int type;
    int press;
    int bnum;
};

typedef union spnav_event {
    int type;
    struct spnav_event_motion motion;
    struct spnav_event_button button;
} spnav_event;


static int spnav_x11_open(Display *dpy, Window win);
static int spnav_x11_window(Window win);
static int spnav_x11_event(const XEvent *xev, spnav_event *event);
static int spnav_close(void);
static int spnav_fd(void);
static int spnav_remove_events(int type);

static SFG_Window *spnav_win;

void fgPlatformInitializeSpaceball(void)
{
    Window w;

    sball_initialized = 1;
    if(!fgStructure.LwrrentWindow)
    {
        sball_initialized = -1;
        return;
    }

    w = fgStructure.LwrrentWindow->Window.Handle;
    if(spnav_x11_open(fgDisplay.pDisplay.Display, w) == -1)
    {
        sball_initialized = -1;
        return;
    }
}

void fgPlatformSpaceballClose(void) 
{
    spnav_close();
}

int fgPlatformHasSpaceball(void) 
{
    /* XXX this function should somehow query the driver if there's a device
     * plugged in, as opposed to just checking if there's a driver to talk to.
     */
    return spnav_fd() == -1 ? 0 : 1;
}

int fgPlatformSpaceballNumButtons(void) {
    return 2;
}

void fgPlatformSpaceballSetWindow(SFG_Window *window) 
{
       if(spnav_win != window) {
        spnav_x11_window(window->Window.Handle);
        spnav_win = window;
    }
}

int fgIsSpaceballXEvent(const XEvent *xev)
{
    spnav_event sev;

    if(spnav_win != fgStructure.LwrrentWindow) {
        /* this will also initialize spaceball if needed (first call) */
        fgSpaceballSetWindow(fgStructure.LwrrentWindow);
    }

    if(sball_initialized != 1) {
        return 0;
    }

    return spnav_x11_event(xev, &sev);
}

void fgSpaceballHandleXEvent(const XEvent *xev)
{
    spnav_event sev;

    if(sball_initialized == 0) {
        fgInitialiseSpaceball();
        if(sball_initialized != 1) {
            return;
        }
    }

    if(spnav_x11_event(xev, &sev)) {
        switch(sev.type) {
        case SPNAV_EVENT_MOTION:
            if(sev.motion.x | sev.motion.y | sev.motion.z) {
                ILWOKE_WCB(*spnav_win, SpaceMotion, (sev.motion.x, sev.motion.y, sev.motion.z));
            }
            if(sev.motion.rx | sev.motion.ry | sev.motion.rz) {
                ILWOKE_WCB(*spnav_win, SpaceRotation, (sev.motion.rx, sev.motion.ry, sev.motion.rz));
            }
            spnav_remove_events(SPNAV_EVENT_MOTION);
            break;

        case SPNAV_EVENT_BUTTON:
            ILWOKE_WCB(*spnav_win, SpaceButton, (sev.button.bnum, sev.button.press ? GLUT_DOWN : GLUT_UP));
            break;

        default:
            break;
        }
    }
}

/*
The following code is part of libspnav, part of the spacenav project (spacenav.sf.net)
Copyright (C) 2007-2009 John Tsiombikas <nuclear@member.fsf.org>

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. The name of the author may not be used to endorse or promote products
   derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROLWREMENT
OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
OF SUCH DAMAGE.
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef HAVE_ERRNO_H
#include <errno.h>
#endif

#include <X11/Xlib.h>
#include <X11/Xutil.h>

static Window get_daemon_window(Display *dpy);
static int catch_badwin(Display *dpy, XErrorEvent *err);

static Display *dpy;
static Window app_win;
static Atom motion_event, button_press_event, button_release_event, command_event;

enum {
  CMD_APP_WINDOW = 27695,
  CMD_APP_SENS
};

#define IS_OPEN    dpy

struct event_node {
  spnav_event event;
  struct event_node *next;
};

static int spnav_x11_open(Display *display, Window win)
{
  if(IS_OPEN) {
    return -1;
  }

  dpy = display;

  motion_event = XInternAtom(dpy, "MotionEvent", True);
  button_press_event = XInternAtom(dpy, "ButtonPressEvent", True);
  button_release_event = XInternAtom(dpy, "ButtonReleaseEvent", True);
  command_event = XInternAtom(dpy, "CommandEvent", True);

  if(!motion_event || !button_press_event || !button_release_event || !command_event) {
    dpy = 0;
    return -1;  /* daemon not started */
  }

  if(spnav_x11_window(win) == -1) {
    dpy = 0;
    return -1;  /* daemon not started */
  }

  app_win = win;
  return 0;
}

static int spnav_close(void)
{
  if(dpy) {
    spnav_x11_window(DefaultRootWindow(dpy));
    app_win = 0;
    dpy = 0;
    return 0;
  }
  return -1;
}

static int spnav_x11_window(Window win)
{
  int (*prev_xerr_handler)(Display*, XErrorEvent*);
  XEvent xev;
  Window daemon_win;

  if(!IS_OPEN) {
    return -1;
  }

  if(!(daemon_win = get_daemon_window(dpy))) {
    return -1;
  }

  prev_xerr_handler = XSetErrorHandler(catch_badwin);

  xev.type = ClientMessage;
  xev.xclient.send_event = False;
  xev.xclient.display = dpy;
  xev.xclient.window = win;
  xev.xclient.message_type = command_event;
  xev.xclient.format = 16;
  xev.xclient.data.s[0] = ((unsigned int)win & 0xffff0000) >> 16;
  xev.xclient.data.s[1] = (unsigned int)win & 0xffff;
  xev.xclient.data.s[2] = CMD_APP_WINDOW;

  XSendEvent(dpy, daemon_win, False, 0, &xev);
  XSync(dpy, False);

  XSetErrorHandler(prev_xerr_handler);
  return 0;
}

static int spnav_fd(void)
{
  if(dpy) {
    return ConnectionNumber(dpy);
  }
  return -1;
}

/*static int spnav_wait_event(spnav_event *event)
{
  if(dpy) {
    for(;;) {
      XEvent xev;
      XNextEvent(dpy, &xev);

      if(spnav_x11_event(&xev, event) > 0) {
        return event->type;
      }
    }
  }
  return 0;
}

static int spnav_poll_event(spnav_event *event)
{
  if(dpy) {
    if(XPending(dpy)) {
      XEvent xev;
      XNextEvent(dpy, &xev);

      return spnav_x11_event(&xev, event);
    }
  }
  return 0;
}*/

static Bool match_events(Display *dpy, XEvent *xev, char *arg)
{
  int evtype = *(int*)arg;

  if(xev->type != ClientMessage) {
    return False;
  }

  if(xev->xclient.message_type == motion_event) {
    return !evtype || evtype == SPNAV_EVENT_MOTION ? True : False;
  }
  if(xev->xclient.message_type == button_press_event ||
      xev->xclient.message_type == button_release_event) {
    return !evtype || evtype == SPNAV_EVENT_BUTTON ? True : False;
  }
  return False;
}

static int spnav_remove_events(int type)
{
  int rm_count = 0;

  if(dpy) {
    XEvent xev;

    while(XCheckIfEvent(dpy, &xev, match_events, (char*)&type)) {
      rm_count++;
    }
    return rm_count;
  }
  return 0;
}

static int spnav_x11_event(const XEvent *xev, spnav_event *event)
{
  int i;
  int xmsg_type;

  if(xev->type != ClientMessage) {
    return 0;
  }

  xmsg_type = xev->xclient.message_type;

  if(xmsg_type != motion_event && xmsg_type != button_press_event &&
      xmsg_type != button_release_event) {
    return 0;
  }

  if(xmsg_type == motion_event) {
    event->type = SPNAV_EVENT_MOTION;
    event->motion.data = &event->motion.x;

    for(i=0; i<6; i++) {
      event->motion.data[i] = xev->xclient.data.s[i + 2];
    }
    event->motion.period = xev->xclient.data.s[8];
  } else {
    event->type = SPNAV_EVENT_BUTTON;
    event->button.press = xmsg_type == button_press_event ? 1 : 0;
    event->button.bnum = xev->xclient.data.s[2];
  }
  return event->type;
}


static Window get_daemon_window(Display *dpy)
{
  Window win, root_win;
  XTextProperty wname;
  Atom type;
  int fmt;
  unsigned long nitems, bytes_after;
  unsigned char *prop;

  root_win = DefaultRootWindow(dpy);

  XGetWindowProperty(dpy, root_win, command_event, 0, 1, False, AnyPropertyType, &type, &fmt, &nitems, &bytes_after, &prop);
  if(!prop) {
    return 0;
  }

  win = *(Window*)prop;
  XFree(prop);

  if(!XGetWMName(dpy, win, &wname) || strcmp("Magellan Window", (char*)wname.value) != 0) {
    return 0;
  }

  return win;
}

static int catch_badwin(Display *dpy, XErrorEvent *err)
{
  char buf[256];

  if(err->error_code == BadWindow) {
    /* do nothing? */
  } else {
    XGetErrorText(dpy, err->error_code, buf, sizeof buf);
    fprintf(stderr, "Caught unexpected X error: %s\n", buf);
  }
  return 0;
}

