#ifndef __NEUTRINO_H_INCLUDED
#define __NEUTRINO_H_INCLUDED


#ifndef __PLATFORM_H_INCLUDED
#include "sys/platform.h"
#endif

#ifndef __TYPES_H_INCLUDED
#include "sys/types.h"
#endif

#ifndef _SIGNAL_H_INCLUDED
#include "signal.h"
#endif

#define _PULSE_CODE_MINAVAIL    0    /* QNX managers will never use this range */
#define _PULSE_CODE_MAXAVAIL    127

struct _pulse {
    _Uint16t                    type;
    _Uint16t                    subtype;
    _Int8t                      code;
    _Uint8t                     zero[3];
    union _sigval               value;
    _Int32t                     scoid;
};

/*
 * Define channel flags
 */
#define _NTO_CHF_FIXED_PRIORITY     0x0001u
#define _NTO_CHF_UNBLOCK            0x0002u
#define _NTO_CHF_THREAD_DEATH       0x0004u
#define _NTO_CHF_DISCONNECT         0x0008u
#define _NTO_CHF_NET_MSG            0x0010u
#define _NTO_CHF_SENDER_LEN         0x0020u
#define _NTO_CHF_COID_DISCONNECT    0x0040u
#define _NTO_CHF_REPLY_LEN          0x0080u
#define _NTO_CHF_STICKY             0x0100u
#define _NTO_CHF_ASYNC_NONBLOCK     0x0200u
#define _NTO_CHF_ASYNC              0x0400u
#define _NTO_CHF_GLOBAL             0x0800u
#define _NTO_CHF_PRIVATE            0x1000u
#define _NTO_CHF_MSG_PAUSING        0x2000u
#define _NTO_CHF_SIG_RESTART        0x4000u
#define _NTO_CHF_UNBLOCK_TIMER      0x8000u

/*
 * Define connect flags
 */
#define _NTO_COF_CLOEXEC        0x0001
#define _NTO_COF_DEAD           0x0002
#define _NTO_COF_NOSHARE        0x0040
#define _NTO_COF_NETCON         0x0080
#define _NTO_COF_NONBLOCK       0x0100
#define _NTO_COF_ASYNC          0x0200
#define _NTO_COF_GLOBAL         0x0400
#define _NTO_COF_NOEVENT        0x0800
#define _NTO_COF_INSELWRE       0x1000
#define _NTO_COF_REG_EVENTS     0x2000

/* If the 2nd from top bit is set then we don't use the fd connection vector. */
#define _NTO_SIDE_CHANNEL            0x40000000

/* scoids are marked with this bit */
#define _NTO_CONNECTION_SCOID        0x00010000


/* If the 2nd from top bit is set in the channel id, then it is a global channel. */
#define _NTO_GLOBAL_CHANNEL          0x40000000

/*
 * Function prototypes for all kernel calls.
 */
extern int ChannelCreate_r(unsigned __flags);
extern int ConnectAttach_r(_Uint32t __nd, pid_t __pid, int __chid, unsigned __index, int __flags);
extern int ConnectDetach(int __coid);
extern int ChannelDestroy(int __chid);
extern int MsgReceivePulse_r(int __chid, void *__pulse, _Sizet __bytes, struct _msg_info *__info);
extern int MsgSendPulse_r(int __coid, int __priority, int __code, int  __value);

#endif /* __NEUTRINO_H_INCLUDED */
