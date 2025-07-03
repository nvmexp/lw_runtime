#ifndef __SIGINFO_H_INCLUDED
#define __SIGINFO_H_INCLUDED


/*
 * for SIGEV_PULSE don't modify the receiving threads priority
 * when the pulse is received
 */
# define SIGEV_PULSE_PRIO_INHERIT (-1)

union _sigval {
    int         sival_int;
    void       *sival_ptr;
};



#endif /* __SIGINFO_H_INCLUDED */
