#!/bin/ksh
##############################################################################
# Copyright (c) 2020, LWPU CORPORATION. All rights reserved.
#
# LWPU CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from LWPU CORPORATION is strictly prohibited.
##############################################################################
# http://www.qnx.com/developers/docs/7.0.0/index.html#com.qnx.doc.neutrino.utilities/topic/g/getconf.html
#

# The name of the instruction set architecture for this node's CPU(s).
# write by setconf()
echo _CS_ARCHITECTURE
getconf _CS_ARCHITECTURE
# A colon-separated list of directories to search for configuration files.
# write by setconf()
echo _CS_CONFIG_PATH
getconf _CS_CONFIG_PATH
# The domain name.
# write by setconf()
echo _CS_DOMAIN
getconf _CS_DOMAIN
# The name of this node in the network.
# Note:	A hostname can consist only of letters, numbers, and hyphens, and must
# not start or end with a hyphen. For more information, see RFC 952.
# write by setconf()
echo _CS_HOSTNAME
getconf _CS_HOSTNAME
# The name of the hardware manufacturer.
# write by setconf()
echo _CS_HW_PROVIDER
getconf _CS_HW_PROVIDER
# Serial number associated with the hardware.
# write by setconf()
echo _CS_HW_SERIAL
getconf _CS_HW_SERIAL
# A value similar to the LD_LIBRARY_PATH environment variable that finds all
# standard libraries.
# write by setconf()
echo _CS_LIBPATH
getconf _CS_LIBPATH
# The name of the current locale.
# write by setconf()
echo _CS_LOCALE
getconf _CS_LOCALE
# This node's hardware type.
# write by setconf()
echo _CS_MACHINE
getconf _CS_MACHINE
# A value similar to the PATH environment variable that finds all standard
# utilities.
# write by setconf()
echo _CS_PATH
getconf _CS_PATH
# The current OS release level.
# write by setconf()
echo _CS_RELEASE
getconf _CS_RELEASE
# The contents of the resolv.conf file, excluding the domain name.
# write by setconf()
echo _CS_RESOLVE
getconf _CS_RESOLVE
# The secure RPC domain.
# write by setconf()
echo _CS_SRPC_DOMAIN
getconf _CS_SRPC_DOMAIN
# The name of the operating system.
# write by setconf()
echo _CS_SYSNAME
getconf _CS_SYSNAME
# Time zone string (TZ style)
# write by setconf()
echo _CS_TIMEZONE
getconf _CS_TIMEZONE
# The current OS version number.
# write by setconf()
echo _CS_VERSION
getconf _CS_VERSION
# The maximum amount by which a process can decrease its asynchronous I/O
# priority level from its own scheduling priority.
# read by sysconf()
echo _SC_AIO_PRIO_DELTA_MAX
getconf _SC_AIO_PRIO_DELTA_MAX
# Maximum length of arguments for the exec*() functions, in bytes,
# including environment data.
# read by sysconf()
echo _SC_ARG_MAX
getconf _SC_ARG_MAX
# Maximum number of simultaneous processes per real user ID.
# read by sysconf()
echo _SC_CHILD_MAX
getconf _SC_CHILD_MAX
# The number of intervals per second used to express the value in type clock_t.
# read by sysconf()
echo _SC_CLK_TCK
getconf _SC_CLK_TCK
# The maximum number of times a timer can overrun and you can still detect it.
# read by sysconf()
echo _SC_DELAYTIMER_MAX
getconf _SC_DELAYTIMER_MAX
# If defined (not -1), the maximum size of buffer that you need to supply to
# getgrgid_r() for any memory that it needs to allocate.
# read by sysconf()
echo _SC_GETGR_R_SIZE_MAX
getconf _SC_GETGR_R_SIZE_MAX
# If defined (not -1), the maximum size of buffer that you need to supply to
# getpwent_r(), getpwuid_r(), getspent_r(), or getspnam_r() for any memory
# that they need to allocate.
# read by sysconf()
echo _SC_GETPW_R_SIZE_MAX
getconf _SC_GETPW_R_SIZE_MAX
# If this variable is defined, then job control is supported.
# read by sysconf()
echo _SC_JOB_CONTROL
getconf _SC_JOB_CONTROL
# The maximum number of simultaneous supplementary group IDs per process.
# read by sysconf()
echo _SC_NGROUPS_MAX
getconf _SC_NGROUPS_MAX
# Maximum number of files that one process can have open at any given time.
# read by sysconf()
echo _SC_OPEN_MAX
getconf _SC_OPEN_MAX
# The default size of a thread's guard area.
# read by sysconf()
echo _SC_PAGESIZE
getconf _SC_PAGESIZE
# The minimum amount of memory that the system retains as a resource
# constraint threshold. Only an unconstrained process (i.e., one with
# the PROCMGR_AID_RCONSTRAINT ability enabled—see procmgr_ability()) can
# allocate memory if there's less than this amount left.
# read by sysconf()
echo _SC_RCT_MEM
getconf _SC_RCT_MEM
# The minimum number of connection IDs that a server retains as a resource
# constraint threshold. Only an unconstrained process (i.e., one with the
# PROCMGR_AID_RCONSTRAINT ability enabled—see procmgr_ability()) can create
# a connection if there are fewer than this number left.
# read by sysconf()
echo _SC_RCT_SCOID
getconf _SC_RCT_SCOID
# If this variable is defined, then each process has a saved set-user ID and
# a saved set-group ID.
# read by sysconf()
echo _SC_SAVED_IDS
getconf _SC_SAVED_IDS
# The maximum number of semaphores that one process can have open at a time.
# The getconf utility reports a value of -1 to indicate that this limit is
# indeterminate because it applies to both named and unnamed semaphores.
# The kernel allows an arbitrary number of unnamed semaphores (they're kernel
# synchronization objects, so the number of them is limited only by the amount
# of available kernel memory).
# read by sysconf()
echo _SC_SEM_NSEMS_MAX
getconf _SC_SEM_NSEMS_MAX
# The maximum number of outstanding realtime signals sent per process.
# read by sysconf()
echo _SC_SIGQUEUE_MAX
getconf _SC_SIGQUEUE_MAX
# The minimum stack size for a thread.
# read by sysconf()
echo _SC_THREAD_STACK_MIN
getconf _SC_THREAD_STACK_MIN
# The maximum length of the names for time zones.
# read by sysconf()
echo _SC_TZNAME_MAX
getconf _SC_TZNAME_MAX
# The current POSIX version that is lwrrently supported. A value of 198808L
# indicates the August (08) 1988 standard, as approved by the IEEE Standards
# Board.
# The second form writes to standard output the value of the specified path
# variable for the given path. The possible values of path_var are those for
# pathconf() (see the Library Reference):
# read by sysconf()
echo _SC_VERSION
getconf _SC_VERSION
# Defined if asynchronous I/O is supported for the file.
# read by pathconf()
echo _PC_ASYNC_IO
getconf _PC_ASYNC_IO $1
# (QNX Neutrino 7.0 or later) 1 if the filesystem preserves the case in file
# names, or 0 if it doesn't. If this variable isn't defined, you shouldn't
# make any assumptions about the filesystem's behavior.
# read by pathconf()
echo _PC_CASE_PRESERVING
getconf _PC_CASE_PRESERVING $1
# If defined (not -1), indicates that the use of the chown function is
# restricted to a process with root privileges, and to changing the group ID
# of a file to the effective group ID of the process or to one of its
# supplementary group IDs.
# read by pathconf()
echo _PC_CHOWN_RESTRICTED
getconf _PC_CHOWN_RESTRICTED $1
# Defined (not -1) if the filesystem permits the unlinking of a directory.
# read by pathconf()
echo _PC_LINK_DIR
getconf _PC_LINK_DIR $1
# Maximum value of a file's link count.
# read by pathconf()
echo _PC_LINK_MAX
getconf _PC_LINK_MAX $1
# Maximum number of bytes in a terminal's canonical input buffer (edit buffer).
# read by pathconf()
echo _PC_MAX_CANON
getconf _PC_MAX_CANON $1
# Maximum number of bytes in a terminal's raw input buffer.
# read by pathconf()
echo _PC_MAX_INPUT
getconf _PC_MAX_INPUT $1
# Maximum number of bytes in a file name (not including the terminating null).
# read by pathconf()
echo _PC_NAME_MAX
getconf _PC_NAME_MAX $1
# If defined (not -1), indicates that the use of pathname components longer
# than the value given by _PC_NAME_MAX will generate an error.
# read by pathconf()
echo _PC_NO_TRUNC
getconf _PC_NO_TRUNC $1
# Maximum number of bytes in a pathname (not including the terminating null).
# read by pathconf()
echo _PC_PATH_MAX $1
getconf _PC_PATH_MAX
# Maximum number of bytes that can be written atomically when writing to
# a pipe.
# read by pathconf()
echo _PC_PIPE_BUF
getconf _PC_PIPE_BUF $1
# Defined (not -1) if prioritized I/O is supported for the file.
# read by pathconf()
echo _PC_PRIO_IO
getconf _PC_PRIO_IO $1
# Defined (not -1) if synchronous I/O is supported for the file.
# read by pathconf()
echo _PC_SYNC_IO
getconf _PC_SYNC_IO $1
# If defined (not -1), this is the character value which can be used to
# individually disable special control characters in the termios control
# structure.
# read by pathconf()
echo _PC_VDISABLE
getconf _PC_VDISABLE $1
