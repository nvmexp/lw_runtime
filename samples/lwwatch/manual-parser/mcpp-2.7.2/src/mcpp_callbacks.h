/* mcpp_callbacks.h: declarations of CALLBACKS data types for MCPP */
#ifndef _MCPP_CALLBACKS_H
#define _MCPP_CALLBACKS_H

typedef struct callbacks {
        void (*macro_defined)(const char *name, short nargs,
                const char *parmnames, const char *repl,
                const char *fname, long mline);
        void (*macro_undefined)(const char *name,
                const char *fname, long mline);
        void (*macro_expanded)(const char *name, const char *expr,
                const char *repl,
                const char *fname, long mline);
        void (*pragma_eval)(const char *expr, int valid,
                unsigned long long val,
                const char *fname, long mline);
} CALLBACKS;

#endif  /* _MCPP_CALLBACKS_H */
