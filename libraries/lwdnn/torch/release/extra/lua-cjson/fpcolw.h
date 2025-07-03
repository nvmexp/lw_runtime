/* Lua CJSON floating point colwersion routines */

/* Buffer required to store the largest string representation of a double.
 *
 * Longest double printed with %.14g is 21 characters long:
 * -1.7976931348623e+308 */
# define FPCOLW_G_FMT_BUFSIZE   32

#ifdef USE_INTERNAL_FPCOLW
static inline void fpcolw_init()
{
    /* Do nothing - not required */
}
#else
extern void fpcolw_init();
#endif

extern int fpcolw_g_fmt(char*, double, int);
extern double fpcolw_strtod(const char*, char**);

/* vi:ai et sw=4 ts=4:
 */
