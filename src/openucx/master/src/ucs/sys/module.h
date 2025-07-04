/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */


#ifndef UCS_MODULE_H_
#define UCS_MODULE_H_

#include <ucs/type/init_once.h>
#include <ucs/sys/compiler_def.h>


/**
 * Flags for @ref UCS_MODULE_FRAMEWORK_LOAD
 */
typedef enum {
    UCS_MODULE_LOAD_FLAG_NODELETE = UCS_BIT(0), /**< Never unload */
    UCS_MODULE_LOAD_FLAG_GLOBAL   = UCS_BIT(1)  /**< Load to global scope */
} ucs_module_load_flags_t;


/**
 * Declare a "framework", which is a context for a specific collection of
 * loadable modules. Usually the modules in a particular framework provide
 * alternative implementations of the same internal interface.
 *
 * @param [in] _name  Framework name (as a token)
 */
#define UCS_MODULE_FRAMEWORK_DECLARE(_name) \
    static ucs_init_once_t ucs_framework_init_once_##_name = \
        UCS_INIT_ONCE_INITIALIZER


/**
 * Load all modules in a particular framework.
 *
 * @param [in]  _name   Framework name, same as passed to
 *                      @ref UCS_MODULE_FRAMEWORK_DECLARE
 * @param [in]  _flags  Modules load flags, see @ref ucs_module_load_flags_t
 *
 * The modules in the framework are loaded by dlopen(). The shared library name
 * of a module is: "lib<framework>_<module>.so.<version>", where:
 * - <framework> is the framework name
 * - <module> is the module name. The list of all modules in a framework is
 *   defined by the preprocessor macro <framework>_MODULES in the auto-generated
 *   config.h file, for example: #define foo_MODULES ":bar1:bar2".
 * - <version> is the shared library version of the module, as generated by
 *   libtool. It's extracted from the full path of the current library (libucs).
 *
 * Module shared libraries are searched in the following locations (in order of
 * priority):
 *  1. 'ucx' sub-directory inside the directory of the current shared library (libucs)
 *  2. ${libdir}/ucx, where ${libdir} is the directory where libraries are installed
 * Note that if libucs is loaded from its installation path, (1) and (2) are the
 * same location. Only if libucs is moved or ran from build directory, the paths
 * will be different, in which case we prefer the 'local' library rather than the
 * 'installed' one.
 *
 * @param [in] _name  Framework name (as a token)
 */
#define UCS_MODULE_FRAMEWORK_LOAD(_name, _flags) \
    ucs_load_modules(#_name, _name##_MODULES, &ucs_framework_init_once_##_name, \
                     _flags)


/**
 * Define a function to be called when a module is loaded.
 * Some things can't be done in shared library constructor, and need to be done
 * only after dlopen() completes. For example, loading another shared library
 * which uses symbols from the current module.
 *
 * Usage:
 *    UCS_MODULE_INIT() { ... code ... }
 */
#define UCS_MODULE_INIT() \
    ucs_status_t __attribute__((visibility("protected"))) \
    UCS_MODULE_CONSTRUCTOR_NAME(void)


/**
 * Define the name of a loadable module global constructor
 */
#define UCS_MODULE_CONSTRUCTOR_NAME \
    ucs_module_global_init


/**
 * Internal function. Please use @ref UCS_MODULE_FRAMEWORK_LOAD macro instead.
 */
void ucs_load_modules(const char *framework, const char *modules,
                      ucs_init_once_t *init_once, unsigned flags);


#endif
