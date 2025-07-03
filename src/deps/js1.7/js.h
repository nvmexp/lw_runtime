

#ifndef js_h___
#define js_h___

JS_BEGIN_EXTERN_C
#include "jstypes.h"

JS_PUBLIC_API(JSBool)
JS_StartModsDebugger(JSRuntime *rt, JSContext *cx, JSObject *glob);

JS_PUBLIC_API(void)
JS_StopModsDebugger();

JS_END_EXTERN_C

#endif
