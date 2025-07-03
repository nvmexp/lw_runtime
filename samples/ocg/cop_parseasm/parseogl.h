#ifndef _PARSEOGL_H_
#define _PARSEOGL_H_

#include <GL/gl.h>
#include <GL/glext.h>

typedef struct _COPBaseArgs COPBaseArgs;
typedef struct ParamsForCOP_Rec ParamsForCOP;

#define OGLPFLAG_VULKAN                  0x00000001
#define OGLPFLAG_GLINTERNAL              0x00000002


typedef void (*PrintFn)(const char *);

#ifdef __cplusplus
extern "C" {
#endif

void oglParseasm_PrintOGLInfo(FILE *f);

int oglParseasm_Parse(COPBaseArgs *pArgs, unsigned char *source, unsigned int bytes,
                      const char *pcChip, int target, FILE *fOutFile, PrintFn pOutputFn,
                      LwU32 dwFlags);

int oglParseasm_CompileProgram(char *pcChip, char *pcProgram, PrintFn pFn, LwU32 dwFlags);

int oglParseasm_MakeGLprogramFromLWInstructions(lwInst *pspgm, int userProgram);

void oglParseasm_FwriteProgram(FILE * outFile);

void oglParseasm_GetUCode(unsigned char **ppcGPUCode, int *pnGPUCodeSize);

#ifdef __cplusplus
} // extern "C"
#endif

#endif