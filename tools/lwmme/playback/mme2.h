#ifndef __MME2_H
#define __MME2_H 1

#include <assert.h>
#include <vector>
#include "mme.h"
#include "mme64/mme64.h"

class MME2Program : public MMEProgram {
public:
    MME2Program(unsigned int* instructions, int numInstructions, void (*callback)(void*, unsigned int, unsigned int), bool t=false, bool disableMemoryUnit=false);

    virtual int run(void *callbackData,
        unsigned int *data, int dataSize,
        unsigned int *stateInput, int stateInputSize,
        unsigned int *methodTriggerState, int methodTriggerStateSize,
        unsigned int *dataRamState, int dataRamStateSize,
        unsigned int *memoryState, int memoryStateSize);

    virtual int numInsts() const {
        const int progSize = static_cast<int>(prog.size());
        assert(progSize >= 0
               && static_cast<unsigned int>(progSize) == prog.size());
        return progSize;
    }

private:
    std::vector<MME64Group> prog;

    // True to disable emitted method exelwtion. All emissions will be
    // released. Simulates emissions behaviour of the MME while in the MME to
    // M2M Inline test mode.
    //
    // Ref: https://p4viewer.lwpu.com/getfile/hw/doc/gpu/turing/turing/design/IAS/FE/Hardware%20MicroArch/TU10x_FE_MME64_IAS.docx
    //   Section 13.1 PGRAPH_PRI_MME_CONFIG_MME_TO_M2M_INLINE
    bool disableMemoryUnit;

    uint32_t Merge(uint32_t base, uint32_t src1,
                   uint32_t sourceBit, uint32_t destBit,
                   uint32_t width);
};

#endif
