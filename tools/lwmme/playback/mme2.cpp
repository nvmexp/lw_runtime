#define _CRT_SELWRE_NO_DEPRECATE 1
#include <stdio.h>
#include <stdarg.h>
#include <stdint.h>
#include "mme.h"
#include "mme2.h"

#include <map>
#include <vector>
#include <deque>

#ifndef BUILD_VSIM
#error Makefile should define BUILD_VSIM
#endif

#if BUILD_VSIM
#include "mme64vref/sim_test.h"
#endif

using namespace std;
using namespace mme;

uint32_t MME2Program::Merge(uint32_t base, uint32_t src1,
                    uint32_t sourceBit, uint32_t destBit,
                    uint32_t width) {
    uint32_t temp = src1;
    uint32_t oldBase = base;
    uint32_t mask = (1 << width) - 1;

    temp = (sourceBit>31)?0:(temp >> sourceBit);
    temp &= mask;
    temp = (destBit>31)?0:(temp << destBit);

    unsigned xmask = (destBit>31)?0: (mask << destBit) ;

    base &= ~(xmask);
    base |= temp;

    tprintf("0x%08x = Insert 0x%08x=0x%08x[%d:%d] into 0x%08x at [%d:%d]",
        base,
        (destBit > 31) ? 0 : temp >> destBit,
        src1, sourceBit+width-1, sourceBit,
        oldBase, destBit+width-1, destBit
        );

    return base;
}

MME2Program::MME2Program(unsigned int* instructions, int numInstructions, void (*callback)(void*, unsigned int, unsigned int), bool t, bool disableMemoryUnit)  :
    MMEProgram(callback, t), disableMemoryUnit(disableMemoryUnit) {
    // XXX Scalability
    if (numInstructions % 3 != 0) {
        setError("MME2 instructions are in 96-bit chunks");
        return;
    }

    // XXX Scalability
    for (int i=0; i<numInstructions; i += 3) {
        // Pre-decode the instruction into the structure
        prog.push_back(MME64Group(&instructions[i]));
    }
}

uint32_t max(uint32_t a, uint32_t b) {
    return a>b ? a : b;
}

const int END_PC = -1;
const int SKIP_PC = -2;
const int ILWALID_PC = -3;

class MME2RunData {
public:
    MME2RunData(MME2Program &program, uint32_t *data, int dataSize,
        uint32_t *dataRamInput, int dataRamInputSize,
        uint32_t *memoryInput, int memoryInputSize)
        : ramSize(MAX_RAM_SIZE-1024), prog(program), cycles(0), methodData(data, data+dataSize), loadData(&methodData),
        methodSkidInUse(false), previousCycleWasTaken(false) {
        // Init the regfile to garbage
        for(int i=0; i<NUM_REGS; i++) regfile[i] = 0xdeadbeef;
        for(int i=0; i<NUM_REGS; i++) cycleReady[i] = 0;

        for(int i=0; i<MAX_RAM_SIZE; i++) dataRam[i] = 0;
        for(int i=0; i<dataRamInputSize; i += 2) {
            dataRamWritten.set(dataRamInput[i]);
            if (dataRamInput[i] > MAX_RAM_SIZE) {
                char msg[64];
                sprintf(msg, "Data ram 0x%08x out of range\n", dataRamInput[i]);
                program.setError(msg);
            }
            dataRam[dataRamInput[i]] = dataRamInput[i+1];
        }
        for (int i = 0; i < memoryInputSize; i += 2) {
            // XXX Pass through more address bits from the parser?
            memory.insert(make_pair((uint64_t)memoryInput[i], memoryInput[i + 1]));
        }

        specialRegfile[MME64Special::LoadSource] = 0;
        specialRegfile[MME64Special::MmeConfig] = 0;
        specialRegfile[MME64Special::PTimerHigh] = 0;
        specialRegfile[MME64Special::PTimerLow] = 0;
        specialRegfile[MME64Special::Scratch] = 0;

        method = methodInc = 0;

        loopCount = 0;
        loopStart = ILWALID_PC;
        loopEnd   = ILWALID_PC;
    }

    void setError(const char * errorString) { prog.setError(errorString); }
    void setWarning(const char *cat, const char * msg) { prog.setWarning(cat, msg); }

    static const int NUM_REGS = 24;
    static const int MAX_RAM_SIZE = 4096;
    static const int RAM_BANKS = 2;
    // Current Data ram size adjusting for FIFO reservation
    uint32_t ramSize;

    MME2Program &prog;

    int cycles;
    uint32_t regfile[NUM_REGS];
    bitset<NUM_REGS> regWritten;
    int cycleReady[NUM_REGS];
    int method, methodInc;
    deque<uint32_t> methodData;
    deque<uint32_t> fifoData;
    deque<uint32_t> fifoInFlightData;
    deque<uint32_t> *loadData;
    bool methodSkidInUse;
    bool previousCycleWasTaken;
    uint32_t specialRegfile[MME64Special::count];

    uint32_t dataRam[MAX_RAM_SIZE];
    bitset<MAX_RAM_SIZE> dataRamWritten;

    uint32_t loopCount, loopStart, loopEnd;

    map<uint64_t, uint32_t> memory;
};

class MME2CycleData {
public:
    MME2CycleData(MME2RunData &r, MME64Group &g) : group(g), run(r) {
        carry = 0;
        loadsUsed = 0;

        for (int i=0; i<MME2RunData::RAM_BANKS; i++) bankUsed[i] = false;
    }

    void setError(const char * errorString) { run.setError(errorString); }
    void setWarning(const char *cat, const char * msg) { run.setWarning(cat, msg); }

    MME64Group &group;
    MME2RunData &run;

    uint32_t ReadReg(MME64Reg::RegEnum reg, bool aluSource, int alu, int regReadLatency = 0);
    void WriteReg(MME64Reg::RegEnum reg, uint32_t val, uint32_t writeLatency);
    uint32_t ReadOut(MME64Out::OutEnum out, MME64Out::OutEnum pairedEmit, bool method);

    uint32_t ReadDataRam(uint32_t addr);
    void WriteDataRam(uint32_t addr, uint32_t data);

    uint32_t carry;

    uint32_t loadsUsed;

    uint32_t aluOut[MME64NumAlu];

    bool bankUsed[MME2RunData::RAM_BANKS];
};

uint32_t SignExtend(uint32_t val, uint32_t bits) {
    assert((val & ~((1<<bits)-1)) == 0);
    return uint32_t((int32_t(val) << (32-bits)) >> (32-bits));
}

// XXX Scalability
uint32_t MME2CycleData::ReadReg(MME64Reg::RegEnum reg, bool aluSource, int alu, int regReadLatency) {
    // XXX Error-check using aluSource
    switch (reg) {
    case MME64Reg::ZERO:
        return 0;
    case MME64Reg::IMMED:
        return SignExtend(group.alu[alu].immed, 16);
    case MME64Reg::IMMEDPAIR:
        return SignExtend(group.alu[alu^1].immed, 16);
    case MME64Reg::IMMED32:
        return group.alu[alu&~1].immed << 16 | group.alu[(alu&~1)|1].immed;
    case MME64Reg::LOAD0:
    case MME64Reg::LOAD1:
        {
            uint32_t ind = reg-MME64Reg::LOAD0;

            loadsUsed = max(loadsUsed, ind+1);

            if (ind >= run.loadData->size()) {
                setError("Too many loads for data passed in\n");
                return 0;
            }

            return (*run.loadData)[ind];
        }
    default:
        if (reg >= MME64Reg::R0 && reg <= MME64Reg::R23) {
            while (run.cycleReady[reg-MME64Reg::R0] + regReadLatency > run.cycles) {
                run.cycles++;
            }

            return run.regfile[reg-MME64Reg::R0];
        }
        setError("Unknown reg");
        return 0xdeadbeef;
    }
}

void MME2CycleData::WriteReg(MME64Reg::RegEnum reg, uint32_t val, uint32_t writeLatency) {
    switch (reg) {
    case MME64Reg::ZERO:
        return;
    default:
        if (reg >= MME64Reg::R0 && reg <= MME64Reg::R23) {
            run.regfile[reg-MME64Reg::R0] = val;
            run.cycleReady[reg-MME64Reg::R0] = run.cycles + writeLatency;
        } else {
            setError("Invalid destination reg");
        }
    }
}

// XXX Scalability
uint32_t MME2CycleData::ReadOut(MME64Out::OutEnum out, MME64Out::OutEnum pairedEmit, bool method) {
    assert(pairedEmit == MME64Out::NONE || method);

    // XXX Error check using method
    switch (out) {
    case MME64Out::NONE:
        assert(0); // Should be handled in the caller
        return 0;
    case MME64Out::ALU0:
    case MME64Out::ALU1:
        return aluOut[out-MME64Out::ALU0];
    case MME64Out::LOAD0:
    case MME64Out::LOAD1:
        {
            uint32_t ind = out-MME64Out::LOAD0;

            loadsUsed = max(loadsUsed, ind+1);

            if (ind >= run.loadData->size()) {
                setError("Too many loads for data passed in\n");
                return 0;
            }

            return (*run.loadData)[ind];
        }
    case MME64Out::IMMED0:
    case MME64Out::IMMED1:
        return group.alu[out-MME64Out::IMMED0].immed;
    case MME64Out::IMMEDHIGH0:
    case MME64Out::IMMEDHIGH1:
        return (group.alu[out-MME64Out::IMMEDHIGH0].immed >> 12) & 0xf;
    case MME64Out::IMMED32_0:
        return group.alu[0].immed << 16 | group.alu[1].immed;
    default:
        setError("Invalid method/emit enum");
        break;
    }

    assert(0);
    return 0;
}

uint32_t MME2CycleData::ReadDataRam(uint32_t addr) {
    if (addr > run.ramSize) {
        setWarning("dataRam", "Out-of-bounds read from data RAM");
        return 0;
    }
    if (!run.dataRamWritten[addr]) {
        setWarning("dataRam", "Read uninitialized data from data RAM");
        return 0;
    }

    int bank = addr % run.RAM_BANKS;
    if (bankUsed[bank]) {
        setError("Two uses of same bank in one cycle");
        return 0;
    }
    bankUsed[bank] = true;

    return run.dataRam[addr];
}

void MME2CycleData::WriteDataRam(uint32_t addr, uint32_t data) {
    if (addr > run.ramSize) {
        setWarning("dataRam", "Out-of-bounds write to data RAM");
        return;
    }

    int bank = addr % run.RAM_BANKS;
    if (bankUsed[bank]) {
        setError("Two uses of same bank in one cycle");
        return;
    }
    bankUsed[bank] = true;

    run.dataRamWritten.set(addr);

    run.dataRam[addr] = data;
}

// Simulate the memory port as infinitely fast for now
class MME2MemoryUnit {
public:
    MME2MemoryUnit(MME2RunData &runData) : m_runData(runData),
        m_memAddressHi(0), m_memAddressLo(0), m_dataRamAddress(0) { ; }

    bool doMethod(uint32_t method, uint32_t data);
    uint32_t memRead(uint64_t addr);
    void memWrite(uint64_t addr, uint32_t val);
private:
    MME2RunData &m_runData;

    uint32_t m_memAddressHi;
    uint32_t m_memAddressLo;
    uint32_t m_dataRamAddress;
};

uint32_t MME2MemoryUnit::memRead(uint64_t addr) {
    if (m_runData.memory.find(addr) == m_runData.memory.end()) {
        char msg[64];
        sprintf(msg, "Read from undefined memory location 0x%08x`%08x", uint32_t(addr >> 32), uint32_t(addr));
        m_runData.setError(msg);
    }


    return m_runData.memory.find(addr)->second;
}

void MME2MemoryUnit::memWrite(uint64_t addr, uint32_t val) {
    m_runData.memory[addr] = val;
}

// XXX This is the only place where the exelwtion unit needs to "understand" the class
#define LWC597_SET_MME_MEM_ADDRESS_A                                                                       0x0550
#define LWC597_SET_MME_MEM_ADDRESS_A_UPPER                                                                    7:0

#define LWC597_SET_MME_MEM_ADDRESS_B                                                                       0x0554
#define LWC597_SET_MME_MEM_ADDRESS_B_LOWER                                                                   31:0

#define LWC597_SET_MME_DATA_RAM_ADDRESS                                                                    0x0558
#define LWC597_SET_MME_DATA_RAM_ADDRESS_WORD                                                                 31:0

#define LWC597_MME_DMA_READ                                                                                0x055c
#define LWC597_MME_DMA_READ_LENGTH                                                                           31:0

#define LWC597_MME_DMA_READ_FIFOED                                                                         0x0560
#define LWC597_MME_DMA_READ_FIFOED_LENGTH                                                                    31:0

#define LWC597_MME_DMA_WRITE                                                                               0x0564
#define LWC597_MME_DMA_WRITE_LENGTH                                                                          31:0

#define LWC597_MME_DMA_REDUCTION                                                                           0x0568
#define LWC597_MME_DMA_REDUCTION_REDUCTION_OP                                                                 2:0
#define LWC597_MME_DMA_REDUCTION_REDUCTION_OP_RED_ADD                                                  0x00000000
#define LWC597_MME_DMA_REDUCTION_REDUCTION_OP_RED_MIN                                                  0x00000001
#define LWC597_MME_DMA_REDUCTION_REDUCTION_OP_RED_MAX                                                  0x00000002
#define LWC597_MME_DMA_REDUCTION_REDUCTION_OP_RED_INC                                                  0x00000003
#define LWC597_MME_DMA_REDUCTION_REDUCTION_OP_RED_DEC                                                  0x00000004
#define LWC597_MME_DMA_REDUCTION_REDUCTION_OP_RED_AND                                                  0x00000005
#define LWC597_MME_DMA_REDUCTION_REDUCTION_OP_RED_OR                                                   0x00000006
#define LWC597_MME_DMA_REDUCTION_REDUCTION_OP_RED_XOR                                                  0x00000007
#define LWC597_MME_DMA_REDUCTION_REDUCTION_FORMAT                                                             5:4
#define LWC597_MME_DMA_REDUCTION_REDUCTION_FORMAT_UNSIGNED_32                                          0x00000000
#define LWC597_MME_DMA_REDUCTION_REDUCTION_FORMAT_SIGNED_32                                            0x00000001
#define LWC597_MME_DMA_REDUCTION_REDUCTION_SIZE                                                               8:8
#define LWC597_MME_DMA_REDUCTION_REDUCTION_SIZE_FOUR_BYTES                                             0x00000000
#define LWC597_MME_DMA_REDUCTION_REDUCTION_SIZE_EIGHT_BYTES                                            0x00000001

#define LWC597_MME_DMA_SYSMEMBAR                                                                           0x056c
#define LWC597_MME_DMA_SYSMEMBAR_V                                                                            0:0

#define LWC597_MME_DMA_SYNC                                                                                0x0570
#define LWC597_MME_DMA_SYNC_VALUE                                                                            31:0

#define LWC597_SET_MME_DATA_FIFO_CONFIG                                                                    0x0574
#define LWC597_SET_MME_DATA_FIFO_CONFIG_FIFO_SIZE                                                             2:0
#define LWC597_SET_MME_DATA_FIFO_CONFIG_FIFO_SIZE_SIZE_0KB                                             0x00000000
#define LWC597_SET_MME_DATA_FIFO_CONFIG_FIFO_SIZE_SIZE_4KB                                             0x00000001
#define LWC597_SET_MME_DATA_FIFO_CONFIG_FIFO_SIZE_SIZE_8KB                                             0x00000002
#define LWC597_SET_MME_DATA_FIFO_CONFIG_FIFO_SIZE_SIZE_12KB                                            0x00000003
#define LWC597_SET_MME_DATA_FIFO_CONFIG_FIFO_SIZE_SIZE_16KB                                            0x00000004

bool MME2MemoryUnit::doMethod(uint32_t method, uint32_t data)
{
    // The MME deals in unshifted space, but the numbers are in register space
    switch (method << 2) {
    default:
        return false;
    case LWC597_SET_MME_MEM_ADDRESS_A:
        m_memAddressHi = data;
        break;
    case LWC597_SET_MME_MEM_ADDRESS_B:
        m_memAddressLo = data;
        break;
    case LWC597_SET_MME_DATA_RAM_ADDRESS:
        m_dataRamAddress = data;
        break;
    case LWC597_MME_DMA_READ:
        {
            // Hardware forces alignment by dropping the low 2 bits
            uint64_t address = uint64_t(m_memAddressLo & ~0x3) | (uint64_t(m_memAddressHi) << 32);
            m_runData.prog.tprintf("DMA_READ of %d words from 0x%08x`%08x\n", data, uint32_t(address >> 32), uint32_t(address));
            for (uint32_t i = 0; i < data; i++) {
                uint32_t val = memRead(address + 4 * i);
                m_runData.dataRamWritten.set(m_dataRamAddress + i);
                m_runData.dataRam[m_dataRamAddress + i] = val;
            }
        }
        break;
    case LWC597_MME_DMA_READ_FIFOED:
        {
            if (m_runData.fifoInFlightData.size() >= 8) {
                m_runData.setError("Too many fifoed reads in flight - can only have 8 with unconsumed data");
            }
            // Hardware forces alignment by dropping the low 2 bits
            uint64_t address = uint64_t(m_memAddressLo & ~0x3) | (uint64_t(m_memAddressHi) << 32);
            m_runData.prog.tprintf("DMA_READ_FIFOED of %d words from 0x%08x`%08x\n", data, uint32_t(address >> 32), uint32_t(address));
            for (uint32_t i = 0; i < data; i++) {
                uint32_t val = memRead(address + 4 * i);
                m_runData.fifoData.push_back(val);
            }
            m_runData.fifoInFlightData.push_back(data);
        }
        break;
    case LWC597_MME_DMA_WRITE:
        {
            // Hardware forces alignment by dropping the low 2 bits
            uint64_t address = uint64_t(m_memAddressLo & ~0x3) | (uint64_t(m_memAddressHi) << 32);
            m_runData.prog.tprintf("DMA_WRITE of %d words from 0x%08x`%08x\n", data, uint32_t(address >> 32), uint32_t(address));
            for (uint32_t i = 0; i < data; i++) {
                if (!m_runData.dataRamWritten[m_dataRamAddress + i]) {
                    m_runData.setError("DmaWrite reading unwritten data RAM address");
                }
                memWrite(address + 4*i, m_runData.dataRam[m_dataRamAddress + i]);
            }
        }
        break;
    case LWC597_MME_DMA_REDUCTION:
        m_runData.setError("Reduction not yet implemented");
        break;
    case LWC597_MME_DMA_SYSMEMBAR:
        // NOP in the simulator - no other clients
        break;
    case LWC597_MME_DMA_SYNC:
        // Simulating infinitely fast, so can always sync immediately
        m_runData.dataRamWritten.set(m_dataRamAddress);
        m_runData.dataRam[m_dataRamAddress] = data;
        break;
    case LWC597_SET_MME_DATA_FIFO_CONFIG:
        switch (data) {
        case LWC597_SET_MME_DATA_FIFO_CONFIG_FIFO_SIZE_SIZE_0KB:
            m_runData.ramSize = m_runData.MAX_RAM_SIZE - 0 * 1024 / 4;
            break;
        case LWC597_SET_MME_DATA_FIFO_CONFIG_FIFO_SIZE_SIZE_4KB:
            m_runData.ramSize = m_runData.MAX_RAM_SIZE - 4 * 1024 / 4;
            break;
        case LWC597_SET_MME_DATA_FIFO_CONFIG_FIFO_SIZE_SIZE_8KB:
            m_runData.ramSize = m_runData.MAX_RAM_SIZE - 8 * 1024 / 4;
            break;
        case LWC597_SET_MME_DATA_FIFO_CONFIG_FIFO_SIZE_SIZE_12KB:
            m_runData.ramSize = m_runData.MAX_RAM_SIZE - 12 * 1024 / 4;
            break;
        case LWC597_SET_MME_DATA_FIFO_CONFIG_FIFO_SIZE_SIZE_16KB:
            m_runData.ramSize = m_runData.MAX_RAM_SIZE - 16 * 1024 / 4;
            break;
        default:
            m_runData.setError("Invalid FIFO config");
            break;
        }
        break;
    }

    return true;
}

const int CYCLE_LIMIT = 20*1000000;

static vector<uint32_t> PackUcode(vector<MME64Group> &hwProg) {
    vector<uint32_t> rv;

    for (int i = 0; i<(int)hwProg.size(); i++) {
        hwProg[i].validate();
        hwProg[i].legalize();

        MME64HW hwInst = hwProg[i].pack();

        // ZZZ Scalability
        rv.push_back(hwInst.extract<MME64GroupBits, 64>().toUint());
        rv.push_back(hwInst.extract<63, 32>().toUint());
        rv.push_back(hwInst.extract<31, 0>().toUint());
    }

    return rv;
}

int MME2Program::run(void *callbackData,
    unsigned int *data, int dataSize,
    unsigned int *stateInput, int stateInputSize,
    unsigned int *methodTriggerState, int methodTriggerStateSize,
    unsigned int *dataRamState, int dataRamStateSize,
    unsigned int *memoryState, int memoryStateSize) {
    bool endingLoop = false;

    vector<uint32_t> ucode = PackUcode(prog);

#if BUILD_VSIM
    vector<sim_cycle_event> sim_events;

    int simClocks = -1;

    if (memoryStateSize == 0) {
        simClocks = simulate(ucode.size(), &ucode[0],
            dataSize, data, stateInputSize, stateInput, dataRamStateSize, dataRamState, &sim_events);
    }

    vector<sim_cycle_event>::const_iterator lwrSimEvent = sim_events.cbegin();
#endif

    const int PcDepth = 5;
    int pc[PcDepth];
    for (int i = 0; i < PcDepth; i++) {
        pc[i] = i;
    }

    bool lwrDelay = false;
    bool nextDelay = false;

    // Initialize the common state in MMEProgram
    startRun(stateInput, stateInputSize, methodTriggerState, methodTriggerStateSize);

    MME2RunData run(*this, data, dataSize, dataRamState, dataRamStateSize, memoryState, memoryStateSize);

    MME2MemoryUnit memory(run);

    if (isError) return -1;

    while (pc[0] != END_PC) {
        tprintf("PC: %d, clock: %d", pc[0], run.cycles);
        if (pc[0] != SKIP_PC) {
            if(pc[0] < 0 || static_cast<unsigned int>(pc[0]) >= prog.size()) {
                setError("PC outside legal range");
                return -1;
            }

            MME64Group &group = prog[pc[0]];
            tprintf("  %s", group.disassemble(false).c_str());
            MME2CycleData cycleData(run, group);
            bool taken = false;
            bool pcModified = false;
            int writeLatency[MME64NumAlu] = { 1, 1 };
            int stateAlu = -1;
            uint32_t stateAddress = 0;

            int specialWriteAddr[MME64NumAlu];
            uint32_t specialWriteData[MME64NumAlu];

            // Skip exelwtion if the predicate indicates
            bool pred = cycleData.ReadReg(group.global.pred, false, 0) == 0;
            bool aluPred[MME64NumAlu];

            lwrDelay = nextDelay;
            nextDelay = false;

            for (int alu = 0; alu < MME64NumAlu; alu++) {
                switch (MME64Pred::mapping[group.global.predMode].name[alu]) {
                case 'T':
                    aluPred[alu] = pred;
                    break;
                case 'F':
                    aluPred[alu] = !pred;
                    break;
                case 'U':
                    aluPred[alu] = true;
                    break;
                default:
                    setError("Unparsable predicate mode");
                }
            }

            // Read the sources and execute
            for (int alu=0; alu<MME64NumAlu; alu++) {
                uint32_t sources[2];

                specialWriteAddr[alu] = -1;

                int srcAlu = alu;

                switch (group.alu[alu].op) {
                case MME64Op::MULH:
                    srcAlu = alu-1;
                    break;
                default:
                    break;
                }

                for (int src=0; src<2; src++) {
                    int readLatency = 0;

                    switch (group.alu[srcAlu].op) {
                    case MME64Op::DREAD:
                    case MME64Op::DWRITE:
                        readLatency = 1;
                        break;
                    default:
                        break;
                    }

                    sources[src] = cycleData.ReadReg(group.alu[srcAlu].src[src], true, srcAlu, readLatency);
                }

                switch (group.alu[alu].op) {
                case MME64Op::BLT:
                case MME64Op::BLTU:
                case MME64Op::BLE:
                case MME64Op::BLEU:
                case MME64Op::BEQ:
                    {
                        bool compareResult = false;

                        if (lwrDelay) {
                            setError("Branch in delay slot");
                        }

                        switch (group.alu[alu].op) {
                        case MME64Op::BLT:
                            compareResult = int32_t(sources[0]) < int32_t(sources[1]);
                            break;
                        case MME64Op::BLTU:
                            compareResult = uint32_t(sources[0]) < uint32_t(sources[1]);
                            break;
                        case MME64Op::BLE:
                            compareResult = int32_t(sources[0]) <= int32_t(sources[1]);
                            break;
                        case MME64Op::BLEU:
                            compareResult = uint32_t(sources[0]) <= uint32_t(sources[1]);
                            break;
                        case MME64Op::BEQ:
                            compareResult = sources[0] == sources[1];
                            break;
                        default:
                            setError("Internal error");
                            break;
                        }
                        bool branchTaken = GetBits(15, 15, group.alu[alu].immed) == (compareResult ? 1 : 0);
                        // Predict taken if the prediction bit is set and we're not targeting the end PC
                        bool predictTaken = GetBits(14, 14, group.alu[alu].immed) && ((group.alu[alu].immed & 0x1fff) != 0x1000);

                        if (pc[1] != END_PC) {
                            pcModified = true;
                            taken = branchTaken;
                            int ind = 1;
                            int fallthroughPC = pc[0] + 1;
                            // Delay slot
                            if (GetBits(13, 13, group.alu[alu].immed)) {
                                pc[ind++] = pc[0] + 1;
                                fallthroughPC = pc[0] + 2;
                                nextDelay = true;
                            }
                            if (branchTaken != predictTaken) {
                                for (; ind < 3; ind++) {
                                    pc[ind] = SKIP_PC;
                                }
                            } else if (predictTaken && run.previousCycleWasTaken) {
                                for (; ind < 2; ind++) {
                                    pc[ind] = SKIP_PC;
                                }
                            }
                            if (branchTaken) {
                                if ((group.alu[alu].immed & 0x1fff) == 0x1000) {
                                    for (; ind < 3; ind++) {
                                        pc[ind] = SKIP_PC;
                                    }
                                    pc[ind++] = END_PC;
                                } else {
                                    pc[ind++] = pc[0] + SignExtend(group.alu[alu].immed & 0x1fff, 13);
                                }
                            } else {
                                pc[ind++] = fallthroughPC;
                                if (group.global.endNext) {
                                    // Falling through and ENDNEXT branch has a one cycle mispredict even if correctly "predicted"
                                    if (!predictTaken) {
                                        pc[ind++] = SKIP_PC;
                                    }
                                    pc[ind++] = END_PC;
                                }
                            }
                            assert(ind <= PcDepth);
                            for (; ind < PcDepth; ind++) {
                                pc[ind] = pc[ind - 1] + 1;
                            }
                        }
                    }
                    break;
                case MME64Op::JAL:
                    if (lwrDelay) {
                        setError("JAL in delay slot");
                    }
                    if (pc[1] != END_PC && pc[2] != END_PC) {
                        int ind = 1;
                        int targetPC = SignExtend(group.alu[alu].immed & 0x1fff, 13);
                        if (GetBits(15, 15, group.alu[alu].immed)) targetPC += pc[0];
                        if (GetBits(14, 14, group.alu[alu].immed)) targetPC += sources[0];

                        if (GetBits(13, 13, group.alu[alu].immed)) {
                            pc[ind++] = pc[0] + 1;
                            nextDelay = true;
                        }
                        // Using a register read is a 2 cycle delay
                        if (GetBits(14, 14, group.alu[alu].immed)) {
                            for (; ind < 3; ind++) {
                                pc[ind] = SKIP_PC;
                            }
                        } else if (run.previousCycleWasTaken && targetPC != pc[0]+1) {
                            for (; ind < 2; ind++) {
                                pc[ind] = SKIP_PC;
                            }
                        }
                        pc[ind] = targetPC;
                        ind++;
                        taken = true;
                        pcModified = true;

                        for (; ind < PcDepth; ind++) {
                            pc[ind] = pc[ind - 1] + 1;
                        }
                    }

                    cycleData.aluOut[alu] = pc[0] + (GetBits(13, 13, group.alu[alu].immed) ? 2 : 1);
                    break;
                case MME64Op::LOOP:
                    if (lwrDelay) {
                        setError("LOOP in delay slot");
                    }
                    if (pc[1] != END_PC) {
                        run.loopCount = sources[0];
                        run.loopStart = pc[0] + 1;
                        run.loopEnd = pc[0] + SignExtend(group.alu[alu].immed, 16);
                        if (sources[0]) {
                            endingLoop = group.global.endNext;
                            // One or two instruction loops have 1 clock startup overhead
                            if (run.loopEnd == run.loopStart + 1 ||
                                run.loopEnd == run.loopStart + 2) {
                                pcModified = true;
                                pc[2] = pc[1];
                                pc[1] = SKIP_PC;
                                for (int ind = 3; ind < PcDepth; ind++) {
                                    pc[ind] = pc[ind - 1] + 1;
                                }
                            }
                        } else {
                            // If we're ending a 0 iteration loop, just skip then end
                            taken = true;
                            pcModified = true;
                            int ind = 1;
                            // 2 clocks for a zero-length loop (LOOP == predict taken)
                            pc[ind++] = SKIP_PC;
                            pc[ind++] = SKIP_PC;
                            if (group.global.endNext) {
                                pc[ind++] = END_PC;
                            } else {
                                pc[ind++] = pc[0] + SignExtend(group.alu[alu].immed, 16);
                            }
                            for (;  ind < PcDepth; ind++) {
                                pc[ind] = pc[ind - 1] + 1;
                            }
                        }
                    }
                    break;
                case MME64Op::SLT:
                    cycleData.aluOut[alu] = (int32_t(sources[0]) < int32_t(sources[1])) ? 0xffffffff : 0;
                    break;
                case MME64Op::SLTU:
                    cycleData.aluOut[alu] = (uint32_t(sources[0]) < uint32_t(sources[1])) ? 0xffffffff : 0;
                    break;
                case MME64Op::SLE:
                    cycleData.aluOut[alu] = (int32_t(sources[0]) <= int32_t(sources[1])) ? 0xffffffff : 0;
                    break;
                case MME64Op::SLEU:
                    cycleData.aluOut[alu] = (uint32_t(sources[0]) <= uint32_t(sources[1])) ? 0xffffffff : 0;
                    break;
                case MME64Op::SEQ:
                    cycleData.aluOut[alu] = (sources[0] == sources[1]) ? 0xffffffff : 0;
                    break;
                case MME64Op::CLZ:
                    {
                        int i=0;
                        while (i < 32 && !(sources[0] & (1<<(31-i)))) i++;
                        cycleData.aluOut[alu] = i;
                    }
                    break;
                case MME64Op::ADD:
                    cycleData.carry = 0;
                    // fallthrough
                case MME64Op::ADDC:
                    {
                        int64_t temp = int64_t(sources[0]) + int64_t(sources[1]) + cycleData.carry;
                        cycleData.aluOut[alu] = uint32_t(temp);
                        cycleData.carry = uint32_t((temp >> 32) & 1);
                    }
                    break;
                case MME64Op::SUB:
                    cycleData.carry = 1;
                    // fallthrough
                case MME64Op::SUBB:
                    {
                        int64_t temp = int64_t(sources[0]) + static_cast<int64_t>(~sources[1]) + cycleData.carry;
                        cycleData.aluOut[alu] = uint32_t(temp);
                        cycleData.carry = uint32_t((temp >> 32) & 1);
                    }
                    break;
                case MME64Op::MUL:
                    {
                        int64_t temp = int64_t(int32_t(sources[0])) * int64_t(int32_t(sources[1]));
                        cycleData.aluOut[alu] = uint32_t(temp);
                        writeLatency[alu] = 2;
                    }
                    break;
                case MME64Op::MULH:
                    if (alu > 0 && group.alu[alu-1].op == MME64Op::MULU) {
                        uint64_t temp = uint64_t(sources[0]) * uint64_t(sources[1]);
                        cycleData.aluOut[alu] = uint32_t(temp >> 32);
                        writeLatency[alu] = 2;
                    } else {
                        assert(alu > 0);
                        int64_t temp = int64_t(int32_t(sources[0])) * int64_t(int32_t(sources[1]));
                        cycleData.aluOut[alu] = uint32_t(temp>>32);
                        writeLatency[alu] = 2;
                    }
                    break;
                case MME64Op::MULU:
                    {
                        uint64_t temp = uint64_t(sources[0]) * uint64_t(sources[1]);
                        cycleData.aluOut[alu] = uint32_t(temp);
                        writeLatency[alu] = 2;
                    }
                    break;
                case MME64Op::EXTENDED:
                    {
                        MME64Extended::ExtendedEnum ext = (MME64Extended::ExtendedEnum)GetBits(15, 12, group.alu[alu].immed);

                        switch (ext) {
                        case MME64Extended::ReadFromSpecial:
                        case MME64Extended::WriteToSpecial:
                            {
                                MME64Special::SpecialEnum special = (MME64Special::SpecialEnum)GetBits(11, 0, sources[0]);
                                if (special > (MME64Special::SpecialEnum)MME64Special::count) {
                                    setError("Special register out of range");
                                }

                                uint32_t validMask = ~0;
                                bool writable = true;

                                switch (special) {
                                case MME64Special::LoadSource:
                                case MME64Special::MmeConfig:
                                    validMask = 1;
                                    break;
                                case MME64Special::PTimerHigh:
                                case MME64Special::PTimerLow:
                                    writable = false;
                                    break;
                                case MME64Special::Scratch:
                                    break;
                                }

                                if (ext == MME64Extended::WriteToSpecial) {
                                    if (!writable) {
                                        setError("Writing to a read-only special register");
                                    }
                                    if (sources[1] & ~validMask) {
                                        setError("Writing undefined bits of a special register");
                                    }

                                    if (aluPred[alu]) {
                                        specialWriteAddr[alu] = special;
                                        specialWriteData[alu] = sources[1];
                                    }
                                } else {
                                    assert(ext == MME64Extended::ReadFromSpecial);
                                    cycleData.aluOut[alu] = run.specialRegfile[special];
                                }
                            }
                            break;
                        }
                    }
                    break;
                case MME64Op::SLL:
                    cycleData.aluOut[alu] = sources[0] << (sources[1] & 0x1f);
                    break;
                case MME64Op::SRL:
                    cycleData.aluOut[alu] = sources[0] >> (sources[1] & 0x1f);
                    break;
                case MME64Op::SRA:
                    cycleData.aluOut[alu] = int32_t(sources[0]) >> (sources[1] & 0x1f);
                    break;
                case MME64Op::AND:
                    cycleData.aluOut[alu] = sources[0] & sources[1];
                    break;
                case MME64Op::NAND:
                    cycleData.aluOut[alu] = ~(sources[0] & sources[1]);
                    break;
                case MME64Op::OR:
                    cycleData.aluOut[alu] = sources[0] | sources[1];
                    break;
                case MME64Op::XOR:
                    cycleData.aluOut[alu] = sources[0] ^ sources[1];
                    break;
                case MME64Op::MERGE:
                    {
                        uint32_t sourceBit = GetBits(4, 0,   group.alu[alu].immed);
                        uint32_t width     = GetBits(9, 5,   group.alu[alu].immed);
                        uint32_t destBit   = GetBits(14, 10, group.alu[alu].immed);

                        cycleData.aluOut[alu] = Merge(sources[0], sources[1], sourceBit, destBit, width);
                    }
                    break;
                case MME64Op::STATE:
                    assert(stateAlu == -1); // Should have already checked that there's only one STATE per group
                    stateAlu = alu;
                    stateAddress = sources[0] + sources[1];
                    // Will do the actual read later in the cycle
                    writeLatency[alu] = 5;
                    break;
                case MME64Op::DREAD:
                    cycleData.aluOut[alu] = cycleData.ReadDataRam(sources[0]);
                    writeLatency[alu] = 2;
                    break;
                case MME64Op::DWRITE:
                    if (aluPred[alu]) {
                        cycleData.WriteDataRam(sources[0], sources[1]);
                    }
                    break;
                default:
                    setError("Unknown opcode");
                    break;
                }
            }

            // Output methods

            // Method skid is allocated on bank conflict
            bool willNeedMethodSkid = false;
            // Method skid is cleared if no methods are output
            bool canClearMethodSkid = true;
            bool bankUsed[4] = { false, false, false, false };

            for (int out=0; out<MME64NumOut; out++) {
                bool outPred = true;
                switch (MME64Pred::mapping[group.global.predMode].name[out+2]) {
                case 'T':
                    outPred = pred;
                    break;
                case 'F':
                    outPred = !pred;
                    break;
                case 'U':
                    outPred = true;
                    break;
                default:
                    setError("Unparsable predicate mode");
                }

                // Read the sources outside the predicate
                uint32_t methodSource = 0;
                if (group.output[out].method != MME64Out::NONE) {
                    methodSource = cycleData.ReadOut(group.output[out].method, group.output[out].emit, true);
                }
                uint32_t emitSource = 0;
                if (group.output[out].emit != MME64Out::NONE) {
                    emitSource = cycleData.ReadOut(group.output[out].emit, MME64Out::NONE, false);
                }

                if (outPred) {
                    uint32_t lwrMethod = run.method;

                    if (group.output[out].method != MME64Out::NONE) {
                        lwrMethod = methodSource;

                        run.method = lwrMethod & 0xfff;
                        run.methodInc = (lwrMethod >> 12) & 0xf;

                        lwrMethod &= 0xfff;
                    }

                    if (group.output[out].emit != MME64Out::NONE) {
                        // #define FE_MME64_SHADOW_HASH_FN_RTL(mthd_addr)  ({mthd_addr[2],mthd_addr[3]} ^ mthd_addr[1:0])
                        int bank = ((lwrMethod >> 1) & 0x2) ^ ((lwrMethod >> 3) & 0x1) ^ (lwrMethod & 0x3);
                        if (bankUsed[bank]) {
                            willNeedMethodSkid = true;
                        }
                        bankUsed[bank] = true;
                        canClearMethodSkid = false;

                        if (disableMemoryUnit || !memory.doMethod(lwrMethod, emitSource)) {
                            Release(callbackData, lwrMethod, emitSource);
#if BUILD_VSIM
                            if (simClocks != -1) {
                                if (lwrSimEvent == sim_events.end()) {
                                    printf("EMIT COUNT MISMATCH\n");
                                } else {
                                    if (lwrSimEvent->method != lwrMethod ||
                                        lwrSimEvent->data != emitSource) {
                                        printf("DATA MISMATCH expected %d:%d RTL %d:%d\n", lwrMethod, emitSource, lwrSimEvent->method, lwrSimEvent->data);
                                    }
                                    lwrSimEvent++;
                                }
                            }
#endif
                        }

                        run.method += run.methodInc;
                        run.method &= 0xfff;
                    }
                }
            }

            if (stateAlu != -1) {
                // XXX Technically incorrect since this means we can have a triple conflict that
                // takes another cycle to clear
                canClearMethodSkid = false;
                int bank = (stateAddress ^ (stateAddress >> 2)) & 3;
                if (bankUsed[bank]) {
                    willNeedMethodSkid = true;
                }
                bankUsed[bank] = true;
            }

            if (willNeedMethodSkid) {
                // Bank conflict with one conflict left unresolved causes a stall
                if (run.methodSkidInUse) {
                    run.cycles++;
                }
                run.methodSkidInUse = true;
            } else if (canClearMethodSkid) {
                run.methodSkidInUse = false;
            }

            // State reads after emit when collision happens, so handle here
            if (stateAlu != -1) {
                cycleData.aluOut[stateAlu] = GetState(stateAddress);
            }

            // Write destination
            for (uint32_t alu = 0; alu<MME64NumAlu; alu++) {
                if (aluPred[alu]) {
                    cycleData.WriteReg(group.alu[alu].dst, cycleData.aluOut[alu], writeLatency[alu]);
                }
            }

            for (uint32_t i = 0; i < cycleData.loadsUsed; i++) {
                run.loadData->pop_front();
                if (run.loadData == &run.fifoData) {
                    if (run.fifoInFlightData.size() == 0) {
                        setError("Trying to load from DMA FIFO with no loads in flight");
                        break;
                    }
                    run.fifoInFlightData[0]--;
                    if (run.fifoInFlightData[0] == 0) {
                        run.fifoInFlightData.pop_front();
                    }
                }
            }

            // Special register writes after all LOAD management
            for (int alu = 0; alu < MME64NumAlu; alu++) {
                if (specialWriteAddr[alu] != -1) {
                    run.specialRegfile[specialWriteAddr[alu]] = specialWriteData[alu];

                    if (run.specialRegfile[MME64Special::LoadSource]) {
                        run.loadData = &run.fifoData;
                    } else {
                        run.loadData = &run.methodData;
                    }
                }
            }

            run.previousCycleWasTaken = taken;

            // Handle all PC transitions
            if (pc[1] >= 0 && static_cast<unsigned int>(pc[1]) == run.loopEnd) {
                if (lwrDelay) {
                    setError("LOOP end in delay slot");
                }

                if (group.global.endNext) {
                    assert(pc[2] != SKIP_PC);
                    pc[0] = pc[1];
                    pc[1] = END_PC;
                } else if (run.loopCount > 1) {
                    run.loopCount--;
                    pc[0] = run.loopStart;
                    pc[1] = pc[0]+1;
                    for (int ind = 2; ind < PcDepth; ind++) {
                        pc[ind] = pc[ind-1] + 1;
                    }
                } else if (endingLoop) {
                    pc[0] = SKIP_PC;
                    pc[1] = END_PC;
                } else {
                    for (int ind = 0; ind < PcDepth-1; ind++) {
                        pc[ind] = pc[ind + 1];
                    }
                    pc[PcDepth - 1] = pc[PcDepth - 2] + 1;
                }
            } else if (pcModified) {
                // If pcModified is set the PC array is already changed - just shift
                for (int ind = 0; ind < PcDepth - 1; ind++) {
                    pc[ind] = pc[ind + 1];
                }
                pc[PcDepth - 1] = pc[PcDepth - 2] + 1;
            } else if (group.global.endNext &&
                       group.alu[0].op != MME64Op::LOOP) {
                pc[0] = pc[1];
                // Branch bubble takes priority over endnext
                // Can happen with ENDNEXT in a delay slot
                if (pc[1] == SKIP_PC) {
                    pc[1] = pc[2];
                    pc[2] = END_PC;
                } else {
                    pc[1] = END_PC;
                }
            } else {
                for (int ind = 0; ind < PcDepth - 1; ind++) {
                    pc[ind] = pc[ind + 1];
                }
                pc[PcDepth - 1] = pc[PcDepth - 2] + 1;
            }
        } else {
            for (int ind = 0; ind < PcDepth - 1; ind++) {
                pc[ind] = pc[ind + 1];
            }
            pc[PcDepth - 1] = pc[PcDepth - 2] + 1;
        }

        run.cycles++;

        run.specialRegfile[MME64Special::PTimerLow] = run.cycles;

        if (run.cycles > CYCLE_LIMIT) {
            setError("Error - exceeded max cycle count\n");
        }

        if (isError) return -1;
    }

    int methodsLeft = static_cast<int>(run.methodData.size());
    assert((methodsLeft >= 0)
           && (static_cast<unsigned>(methodsLeft) == run.methodData.size()));
    // If the skid is in use at the end, it needs one cycle to drain
    // Overlaps with discarding leftover methods
    if (run.methodSkidInUse && methodsLeft <= 4) {
        run.cycles++;
        tprintf("clock: %d, method skid stall", run.cycles);
    }

    if (methodsLeft) {
        // Can eliminate the first 8 leftovers for free
        if (methodsLeft <= 8) {
            // Assuming they've had enough cycles to load into the FIFO at 2 per clk
            run.cycles += max(0, (methodsLeft + 1) / 2 - run.cycles);
        } else {
            // Behavior after the first 8 is 2 bubble clocks then 2 per clock
            run.cycles += 2 + (methodsLeft - 8 + 1) / 2;
            //run.cycles += (methodsLeft - min(4, methodsLeft) + 1) / 2;
        }
        tprintf("clock: %d, discard %d leftover load data", run.cycles, methodsLeft);
    }

#if BUILD_VSIM
    if (simClocks != run.cycles) {
        printf("SW simulated expected %d clocks, RTL generated %d\n", run.cycles, simClocks);
    }
#endif
    return run.cycles;
}
