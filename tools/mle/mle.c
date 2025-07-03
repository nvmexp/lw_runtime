/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright 2020-2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

#include <assert.h>
#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef FUZZ_MLE

FILE* s_DevNull = NULL;

#   define STDOUT s_DevNull
#   define STDERR s_DevNull
#else
#   define STDOUT stdout
#   define STDERR stderr
#endif

typedef enum
{
    VarInt      = 0,
    Double      = 1,
    Bytes       = 2,
    Float       = 5,
    IlwalidWire = 8
} Wire;

typedef enum
{
    f_entry           = 1,
    f_file_type       = 2,
    f_sync            = 3,

    f_uid_delta       = 1,
    f_timestamp_delta = 2,
    f_thread_id       = 3,
    f_test_id         = 4,
    f_dev_id          = 5,
    f_priority        = 6,
    f_print           = 7,
} MLEIndex;

typedef struct
{
    uint64_t uid;
    uint64_t timestamp;
    int32_t  threadId;
    int32_t  testId;
    int32_t  devId;
    unsigned priority;
} Context;

static void ResetContext(Context* pCtx)
{
    pCtx->uid       = 0;
    pCtx->timestamp = 0;
    pCtx->threadId  = 0;
    pCtx->testId    = 0;
    pCtx->devId     = 0;
    pCtx->priority  = 0;
}

typedef struct
{
    uint8_t* begin;
    uint8_t* end;
} Range;

static uint64_t LoadVarInt(Range* pRange)
{
    uint8_t*       begin = pRange->begin;
    uint8_t* const end   = pRange->end;

    uint64_t value = 0;
    int      bit   = 0;
    bool     done  = false;

    while ((begin < end) && !done)
    {
        uint8_t lwrByte = *(begin++);
        done = lwrByte <= 0x7FU;
        value += ((uint64_t)(lwrByte & 0x7FU)) << bit;
        bit += 7;
    }

    pRange->begin = begin;

    return value;
}

static int64_t LoadSignedVarInt(Range* pRange)
{
    const int64_t value = LoadVarInt(pRange);
    // In protobuf format, signed numbers are encoded using "zigzag" encoding,
    // where the sign bit is migrated to bit 0.
    return ((value << 63) >> 63) ^ (value >> 1);
}

static int32_t LoadId(Range* pRange, int32_t oldId)
{
    const int32_t newId = (int32_t)LoadSignedVarInt(pRange);
    if (newId)
    {
        return (newId == -1) ? newId : (newId - 1);
    }
    else
    {
        return oldId;
    }
}

static uint32_t LoadPri(Range* pRange, uint32_t oldPri)
{
    const uint32_t newPri = LoadVarInt(pRange);
    return newPri ? newPri : oldPri;
}

typedef struct
{
    MLEIndex index;
    Wire     wire;
} FieldHdr;

static FieldHdr LoadFieldHdr(Range* pRange)
{
    const uint64_t value = LoadVarInt(pRange);
    if (value < 0x800000000ULL)
    {
        FieldHdr hdr =
        {
            .index = (MLEIndex)(value >> 3),
            .wire  = (Wire)(value & 7U)
        };
        return hdr;
    }
    else
    {
        FieldHdr hdr = { .index = 0U, .wire = IlwalidWire };
        return hdr;
    }
}

static bool SkipField(FieldHdr hdr, Range* pRange)
{
    bool error = false;

    switch (hdr.wire)
    {
        case VarInt:
            LoadVarInt(pRange);
            break;

        case Double:
            pRange->begin += 8;
            break;

        case Bytes:
            {
                const int64_t size = (int64_t)LoadVarInt(pRange);
                if (size <= pRange->end - pRange->begin)
                {
                    pRange->begin += size;
                }
                else
                {
                    ++pRange->begin;
                    error = true;
                }
            }
            break;

        case Float:
            pRange->begin += 4;
            break;

        default:
            break;
    }

    if (pRange->begin > pRange->end)
    {
        error = true;
    }

    return error;
}

typedef struct
{
    FILE*       f;
    const char* filename;
    uint8_t*    buf;
    unsigned    capacity;
    unsigned    size;
    unsigned    pos;
    uint64_t    filePos;
    bool        error;
} Loader;

static void InitLoader(Loader* pLoader, FILE* f, const char* filename, uint8_t* buf, unsigned size)
{
    pLoader->f        = f;
    pLoader->filename = filename;
    pLoader->buf      = buf;
    pLoader->capacity = size;
    pLoader->size     = 0;
    pLoader->pos      = 0;
    pLoader->filePos  = 0;
    pLoader->error    = false;
}

static bool LoadNextEntry(Loader* pLoader, Range* pRange, FieldHdr* pHdr)
{
    const unsigned remaining = pLoader->size - pLoader->pos;

    if (((remaining < pLoader->capacity / 2) && (pLoader->size == pLoader->capacity))
        || ((pLoader->filePos == 0) && (pLoader->pos == 0)))
    {
        if (pLoader->pos > 0)
        {
            pLoader->filePos += pLoader->pos;
            if (pLoader->pos < pLoader->size)
            {
                memmove(pLoader->buf, pLoader->buf + pLoader->pos, pLoader->size - pLoader->pos);
                pLoader->size -= pLoader->pos;
            }
            else
            {
                pLoader->size = 0;
            }
            pLoader->pos  = 0;
        }

        const unsigned toRead = pLoader->capacity - pLoader->size;
#if defined(FUZZ_MLE) && (FUZZ_MLE == 1)
        const size_t numRead = 0;
#else
        const size_t numRead = fread(pLoader->buf + pLoader->size, 1, toRead, pLoader->f);
        if ((numRead < toRead) && ferror(pLoader->f))
        {
            perror(pLoader->filename);
            pLoader->error = true;
        }
#endif
        pLoader->size += numRead;
    }

    if (pLoader->pos >= pLoader->size)
    {
        return false;
    }

    Range range =
    {
        .begin = pLoader->buf + pLoader->pos,
        .end   = pLoader->buf + pLoader->size
    };

    // Workaround for concatenated MLE logs where random data was inserted
    // in-between these logs.
    static const uint8_t validMleHeader[] = { 0x12U, 0x03U, 0x4DU, 0x4LW, 0x45U };
    switch (*range.begin)
    {
        case 0x0AU: // f_entry, Bytes
        case 0x1AU: // f_sync, Bytes
            break;

        case 0x12U: // f_file_type, Bytes
            {
                if ((pLoader->size - pLoader->pos >= 5) &&
                    (memcmp(range.begin, validMleHeader, sizeof(validMleHeader)) == 0))
                {
                    break;
                }
            }
            // fall-through
        default:
            {
                uint8_t* fileHeader = range.begin;
                do
                {
                    fileHeader = (uint8_t *)memchr(fileHeader, 0x12, range.end - fileHeader);
                    if (!fileHeader)
                    {
                        fileHeader = range.end;
                        break;
                    }

                    if (range.end - fileHeader >= sizeof(validMleHeader) &&
                        memcmp(fileHeader, validMleHeader, sizeof(validMleHeader)) == 0)
                    {
                        break;
                    }

                    ++fileHeader;
                } while (fileHeader < range.end);

                fprintf(STDERR, "Skipping %" PRIu64 " bytes, possibly corrupted log\n",
                        (uint64_t)(fileHeader - range.begin));
                range.begin = fileHeader;
                pLoader->pos = (unsigned)(fileHeader - pLoader->buf);

                if (range.begin == range.end)
                {
                    return LoadNextEntry(pLoader, pRange, pHdr);
                }
            }
    }

    FieldHdr hdr = LoadFieldHdr(&range);

    if (hdr.wire == Bytes)
    {
        const int64_t size = (int64_t)LoadVarInt(&range);

        if (size <= (range.end - range.begin))
        {
            range.end = range.begin + size;
        }
        else
        {
            fprintf(STDERR, "Invalid entry size at pos %" PRIu64 "\n",
                    pLoader->filePos + pLoader->pos);
            range.end = range.begin;
        }
    }
    else
    {
        fprintf(STDERR, "Unrecognized entry index %u wire %u at pos %" PRIu64 "\n",
                (unsigned)hdr.index, (unsigned)hdr.wire,
                pLoader->filePos + pLoader->pos);
        pLoader->error = true;
        SkipField(hdr, &range);
        range.end = range.begin;
    }

    pLoader->pos = (unsigned)(range.end - pLoader->buf);
    *pRange = range;
    *pHdr   = hdr;

    return true;
}

enum PrintColumns
{
    m_uid       = 1 << 0,
    m_timestamp = 1 << 1,
    m_threadId  = 1 << 2,
    m_testId    = 1 << 3,
    m_devId     = 1 << 4,
    m_priority  = 1 << 5,
    m_color     = 1 << 6,
    m_binary    = 1 << 7,
    m_level_bit =      8, // must be last
};

static void PrintId(char* buf, size_t bufSize, int32_t value)
{
    if (value == -1)
        snprintf(buf, bufSize, "N/A ");
    else
        snprintf(buf, bufSize, "%3d ", value);
}

typedef struct
{
    char uid[22];
    char timestamp[22];
    char threadId[14];
    char testId[14];
    char devId[14];
    char pri[4];
} ContextPrint;

static void PrintContext(uint32_t mask, Context* pCtx, ContextPrint* print)
{
    print->uid[0]       = 0;
    print->timestamp[0] = 0;
    print->threadId[0]  = 0;
    print->testId[0]    = 0;
    print->devId[0]     = 0;
    print->pri[0]       = 0;

    if (mask & m_uid)
        snprintf(print->uid,       sizeof(print->uid),       "%" PRIu64 "\t", pCtx->uid);
    if (mask & m_timestamp)
        snprintf(print->timestamp, sizeof(print->timestamp), "%" PRIu64 "\t", pCtx->timestamp);
    if (mask & m_threadId)
        PrintId(print->threadId,   sizeof(print->threadId),  pCtx->threadId);
    if (mask & m_testId)
        PrintId(print->testId,     sizeof(print->testId),    pCtx->testId);
    if (mask & m_devId)
        PrintId(print->devId,      sizeof(print->devId),     pCtx->devId);
    if (mask & m_priority)
        snprintf(print->pri,       sizeof(print->pri),       "%2u ",  pCtx->priority & 63U);
}

// ANSI escape sequences for color codes
static const char s_ColorNone[]   = "\033[0m";  // restores default console color
static const char s_ColorHigh[]   = "\033[1m";  // bold
static const char s_ColorLow[]    = "\033[37m"; // gray
static const char s_ColorBinary[] = "\033[37m"; // gray

static bool PrintLine(uint32_t mask, Context* pCtx, Range* pRange)
{
    if (pCtx->priority < (mask >> m_level_bit))
        return false;

    ContextPrint cp;
    PrintContext(mask, pCtx, &cp);

    uint64_t size = LoadVarInt(pRange);

    if ((int64_t)size > pRange->end - pRange->begin)
    {
        fprintf(STDERR, "Invalid size\n");
        return true;
    }

    const char* openColor = ! (mask & m_color) ? "" :
                            ((pCtx->priority < 3) || (pCtx->priority > 7)) ? s_ColorLow :
                            ((pCtx->priority > 3) && (pCtx->priority < 7)) ? s_ColorHigh :
                            "";
    const char* noColor = (mask & m_color) ? s_ColorNone : "";

    // Print multiple log lines separately.  If a single print from MODS contains
    // multiple lines separated by \ns (LFs), prefix each of these lines with
    // context columns.
    for (;;)
    {
        const char* const text     = (const char*)pRange->begin;
        const char* const eol      = (const char*)memchr(text, '\n', (size_t)size);
        const uint64_t    textSize = eol ? ((uint64_t)(eol - text)) : size;

        fprintf(STDOUT, "%s%s%s%s%s%s%s%.*s%s\n",
                cp.uid, cp.timestamp, cp.threadId, cp.testId, cp.devId, cp.pri, openColor,
                (int)textSize, text, noColor);

        pRange->begin += textSize;
        size          -= textSize;

        if (!eol)
            break;

        ++pRange->begin;
        --size;
    }

    return false;
}

static bool PrintBinary(uint32_t mask, Context* pCtx, FieldHdr hdr, Range* pRange)
{
    if (pCtx->priority < (mask >> m_level_bit))
        return false;

    ContextPrint cp;
    PrintContext(mask, pCtx, &cp);

    if (hdr.wire == Bytes)
    {
        const int64_t size = (int64_t)LoadVarInt(pRange);

        if (size > pRange->end - pRange->begin)
        {
            fprintf(STDERR, "Invalid size %" PRIu64 "\n", size);
            return true;
        }
    }

    const char* openColor = (mask & m_color) ? s_ColorBinary : "";
    const char* noColor   = (mask & m_color) ? s_ColorNone   : "";

    char  wireBuf[3] = { 0 };
    char* wire       = wireBuf;
    switch (hdr.wire)
    {
        case VarInt: wire = "int";    break;
        case Bytes:  wire = "bytes";  break;
        case Float:  wire = "float";  break;
        case Double: wire = "double"; break;
        default:     snprintf(wireBuf, sizeof(wireBuf) - 1, "%u", (unsigned)hdr.wire);
    }

    fprintf(STDOUT, "%s%s%s%s%s%s%sentry %u (%s)",
            cp.uid, cp.timestamp, cp.threadId, cp.testId, cp.devId, cp.pri, openColor,
            (unsigned)hdr.index, wire);

    switch (hdr.wire)
    {
        case Bytes:
            while (pRange->begin < pRange->end)
            {
                fprintf(STDOUT, " %02x", *(pRange->begin++));
            }
            fprintf(STDOUT, "%s\n", noColor);
            break;

        case VarInt:
            {
                const uint64_t value = LoadVarInt(pRange);
                fprintf(STDOUT, " %" PRIu64 "\n", value);
            }
            break;

        case Float:
            if (pRange->begin + sizeof(float) <= pRange->end)
            {
                const double value = *(float*)pRange->begin;
                pRange->begin += sizeof(float);
                fprintf(STDOUT, " %f\n", value);
            }
            else
                return true;
            break;

        case Double:
            if (pRange->begin + sizeof(double) <= pRange->end)
            {
                const double value = *(double*)pRange->begin;
                pRange->begin += sizeof(double);
                fprintf(STDOUT, " %f\n", value);
            }
            else
                return true;
            break;

        default:
            fprintf(STDOUT, " INVALID WIRE TYPE, POSSIBLY CORRUPTED LOG\n");
            return true;
    }

    return false;
}

#define SKIP_NOTHING

#define CHECK_WIRE(wireType, lwstomSkip)                  \
    if (hdr.wire != (wireType))                           \
    {                                                     \
        lwstomSkip;                                       \
        fprintf(STDERR, "Invalid wire %u for field %u\n", \
                (unsigned)hdr.wire, (unsigned)hdr.index); \
        error = true;                                     \
        break;                                            \
    }

static bool PrintMLEEntry(uint32_t mask, Context* pCtx, Range ctxRange)
{
    bool error = false;

    Range dataRange = ctxRange;

    Context newCtx = *pCtx;
    newCtx.uid = pCtx->uid + 1;

    while (ctxRange.begin < ctxRange.end)
    {
        const FieldHdr hdr = LoadFieldHdr(&ctxRange);
        switch (hdr.index)
        {
            case f_uid_delta:
                CHECK_WIRE(VarInt, SkipField(hdr, &ctxRange));
                newCtx.uid = pCtx->uid + (uint64_t)LoadSignedVarInt(&ctxRange) + 1;
                break;

            case f_timestamp_delta:
                CHECK_WIRE(VarInt, SkipField(hdr, &ctxRange));
                newCtx.timestamp = pCtx->timestamp + (uint64_t)LoadSignedVarInt(&ctxRange);
                break;

            case f_thread_id:
                CHECK_WIRE(VarInt, SkipField(hdr, &ctxRange));
                newCtx.threadId = LoadId(&ctxRange, pCtx->threadId);
                break;

            case f_test_id:
                CHECK_WIRE(VarInt, SkipField(hdr, &ctxRange));
                newCtx.testId = LoadId(&ctxRange, pCtx->testId);
                break;

            case f_dev_id:
                CHECK_WIRE(VarInt, SkipField(hdr, &ctxRange));
                newCtx.devId = LoadId(&ctxRange, pCtx->devId);
                break;

            case f_priority:
                CHECK_WIRE(VarInt, SkipField(hdr, &ctxRange));
                newCtx.priority = LoadPri(&ctxRange, pCtx->priority);
                break;

            default:
                error = SkipField(hdr, &ctxRange) || error;
                break;
        }
    }

    *pCtx = newCtx;

    while (dataRange.begin < dataRange.end)
    {
        const FieldHdr hdr = LoadFieldHdr(&dataRange);
        switch (hdr.index)
        {
            case f_print:
                error = PrintLine(mask, pCtx, &dataRange) || error;
                break;

            case f_uid_delta:
            case f_timestamp_delta:
            case f_thread_id:
            case f_test_id:
            case f_dev_id:
            case f_priority:
                SkipField(hdr, &dataRange);
                break;

            default:
                if (mask & m_binary)
                {
                    error = PrintBinary(mask, pCtx, hdr, &dataRange) || error;
                }
                else
                {
                    SkipField(hdr, &dataRange);
                }
                break;
        }
    }

    return error;
}

static void PrintHeader(uint32_t mask)
{
    if (mask & m_uid)
        fprintf(STDOUT, "UID\t");
    if (mask & m_timestamp)
        fprintf(STDOUT, "timestamp\t");
    if (mask & m_threadId)
        fprintf(STDOUT, "thr ");
    if (mask & m_testId)
        fprintf(STDOUT, "tst ");
    if (mask & m_devId)
        fprintf(STDOUT, "dev ");
    if (mask & m_priority)
        fprintf(STDOUT, "pr ");
    fprintf(STDOUT, "log\n");
}

static int PrintMLEFileWithLoader(Loader* pLoader, uint32_t mask)
{
    bool error = false;

    Context ctx;
    ResetContext(&ctx);

    Range    range;
    FieldHdr hdr;
    while (LoadNextEntry(pLoader, &range, &hdr))
    {
        switch (hdr.index)
        {
            case f_entry:
                CHECK_WIRE(Bytes, SKIP_NOTHING);
                error = PrintMLEEntry(mask, &ctx, range) || error;
                break;

            case f_file_type:
                PrintHeader(mask);
                // fall through
            case f_sync:
                CHECK_WIRE(Bytes, SKIP_NOTHING);
                ResetContext(&ctx);
                break;

            default:
                // Ignore anything else
                break;
        }
    }

    return (pLoader->error || error) ? 1 : 0;
}

static int PrintMLEFile(FILE* f, const char* filename, uint32_t mask)
{
    static uint8_t buffer[1024*1024];
    Loader loader;
    InitLoader(&loader, f, filename, buffer, sizeof(buffer));

    return PrintMLEFileWithLoader(&loader, mask);
}

#if defined(FUZZ_MLE) && (FUZZ_MLE == 1)

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    uint8_t buf[4096];
    if (size > sizeof(buf))
    {
        return 1;
    }
    memcpy(buf, data, size);

    s_DevNull = fopen("/dev/null", "w");
    if (!s_DevNull)
    {
        return 1;
    }

    Loader loader;
    InitLoader(&loader, NULL, "fuzz input", buf, size);
    loader.size     = size;
    loader.capacity = size + 1; // this prevents loading more from file

    PrintMLEFileWithLoader(&loader, (1U << m_level_bit) - 1U);

    fclose(s_DevNull);

    return 0;
}

#else

static bool IsOption(const char* string, char shortOpt, const char* longOpt)
{
    if (string[1] == '-')
    {
        return strcmp(string + 2, longOpt) == 0;
    }

    char*      charPtr = strchr(string, shortOpt);
    const bool have    = charPtr != NULL;
    if (charPtr)
    {
        do
        {
            *charPtr = '\n';
            charPtr  = strchr(string, shortOpt);
        } while (charPtr);
    }
    return have;
}

static char FindBadOption(const char* string, const char* opts)
{
    if ((string[0] != '-') || (string[1] == '-'))
    {
        return 0;
    }

    for (;;)
    {
        const char opt = *(opts++);
        if (!opt)
        {
            break;
        }

        const char* const charPtr = strchr(string, opt);
        if (charPtr)
        {
            if (charPtr[1] != 0)
            {
                return opt;
            }
        }
    }

    return 0;
}

static char RemainingOption(const char* string)
{
    if (string[1] == '-')
    {
        return 0;
    }

    char c;
    do
    {
        ++string;
        c = *string;
    } while (c == '\n');

    return c;
}

static const char s_Help[] =
    "Usage: mle [OPTION]... [FILE]...\n"
    "\n"
    "Decodes MLE files.\n"
    "If no files are specified on the command line, reads MLE from stdin.\n"
    "\n"
    "Options:\n"
    "    -h, --help         Print this help.\n"
    "    -v, --verbose      Print verbose log entries.\n"
    "    -d, --debug        Print verbose and debug log entries.\n"
    "    -c, --nocolor      Disable color printing.\n"
    "    -a, --all          Print all columns and all entries.\n"
    "    -n, --none         Print no columns.\n"
    "    -b, --binary       Print binary entries (not decoded).\n"
    "    -l, --level LEVEL  Print only entries at or above pri LEVEL.\n"
    "    -f, --flags FLAGS  Enable individual columns.\n"
    "\n"
    "Flags:\n"
    "    u  UID\n"
    "    i  Timestamp\n"
    "    h  Thread ID\n"
    "    t  Test ID\n"
    "    d  Device ID\n"
    "    p  Print level\n";

int main(int argc, char* argv[])
{
#ifdef FUZZ_MLE
    s_DevNull = fopen("/dev/null", "w");
    if (!s_DevNull)
    {
        return 1;
    }
#endif

    // Parse options
    uint32_t mask  = m_threadId | m_devId;
    uint32_t flags = m_color;
    int      i     = 1;
    while ((i < argc) && (argv[i][0] == '-'))
    {
        const char* opt = argv[i++];
        bool        ok  = false;
        const char  bad = FindBadOption(opt, "lf");
        if (bad)
        {
            fprintf(STDERR, "Option %c must be last\n", bad);
            return 1;
        }
        if (IsOption(opt, 'h', "help"))
        {
            const size_t size = sizeof(s_Help) - 1;
            return fwrite(s_Help, 1, size, STDOUT) == size ? 0 : 1;
        }
        if (IsOption(opt, 'c', "nocolor"))
        {
            flags &= ~m_color;
            ok    = true;
        }
        if (IsOption(opt, 'a', "all"))
        {
            mask = ~m_color & ((1U << m_level_bit) - 1U);
            ok   = true;
        }
        if (IsOption(opt, 'n', "none"))
        {
            mask = 0;
            ok   = true;
        }
        if (IsOption(opt, 'b', "binary"))
        {
            flags |= m_binary;
            ok    = true;
        }
        if (IsOption(opt, 'v', "verbose"))
        {
            flags |= 2 << m_level_bit;
            ok    = true;
        }
        if (IsOption(opt, 'd', "debug"))
        {
            flags |= 1 << m_level_bit;
            ok    = true;
        }
        if (IsOption(opt, 'l', "level"))
        {
            ok = true;
            const char* levArg = i < argc ? argv[i++] : "0";
            char* end = NULL;
            const long level = strtol(levArg, &end, 10);
            if ((level < 1) || (level > 15) || (end && *end))
            {
                fprintf(STDERR, "Invalid pri level %s\n", levArg);
                return 1;
            }
            flags |= (uint32_t)level << m_level_bit;
        }
        if (IsOption(opt, 'f', "flags"))
        {
            ok = true;
            const char* flagsArg = i < argc ? argv[i++] : "";
            while (*flagsArg)
            {
                const char f = *(flagsArg++);
                switch (f)
                {
                    case 'u': flags |= m_uid;       break;
                    case 'i': flags |= m_timestamp; break;
                    case 'h': flags |= m_threadId;  break;
                    case 't': flags |= m_testId;    break;
                    case 'd': flags |= m_devId;     break;
                    case 'p': flags |= m_priority;  break;
                    default:
                        fprintf(STDERR, "Unrecognized flags %c\n", f);
                        return 1;
                }
            }
        }

        const char rem = RemainingOption(opt);
        if (rem)
        {
            fprintf(STDERR, "Unrecognized option -%c\n", rem);
            return 1;
        }
        else if (!ok)
        {
            fprintf(STDERR, "Unrecognized option %s\n", opt);
            return 1;
        }
    }
    mask |= flags;

    int error = 0;
    if (i < argc)
    {
        // Process MLE files specified on the command line
        for ( ; i < argc; i++)
        {
            FILE* f = fopen(argv[i], "rb");
            if (f == NULL)
            {
                ++error;
                perror(argv[i]);
                continue;
            }
            error += PrintMLEFile(f, argv[i], mask);
            fclose(f);
        }
    }
    else
    {
        // Load MLE file from stdin
        error += PrintMLEFile(stdin, "stdin", mask);
    }

    return error ? 1 : 0;
}

#endif
