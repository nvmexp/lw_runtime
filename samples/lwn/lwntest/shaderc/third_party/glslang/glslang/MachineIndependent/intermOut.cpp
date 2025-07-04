//
// Copyright (C) 2002-2005  3Dlabs Inc. Ltd.
// Copyright (C) 2012-2016 LunarG, Inc.
// Copyright (C) 2017 ARM Limited.
// Modifications Copyright (C) 2020 Advanced Micro Devices, Inc. All rights reserved.
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
//    Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
//    Redistributions in binary form must reproduce the above
//    copyright notice, this list of conditions and the following
//    disclaimer in the documentation and/or other materials provided
//    with the distribution.
//
//    Neither the name of 3Dlabs Inc. Ltd. nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROLWREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//

#ifndef GLSLANG_WEB

#include "localintermediate.h"
#include "../Include/InfoSink.h"

#ifdef _MSC_VER
#include <cfloat>
#else
#include <cmath>
#endif
#include <cstdint>

namespace {

bool IsInfinity(double x) {
#ifdef _MSC_VER
    switch (_fpclass(x)) {
    case _FPCLASS_NINF:
    case _FPCLASS_PINF:
        return true;
    default:
        return false;
    }
#else
    return std::isinf(x);
#endif
}

bool IsNan(double x) {
#ifdef _MSC_VER
    switch (_fpclass(x)) {
    case _FPCLASS_SNAN:
    case _FPCLASS_QNAN:
        return true;
    default:
        return false;
    }
#else
  return std::isnan(x);
#endif
}

}

namespace glslang {

//
// Two purposes:
// 1.  Show an example of how to iterate tree.  Functions can
//     also directly call Traverse() on children themselves to
//     have finer grained control over the process than shown here.
//     See the last function for how to get started.
// 2.  Print out a text based description of the tree.
//

//
// Use this class to carry along data from node to node in
// the traversal
//
class TOutputTraverser : public TIntermTraverser {
public:
    TOutputTraverser(TInfoSink& i) : infoSink(i), extraOutput(NoExtraOutput) { }

    enum EExtraOutput {
        NoExtraOutput,
        BinaryDoubleOutput
    };
    void setDoubleOutput(EExtraOutput extra) { extraOutput = extra; }

    virtual bool visitBinary(TVisit, TIntermBinary* node);
    virtual bool visitUnary(TVisit, TIntermUnary* node);
    virtual bool visitAggregate(TVisit, TIntermAggregate* node);
    virtual bool visitSelection(TVisit, TIntermSelection* node);
    virtual void visitConstantUnion(TIntermConstantUnion* node);
    virtual void visitSymbol(TIntermSymbol* node);
    virtual bool visitLoop(TVisit, TIntermLoop* node);
    virtual bool visitBranch(TVisit, TIntermBranch* node);
    virtual bool visitSwitch(TVisit, TIntermSwitch* node);

    TInfoSink& infoSink;
protected:
    TOutputTraverser(TOutputTraverser&);
    TOutputTraverser& operator=(TOutputTraverser&);

    EExtraOutput extraOutput;
};

//
// Helper functions for printing, not part of traversing.
//

static void OutputTreeText(TInfoSink& infoSink, const TIntermNode* node, const int depth)
{
    int i;

    infoSink.debug << node->getLoc().string << ":";
    if (node->getLoc().line)
        infoSink.debug << node->getLoc().line;
    else
        infoSink.debug << "? ";

    for (i = 0; i < depth; ++i)
        infoSink.debug << "  ";
}

//
// The rest of the file are the traversal functions.  The last one
// is the one that starts the traversal.
//
// Return true from interior nodes to have the external traversal
// continue on to children.  If you process children yourself,
// return false.
//

bool TOutputTraverser::visitBinary(TVisit /* visit */, TIntermBinary* node)
{
    TInfoSink& out = infoSink;

    OutputTreeText(out, node, depth);

    switch (node->getOp()) {
    case EOpAssign:                   out.debug << "move second child to first child";           break;
    case EOpAddAssign:                out.debug << "add second child into first child";          break;
    case EOpSubAssign:                out.debug << "subtract second child into first child";     break;
    case EOpMulAssign:                out.debug << "multiply second child into first child";     break;
    case EOpVectorTimesMatrixAssign:  out.debug << "matrix mult second child into first child";  break;
    case EOpVectorTimesScalarAssign:  out.debug << "vector scale second child into first child"; break;
    case EOpMatrixTimesScalarAssign:  out.debug << "matrix scale second child into first child"; break;
    case EOpMatrixTimesMatrixAssign:  out.debug << "matrix mult second child into first child";  break;
    case EOpDivAssign:                out.debug << "divide second child into first child";       break;
    case EOpModAssign:                out.debug << "mod second child into first child";          break;
    case EOpAndAssign:                out.debug << "and second child into first child";          break;
    case EOpInclusiveOrAssign:        out.debug << "or second child into first child";           break;
    case EOpExclusiveOrAssign:        out.debug << "exclusive or second child into first child"; break;
    case EOpLeftShiftAssign:          out.debug << "left shift second child into first child";   break;
    case EOpRightShiftAssign:         out.debug << "right shift second child into first child";  break;

    case EOpIndexDirect:   out.debug << "direct index";   break;
    case EOpIndexIndirect: out.debug << "indirect index"; break;
    case EOpIndexDirectStruct:
        {
            bool reference = node->getLeft()->getType().isReference();
            const TTypeList *members = reference ? node->getLeft()->getType().getReferentType()->getStruct() : node->getLeft()->getType().getStruct();
            out.debug << (*members)[node->getRight()->getAsConstantUnion()->getConstArray()[0].getIConst()].type->getFieldName();
            out.debug << ": direct index for structure";      break;
        }
    case EOpVectorSwizzle: out.debug << "vector swizzle"; break;
    case EOpMatrixSwizzle: out.debug << "matrix swizzle"; break;

    case EOpAdd:    out.debug << "add";                     break;
    case EOpSub:    out.debug << "subtract";                break;
    case EOpMul:    out.debug << "component-wise multiply"; break;
    case EOpDiv:    out.debug << "divide";                  break;
    case EOpMod:    out.debug << "mod";                     break;
    case EOpRightShift:  out.debug << "right-shift";  break;
    case EOpLeftShift:   out.debug << "left-shift";   break;
    case EOpAnd:         out.debug << "bitwise and";  break;
    case EOpInclusiveOr: out.debug << "inclusive-or"; break;
    case EOpExclusiveOr: out.debug << "exclusive-or"; break;
    case EOpEqual:            out.debug << "Compare Equal";                 break;
    case EOpNotEqual:         out.debug << "Compare Not Equal";             break;
    case EOpLessThan:         out.debug << "Compare Less Than";             break;
    case EOpGreaterThan:      out.debug << "Compare Greater Than";          break;
    case EOpLessThanEqual:    out.debug << "Compare Less Than or Equal";    break;
    case EOpGreaterThanEqual: out.debug << "Compare Greater Than or Equal"; break;
    case EOpVectorEqual:      out.debug << "Equal";                         break;
    case EOpVectorNotEqual:   out.debug << "NotEqual";                      break;

    case EOpVectorTimesScalar: out.debug << "vector-scale";          break;
    case EOpVectorTimesMatrix: out.debug << "vector-times-matrix";   break;
    case EOpMatrixTimesVector: out.debug << "matrix-times-vector";   break;
    case EOpMatrixTimesScalar: out.debug << "matrix-scale";          break;
    case EOpMatrixTimesMatrix: out.debug << "matrix-multiply";       break;

    case EOpLogicalOr:  out.debug << "logical-or";   break;
    case EOpLogicalXor: out.debug << "logical-xor"; break;
    case EOpLogicalAnd: out.debug << "logical-and"; break;

    case EOpAbsDifference:          out.debug << "absoluteDifference";    break;
    case EOpAddSaturate:            out.debug << "addSaturate";           break;
    case EOpSubSaturate:            out.debug << "subtractSaturate";      break;
    case EOpAverage:                out.debug << "average";               break;
    case EOpAverageRounded:         out.debug << "averageRounded";        break;
    case EOpMul32x16:               out.debug << "multiply32x16";         break;

    default: out.debug << "<unknown op>";
    }

    out.debug << " (" << node->getCompleteString() << ")";

    out.debug << "\n";

    return true;
}

bool TOutputTraverser::visitUnary(TVisit /* visit */, TIntermUnary* node)
{
    TInfoSink& out = infoSink;

    OutputTreeText(out, node, depth);

    switch (node->getOp()) {
    case EOpNegative:       out.debug << "Negate value";         break;
    case EOpVectorLogicalNot:
    case EOpLogicalNot:     out.debug << "Negate conditional";   break;
    case EOpBitwiseNot:     out.debug << "Bitwise not";          break;

    case EOpPostIncrement:  out.debug << "Post-Increment";       break;
    case EOpPostDecrement:  out.debug << "Post-Decrement";       break;
    case EOpPreIncrement:   out.debug << "Pre-Increment";        break;
    case EOpPreDecrement:   out.debug << "Pre-Decrement";        break;
    case EOpCopyObject:     out.debug << "copy object";          break;

    // * -> bool
    case EOpColwInt8ToBool:    out.debug << "Colwert int8_t to bool";  break;
    case EOpColwUint8ToBool:   out.debug << "Colwert uint8_t to bool"; break;
    case EOpColwInt16ToBool:   out.debug << "Colwert int16_t to bool"; break;
    case EOpColwUint16ToBool:  out.debug << "Colwert uint16_t to bool";break;
    case EOpColwIntToBool:     out.debug << "Colwert int to bool";     break;
    case EOpColwUintToBool:    out.debug << "Colwert uint to bool";    break;
    case EOpColwInt64ToBool:   out.debug << "Colwert int64 to bool";   break;
    case EOpColwUint64ToBool:  out.debug << "Colwert uint64 to bool";  break;
    case EOpColwFloat16ToBool: out.debug << "Colwert float16_t to bool";   break;
    case EOpColwFloatToBool:   out.debug << "Colwert float to bool";   break;
    case EOpColwDoubleToBool:  out.debug << "Colwert double to bool";  break;

    // bool -> *
    case EOpColwBoolToInt8:    out.debug << "Colwert bool to int8_t";  break;
    case EOpColwBoolToUint8:   out.debug << "Colwert bool to uint8_t"; break;
    case EOpColwBoolToInt16:   out.debug << "Colwert bool to in16t_t"; break;
    case EOpColwBoolToUint16:  out.debug << "Colwert bool to uint16_t";break;
    case EOpColwBoolToInt:     out.debug << "Colwert bool to int"  ;   break;
    case EOpColwBoolToUint:    out.debug << "Colwert bool to uint";    break;
    case EOpColwBoolToInt64:   out.debug << "Colwert bool to int64"; break;
    case EOpColwBoolToUint64:  out.debug << "Colwert bool to uint64";break;
    case EOpColwBoolToFloat16: out.debug << "Colwert bool to float16_t";   break;
    case EOpColwBoolToFloat:   out.debug << "Colwert bool to float";   break;
    case EOpColwBoolToDouble:  out.debug << "Colwert bool to double";   break;

    // int8_t -> (u)int*
    case EOpColwInt8ToInt16:   out.debug << "Colwert int8_t to int16_t";break;
    case EOpColwInt8ToInt:     out.debug << "Colwert int8_t to int";    break;
    case EOpColwInt8ToInt64:   out.debug << "Colwert int8_t to int64";   break;
    case EOpColwInt8ToUint8:   out.debug << "Colwert int8_t to uint8_t";break;
    case EOpColwInt8ToUint16:  out.debug << "Colwert int8_t to uint16_t";break;
    case EOpColwInt8ToUint:    out.debug << "Colwert int8_t to uint";    break;
    case EOpColwInt8ToUint64:  out.debug << "Colwert int8_t to uint64";   break;

    // uint8_t -> (u)int*
    case EOpColwUint8ToInt8:    out.debug << "Colwert uint8_t to int8_t";break;
    case EOpColwUint8ToInt16:   out.debug << "Colwert uint8_t to int16_t";break;
    case EOpColwUint8ToInt:     out.debug << "Colwert uint8_t to int";    break;
    case EOpColwUint8ToInt64:   out.debug << "Colwert uint8_t to int64";   break;
    case EOpColwUint8ToUint16:  out.debug << "Colwert uint8_t to uint16_t";break;
    case EOpColwUint8ToUint:    out.debug << "Colwert uint8_t to uint";    break;
    case EOpColwUint8ToUint64:  out.debug << "Colwert uint8_t to uint64";   break;

    // int8_t -> float*
    case EOpColwInt8ToFloat16:  out.debug << "Colwert int8_t to float16_t";break;
    case EOpColwInt8ToFloat:    out.debug << "Colwert int8_t to float";    break;
    case EOpColwInt8ToDouble:   out.debug << "Colwert int8_t to double";   break;

    // uint8_t -> float*
    case EOpColwUint8ToFloat16: out.debug << "Colwert uint8_t to float16_t";break;
    case EOpColwUint8ToFloat:   out.debug << "Colwert uint8_t to float";    break;
    case EOpColwUint8ToDouble:  out.debug << "Colwert uint8_t to double";   break;

    // int16_t -> (u)int*
    case EOpColwInt16ToInt8:    out.debug << "Colwert int16_t to int8_t";break;
    case EOpColwInt16ToInt:     out.debug << "Colwert int16_t to int";    break;
    case EOpColwInt16ToInt64:   out.debug << "Colwert int16_t to int64";   break;
    case EOpColwInt16ToUint8:   out.debug << "Colwert int16_t to uint8_t";break;
    case EOpColwInt16ToUint16:  out.debug << "Colwert int16_t to uint16_t";break;
    case EOpColwInt16ToUint:    out.debug << "Colwert int16_t to uint";    break;
    case EOpColwInt16ToUint64:  out.debug << "Colwert int16_t to uint64";   break;

    // int16_t -> float*
    case EOpColwInt16ToFloat16:  out.debug << "Colwert int16_t to float16_t";break;
    case EOpColwInt16ToFloat:    out.debug << "Colwert int16_t to float";    break;
    case EOpColwInt16ToDouble:   out.debug << "Colwert int16_t to double";   break;

    // uint16_t -> (u)int*
    case EOpColwUint16ToInt8:    out.debug << "Colwert uint16_t to int8_t";break;
    case EOpColwUint16ToInt16:   out.debug << "Colwert uint16_t to int16_t";break;
    case EOpColwUint16ToInt:     out.debug << "Colwert uint16_t to int";    break;
    case EOpColwUint16ToInt64:   out.debug << "Colwert uint16_t to int64";   break;
    case EOpColwUint16ToUint8:   out.debug << "Colwert uint16_t to uint8_t";break;
    case EOpColwUint16ToUint:    out.debug << "Colwert uint16_t to uint";    break;
    case EOpColwUint16ToUint64:  out.debug << "Colwert uint16_t to uint64";   break;

    // uint16_t -> float*
    case EOpColwUint16ToFloat16: out.debug << "Colwert uint16_t to float16_t";break;
    case EOpColwUint16ToFloat:   out.debug << "Colwert uint16_t to float";    break;
    case EOpColwUint16ToDouble:  out.debug << "Colwert uint16_t to double";   break;

    // int32_t -> (u)int*
    case EOpColwIntToInt8:    out.debug << "Colwert int to int8_t";break;
    case EOpColwIntToInt16:   out.debug << "Colwert int to int16_t";break;
    case EOpColwIntToInt64:   out.debug << "Colwert int to int64";   break;
    case EOpColwIntToUint8:   out.debug << "Colwert int to uint8_t";break;
    case EOpColwIntToUint16:  out.debug << "Colwert int to uint16_t";break;
    case EOpColwIntToUint:    out.debug << "Colwert int to uint";    break;
    case EOpColwIntToUint64:  out.debug << "Colwert int to uint64";   break;

    // int32_t -> float*
    case EOpColwIntToFloat16:  out.debug << "Colwert int to float16_t";break;
    case EOpColwIntToFloat:    out.debug << "Colwert int to float";    break;
    case EOpColwIntToDouble:   out.debug << "Colwert int to double";   break;

    // uint32_t -> (u)int*
    case EOpColwUintToInt8:    out.debug << "Colwert uint to int8_t";break;
    case EOpColwUintToInt16:   out.debug << "Colwert uint to int16_t";break;
    case EOpColwUintToInt:     out.debug << "Colwert uint to int";break;
    case EOpColwUintToInt64:   out.debug << "Colwert uint to int64";   break;
    case EOpColwUintToUint8:   out.debug << "Colwert uint to uint8_t";break;
    case EOpColwUintToUint16:  out.debug << "Colwert uint to uint16_t";break;
    case EOpColwUintToUint64:  out.debug << "Colwert uint to uint64";   break;

    // uint32_t -> float*
    case EOpColwUintToFloat16: out.debug << "Colwert uint to float16_t";break;
    case EOpColwUintToFloat:   out.debug << "Colwert uint to float";    break;
    case EOpColwUintToDouble:  out.debug << "Colwert uint to double";   break;

    // int64 -> (u)int*
    case EOpColwInt64ToInt8:    out.debug << "Colwert int64 to int8_t";  break;
    case EOpColwInt64ToInt16:   out.debug << "Colwert int64 to int16_t"; break;
    case EOpColwInt64ToInt:     out.debug << "Colwert int64 to int";   break;
    case EOpColwInt64ToUint8:   out.debug << "Colwert int64 to uint8_t";break;
    case EOpColwInt64ToUint16:  out.debug << "Colwert int64 to uint16_t";break;
    case EOpColwInt64ToUint:    out.debug << "Colwert int64 to uint";    break;
    case EOpColwInt64ToUint64:  out.debug << "Colwert int64 to uint64";   break;

     // int64 -> float*
    case EOpColwInt64ToFloat16:  out.debug << "Colwert int64 to float16_t";break;
    case EOpColwInt64ToFloat:    out.debug << "Colwert int64 to float";    break;
    case EOpColwInt64ToDouble:   out.debug << "Colwert int64 to double";   break;

    // uint64 -> (u)int*
    case EOpColwUint64ToInt8:    out.debug << "Colwert uint64 to int8_t";break;
    case EOpColwUint64ToInt16:   out.debug << "Colwert uint64 to int16_t";break;
    case EOpColwUint64ToInt:     out.debug << "Colwert uint64 to int";    break;
    case EOpColwUint64ToInt64:   out.debug << "Colwert uint64 to int64";   break;
    case EOpColwUint64ToUint8:   out.debug << "Colwert uint64 to uint8_t";break;
    case EOpColwUint64ToUint16:  out.debug << "Colwert uint64 to uint16";    break;
    case EOpColwUint64ToUint:    out.debug << "Colwert uint64 to uint";   break;

    // uint64 -> float*
    case EOpColwUint64ToFloat16: out.debug << "Colwert uint64 to float16_t";break;
    case EOpColwUint64ToFloat:   out.debug << "Colwert uint64 to float";    break;
    case EOpColwUint64ToDouble:  out.debug << "Colwert uint64 to double";   break;

    // float16_t -> int*
    case EOpColwFloat16ToInt8:  out.debug << "Colwert float16_t to int8_t"; break;
    case EOpColwFloat16ToInt16: out.debug << "Colwert float16_t to int16_t"; break;
    case EOpColwFloat16ToInt:   out.debug << "Colwert float16_t to int"; break;
    case EOpColwFloat16ToInt64: out.debug << "Colwert float16_t to int64"; break;

    // float16_t -> uint*
    case EOpColwFloat16ToUint8:  out.debug << "Colwert float16_t to uint8_t"; break;
    case EOpColwFloat16ToUint16: out.debug << "Colwert float16_t to uint16_t"; break;
    case EOpColwFloat16ToUint:   out.debug << "Colwert float16_t to uint"; break;
    case EOpColwFloat16ToUint64: out.debug << "Colwert float16_t to uint64"; break;

    // float16_t -> float*
    case EOpColwFloat16ToFloat:  out.debug << "Colwert float16_t to float"; break;
    case EOpColwFloat16ToDouble: out.debug << "Colwert float16_t to double"; break;

    // float32 -> float*
    case EOpColwFloatToFloat16: out.debug << "Colwert float to float16_t"; break;
    case EOpColwFloatToDouble:  out.debug << "Colwert float to double"; break;

    // float32_t -> int*
    case EOpColwFloatToInt8:  out.debug << "Colwert float to int8_t"; break;
    case EOpColwFloatToInt16: out.debug << "Colwert float to int16_t"; break;
    case EOpColwFloatToInt:   out.debug << "Colwert float to int"; break;
    case EOpColwFloatToInt64: out.debug << "Colwert float to int64"; break;

    // float32_t -> uint*
    case EOpColwFloatToUint8:  out.debug << "Colwert float to uint8_t"; break;
    case EOpColwFloatToUint16: out.debug << "Colwert float to uint16_t"; break;
    case EOpColwFloatToUint:   out.debug << "Colwert float to uint"; break;
    case EOpColwFloatToUint64: out.debug << "Colwert float to uint64"; break;

    // double -> float*
    case EOpColwDoubleToFloat16: out.debug << "Colwert double to float16_t"; break;
    case EOpColwDoubleToFloat:   out.debug << "Colwert double to float"; break;

    // double -> int*
    case EOpColwDoubleToInt8:  out.debug << "Colwert double to int8_t"; break;
    case EOpColwDoubleToInt16: out.debug << "Colwert double to int16_t"; break;
    case EOpColwDoubleToInt:   out.debug << "Colwert double to int"; break;
    case EOpColwDoubleToInt64: out.debug << "Colwert double to int64"; break;

    // float32_t -> uint*
    case EOpColwDoubleToUint8:  out.debug << "Colwert double to uint8_t"; break;
    case EOpColwDoubleToUint16: out.debug << "Colwert double to uint16_t"; break;
    case EOpColwDoubleToUint:   out.debug << "Colwert double to uint"; break;
    case EOpColwDoubleToUint64: out.debug << "Colwert double to uint64"; break;

    case EOpColwUint64ToPtr:  out.debug << "Colwert uint64_t to pointer";   break;
    case EOpColwPtrToUint64:  out.debug << "Colwert pointer to uint64_t";   break;

    case EOpRadians:        out.debug << "radians";              break;
    case EOpDegrees:        out.debug << "degrees";              break;
    case EOpSin:            out.debug << "sine";                 break;
    case EOpCos:            out.debug << "cosine";               break;
    case EOpTan:            out.debug << "tangent";              break;
    case EOpAsin:           out.debug << "arc sine";             break;
    case EOpAcos:           out.debug << "arc cosine";           break;
    case EOpAtan:           out.debug << "arc tangent";          break;
    case EOpSinh:           out.debug << "hyp. sine";            break;
    case EOpCosh:           out.debug << "hyp. cosine";          break;
    case EOpTanh:           out.debug << "hyp. tangent";         break;
    case EOpAsinh:          out.debug << "arc hyp. sine";        break;
    case EOpAcosh:          out.debug << "arc hyp. cosine";      break;
    case EOpAtanh:          out.debug << "arc hyp. tangent";     break;

    case EOpExp:            out.debug << "exp";                  break;
    case EOpLog:            out.debug << "log";                  break;
    case EOpExp2:           out.debug << "exp2";                 break;
    case EOpLog2:           out.debug << "log2";                 break;
    case EOpSqrt:           out.debug << "sqrt";                 break;
    case EOpIlwerseSqrt:    out.debug << "ilwerse sqrt";         break;

    case EOpAbs:            out.debug << "Absolute value";       break;
    case EOpSign:           out.debug << "Sign";                 break;
    case EOpFloor:          out.debug << "Floor";                break;
    case EOpTrunc:          out.debug << "trunc";                break;
    case EOpRound:          out.debug << "round";                break;
    case EOpRoundEven:      out.debug << "roundEven";            break;
    case EOpCeil:           out.debug << "Ceiling";              break;
    case EOpFract:          out.debug << "Fraction";             break;

    case EOpIsNan:          out.debug << "isnan";                break;
    case EOpIsInf:          out.debug << "isinf";                break;

    case EOpFloatBitsToInt: out.debug << "floatBitsToInt";       break;
    case EOpFloatBitsToUint:out.debug << "floatBitsToUint";      break;
    case EOpIntBitsToFloat: out.debug << "intBitsToFloat";       break;
    case EOpUintBitsToFloat:out.debug << "uintBitsToFloat";      break;
    case EOpDoubleBitsToInt64:  out.debug << "doubleBitsToInt64";  break;
    case EOpDoubleBitsToUint64: out.debug << "doubleBitsToUint64"; break;
    case EOpInt64BitsToDouble:  out.debug << "int64BitsToDouble";  break;
    case EOpUint64BitsToDouble: out.debug << "uint64BitsToDouble"; break;
    case EOpFloat16BitsToInt16:  out.debug << "float16BitsToInt16";  break;
    case EOpFloat16BitsToUint16: out.debug << "float16BitsToUint16"; break;
    case EOpInt16BitsToFloat16:  out.debug << "int16BitsToFloat16";  break;
    case EOpUint16BitsToFloat16: out.debug << "uint16BitsToFloat16"; break;

    case EOpPackSnorm2x16:  out.debug << "packSnorm2x16";        break;
    case EOpUnpackSnorm2x16:out.debug << "unpackSnorm2x16";      break;
    case EOpPackUnorm2x16:  out.debug << "packUnorm2x16";        break;
    case EOpUnpackUnorm2x16:out.debug << "unpackUnorm2x16";      break;
    case EOpPackHalf2x16:   out.debug << "packHalf2x16";         break;
    case EOpUnpackHalf2x16: out.debug << "unpackHalf2x16";       break;
    case EOpPack16:           out.debug << "pack16";                 break;
    case EOpPack32:           out.debug << "pack32";                 break;
    case EOpPack64:           out.debug << "pack64";                 break;
    case EOpUnpack32:         out.debug << "unpack32";               break;
    case EOpUnpack16:         out.debug << "unpack16";               break;
    case EOpUnpack8:          out.debug << "unpack8";               break;

    case EOpPackSnorm4x8:     out.debug << "PackSnorm4x8";       break;
    case EOpUnpackSnorm4x8:   out.debug << "UnpackSnorm4x8";     break;
    case EOpPackUnorm4x8:     out.debug << "PackUnorm4x8";       break;
    case EOpUnpackUnorm4x8:   out.debug << "UnpackUnorm4x8";     break;
    case EOpPackDouble2x32:   out.debug << "PackDouble2x32";     break;
    case EOpUnpackDouble2x32: out.debug << "UnpackDouble2x32";   break;

    case EOpPackInt2x32:      out.debug << "packInt2x32";        break;
    case EOpUnpackInt2x32:    out.debug << "unpackInt2x32";      break;
    case EOpPackUint2x32:     out.debug << "packUint2x32";       break;
    case EOpUnpackUint2x32:   out.debug << "unpackUint2x32";     break;

    case EOpPackInt2x16:      out.debug << "packInt2x16";        break;
    case EOpUnpackInt2x16:    out.debug << "unpackInt2x16";      break;
    case EOpPackUint2x16:     out.debug << "packUint2x16";       break;
    case EOpUnpackUint2x16:   out.debug << "unpackUint2x16";     break;

    case EOpPackInt4x16:      out.debug << "packInt4x16";        break;
    case EOpUnpackInt4x16:    out.debug << "unpackInt4x16";      break;
    case EOpPackUint4x16:     out.debug << "packUint4x16";       break;
    case EOpUnpackUint4x16:   out.debug << "unpackUint4x16";     break;
    case EOpPackFloat2x16:    out.debug << "packFloat2x16";      break;
    case EOpUnpackFloat2x16:  out.debug << "unpackFloat2x16";    break;

    case EOpLength:         out.debug << "length";               break;
    case EOpNormalize:      out.debug << "normalize";            break;
    case EOpDPdx:           out.debug << "dPdx";                 break;
    case EOpDPdy:           out.debug << "dPdy";                 break;
    case EOpFwidth:         out.debug << "fwidth";               break;
    case EOpDPdxFine:       out.debug << "dPdxFine";             break;
    case EOpDPdyFine:       out.debug << "dPdyFine";             break;
    case EOpFwidthFine:     out.debug << "fwidthFine";           break;
    case EOpDPdxCoarse:     out.debug << "dPdxCoarse";           break;
    case EOpDPdyCoarse:     out.debug << "dPdyCoarse";           break;
    case EOpFwidthCoarse:   out.debug << "fwidthCoarse";         break;

    case EOpInterpolateAtCentroid: out.debug << "interpolateAtCentroid";  break;

    case EOpDeterminant:    out.debug << "determinant";          break;
    case EOpMatrixIlwerse:  out.debug << "ilwerse";              break;
    case EOpTranspose:      out.debug << "transpose";            break;

    case EOpAny:            out.debug << "any";                  break;
    case EOpAll:            out.debug << "all";                  break;

    case EOpArrayLength:    out.debug << "array length";         break;

    case EOpEmitStreamVertex:   out.debug << "EmitStreamVertex";   break;
    case EOpEndStreamPrimitive: out.debug << "EndStreamPrimitive"; break;

    case EOpAtomicCounterIncrement: out.debug << "AtomicCounterIncrement";break;
    case EOpAtomicCounterDecrement: out.debug << "AtomicCounterDecrement";break;
    case EOpAtomicCounter:          out.debug << "AtomicCounter";         break;

    case EOpTextureQuerySize:       out.debug << "textureSize";           break;
    case EOpTextureQueryLod:        out.debug << "textureQueryLod";       break;
    case EOpTextureQueryLevels:     out.debug << "textureQueryLevels";    break;
    case EOpTextureQuerySamples:    out.debug << "textureSamples";        break;
    case EOpImageQuerySize:         out.debug << "imageQuerySize";        break;
    case EOpImageQuerySamples:      out.debug << "imageQuerySamples";     break;
    case EOpImageLoad:              out.debug << "imageLoad";             break;

    case EOpBitFieldReverse:        out.debug << "bitFieldReverse";       break;
    case EOpBitCount:               out.debug << "bitCount";              break;
    case EOpFindLSB:                out.debug << "findLSB";               break;
    case EOpFindMSB:                out.debug << "findMSB";               break;

    case EOpCountLeadingZeros:      out.debug << "countLeadingZeros";     break;
    case EOpCountTrailingZeros:     out.debug << "countTrailingZeros";    break;

    case EOpNoise:                  out.debug << "noise";                 break;

    case EOpBallot:                 out.debug << "ballot";                break;
    case EOpReadFirstIlwocation:    out.debug << "readFirstIlwocation";   break;

    case EOpAnyIlwocation:          out.debug << "anyIlwocation";         break;
    case EOpAllIlwocations:         out.debug << "allIlwocations";        break;
    case EOpAllIlwocationsEqual:    out.debug << "allIlwocationsEqual";   break;

    case EOpSubgroupElect:                   out.debug << "subgroupElect";                   break;
    case EOpSubgroupAll:                     out.debug << "subgroupAll";                     break;
    case EOpSubgroupAny:                     out.debug << "subgroupAny";                     break;
    case EOpSubgroupAllEqual:                out.debug << "subgroupAllEqual";                break;
    case EOpSubgroupBroadcast:               out.debug << "subgroupBroadcast";               break;
    case EOpSubgroupBroadcastFirst:          out.debug << "subgroupBroadcastFirst";          break;
    case EOpSubgroupBallot:                  out.debug << "subgroupBallot";                  break;
    case EOpSubgroupIlwerseBallot:           out.debug << "subgroupIlwerseBallot";           break;
    case EOpSubgroupBallotBitExtract:        out.debug << "subgroupBallotBitExtract";        break;
    case EOpSubgroupBallotBitCount:          out.debug << "subgroupBallotBitCount";          break;
    case EOpSubgroupBallotInclusiveBitCount: out.debug << "subgroupBallotInclusiveBitCount"; break;
    case EOpSubgroupBallotExclusiveBitCount: out.debug << "subgroupBallotExclusiveBitCount"; break;
    case EOpSubgroupBallotFindLSB:           out.debug << "subgroupBallotFindLSB";           break;
    case EOpSubgroupBallotFindMSB:           out.debug << "subgroupBallotFindMSB";           break;
    case EOpSubgroupShuffle:                 out.debug << "subgroupShuffle";                 break;
    case EOpSubgroupShuffleXor:              out.debug << "subgroupShuffleXor";              break;
    case EOpSubgroupShuffleUp:               out.debug << "subgroupShuffleUp";               break;
    case EOpSubgroupShuffleDown:             out.debug << "subgroupShuffleDown";             break;
    case EOpSubgroupAdd:                     out.debug << "subgroupAdd";                     break;
    case EOpSubgroupMul:                     out.debug << "subgroupMul";                     break;
    case EOpSubgroupMin:                     out.debug << "subgroupMin";                     break;
    case EOpSubgroupMax:                     out.debug << "subgroupMax";                     break;
    case EOpSubgroupAnd:                     out.debug << "subgroupAnd";                     break;
    case EOpSubgroupOr:                      out.debug << "subgroupOr";                      break;
    case EOpSubgroupXor:                     out.debug << "subgroupXor";                     break;
    case EOpSubgroupInclusiveAdd:            out.debug << "subgroupInclusiveAdd";            break;
    case EOpSubgroupInclusiveMul:            out.debug << "subgroupInclusiveMul";            break;
    case EOpSubgroupInclusiveMin:            out.debug << "subgroupInclusiveMin";            break;
    case EOpSubgroupInclusiveMax:            out.debug << "subgroupInclusiveMax";            break;
    case EOpSubgroupInclusiveAnd:            out.debug << "subgroupInclusiveAnd";            break;
    case EOpSubgroupInclusiveOr:             out.debug << "subgroupInclusiveOr";             break;
    case EOpSubgroupInclusiveXor:            out.debug << "subgroupInclusiveXor";            break;
    case EOpSubgroupExclusiveAdd:            out.debug << "subgroupExclusiveAdd";            break;
    case EOpSubgroupExclusiveMul:            out.debug << "subgroupExclusiveMul";            break;
    case EOpSubgroupExclusiveMin:            out.debug << "subgroupExclusiveMin";            break;
    case EOpSubgroupExclusiveMax:            out.debug << "subgroupExclusiveMax";            break;
    case EOpSubgroupExclusiveAnd:            out.debug << "subgroupExclusiveAnd";            break;
    case EOpSubgroupExclusiveOr:             out.debug << "subgroupExclusiveOr";             break;
    case EOpSubgroupExclusiveXor:            out.debug << "subgroupExclusiveXor";            break;
    case EOpSubgroupClusteredAdd:            out.debug << "subgroupClusteredAdd";            break;
    case EOpSubgroupClusteredMul:            out.debug << "subgroupClusteredMul";            break;
    case EOpSubgroupClusteredMin:            out.debug << "subgroupClusteredMin";            break;
    case EOpSubgroupClusteredMax:            out.debug << "subgroupClusteredMax";            break;
    case EOpSubgroupClusteredAnd:            out.debug << "subgroupClusteredAnd";            break;
    case EOpSubgroupClusteredOr:             out.debug << "subgroupClusteredOr";             break;
    case EOpSubgroupClusteredXor:            out.debug << "subgroupClusteredXor";            break;
    case EOpSubgroupQuadBroadcast:           out.debug << "subgroupQuadBroadcast";           break;
    case EOpSubgroupQuadSwapHorizontal:      out.debug << "subgroupQuadSwapHorizontal";      break;
    case EOpSubgroupQuadSwapVertical:        out.debug << "subgroupQuadSwapVertical";        break;
    case EOpSubgroupQuadSwapDiagonal:        out.debug << "subgroupQuadSwapDiagonal";        break;

    case EOpSubgroupPartition:                          out.debug << "subgroupPartitionLW";                          break;
    case EOpSubgroupPartitionedAdd:                     out.debug << "subgroupPartitionedAddLW";                     break;
    case EOpSubgroupPartitionedMul:                     out.debug << "subgroupPartitionedMulLW";                     break;
    case EOpSubgroupPartitionedMin:                     out.debug << "subgroupPartitionedMinLW";                     break;
    case EOpSubgroupPartitionedMax:                     out.debug << "subgroupPartitionedMaxLW";                     break;
    case EOpSubgroupPartitionedAnd:                     out.debug << "subgroupPartitionedAndLW";                     break;
    case EOpSubgroupPartitionedOr:                      out.debug << "subgroupPartitionedOrLW";                      break;
    case EOpSubgroupPartitionedXor:                     out.debug << "subgroupPartitionedXorLW";                     break;
    case EOpSubgroupPartitionedInclusiveAdd:            out.debug << "subgroupPartitionedInclusiveAddLW";            break;
    case EOpSubgroupPartitionedInclusiveMul:            out.debug << "subgroupPartitionedInclusiveMulLW";            break;
    case EOpSubgroupPartitionedInclusiveMin:            out.debug << "subgroupPartitionedInclusiveMinLW";            break;
    case EOpSubgroupPartitionedInclusiveMax:            out.debug << "subgroupPartitionedInclusiveMaxLW";            break;
    case EOpSubgroupPartitionedInclusiveAnd:            out.debug << "subgroupPartitionedInclusiveAndLW";            break;
    case EOpSubgroupPartitionedInclusiveOr:             out.debug << "subgroupPartitionedInclusiveOrLW";             break;
    case EOpSubgroupPartitionedInclusiveXor:            out.debug << "subgroupPartitionedInclusiveXorLW";            break;
    case EOpSubgroupPartitionedExclusiveAdd:            out.debug << "subgroupPartitionedExclusiveAddLW";            break;
    case EOpSubgroupPartitionedExclusiveMul:            out.debug << "subgroupPartitionedExclusiveMulLW";            break;
    case EOpSubgroupPartitionedExclusiveMin:            out.debug << "subgroupPartitionedExclusiveMinLW";            break;
    case EOpSubgroupPartitionedExclusiveMax:            out.debug << "subgroupPartitionedExclusiveMaxLW";            break;
    case EOpSubgroupPartitionedExclusiveAnd:            out.debug << "subgroupPartitionedExclusiveAndLW";            break;
    case EOpSubgroupPartitionedExclusiveOr:             out.debug << "subgroupPartitionedExclusiveOrLW";             break;
    case EOpSubgroupPartitionedExclusiveXor:            out.debug << "subgroupPartitionedExclusiveXorLW";            break;

    case EOpClip:                   out.debug << "clip";                  break;
    case EOpIsFinite:               out.debug << "isfinite";              break;
    case EOpLog10:                  out.debug << "log10";                 break;
    case EOpRcp:                    out.debug << "rcp";                   break;
    case EOpSaturate:               out.debug << "saturate";              break;

    case EOpSparseTexelsResident:   out.debug << "sparseTexelsResident";  break;

    case EOpMinIlwocations:             out.debug << "minIlwocations";              break;
    case EOpMaxIlwocations:             out.debug << "maxIlwocations";              break;
    case EOpAddIlwocations:             out.debug << "addIlwocations";              break;
    case EOpMinIlwocationsNonUniform:   out.debug << "minIlwocationsNonUniform";    break;
    case EOpMaxIlwocationsNonUniform:   out.debug << "maxIlwocationsNonUniform";    break;
    case EOpAddIlwocationsNonUniform:   out.debug << "addIlwocationsNonUniform";    break;

    case EOpMinIlwocationsInclusiveScan:            out.debug << "minIlwocationsInclusiveScan";             break;
    case EOpMaxIlwocationsInclusiveScan:            out.debug << "maxIlwocationsInclusiveScan";             break;
    case EOpAddIlwocationsInclusiveScan:            out.debug << "addIlwocationsInclusiveScan";             break;
    case EOpMinIlwocationsInclusiveScanNonUniform:  out.debug << "minIlwocationsInclusiveScanNonUniform";   break;
    case EOpMaxIlwocationsInclusiveScanNonUniform:  out.debug << "maxIlwocationsInclusiveScanNonUniform";   break;
    case EOpAddIlwocationsInclusiveScanNonUniform:  out.debug << "addIlwocationsInclusiveScanNonUniform";   break;

    case EOpMinIlwocationsExclusiveScan:            out.debug << "minIlwocationsExclusiveScan";             break;
    case EOpMaxIlwocationsExclusiveScan:            out.debug << "maxIlwocationsExclusiveScan";             break;
    case EOpAddIlwocationsExclusiveScan:            out.debug << "addIlwocationsExclusiveScan";             break;
    case EOpMinIlwocationsExclusiveScanNonUniform:  out.debug << "minIlwocationsExclusiveScanNonUniform";   break;
    case EOpMaxIlwocationsExclusiveScanNonUniform:  out.debug << "maxIlwocationsExclusiveScanNonUniform";   break;
    case EOpAddIlwocationsExclusiveScanNonUniform:  out.debug << "addIlwocationsExclusiveScanNonUniform";   break;

    case EOpMbcnt:                  out.debug << "mbcnt";                       break;

    case EOpFragmentMaskFetch:      out.debug << "fragmentMaskFetchAMD";        break;
    case EOpFragmentFetch:          out.debug << "fragmentFetchAMD";            break;

    case EOpLwbeFaceIndex:          out.debug << "lwbeFaceIndex";               break;
    case EOpLwbeFaceCoord:          out.debug << "lwbeFaceCoord";               break;

    case EOpSubpassLoad:   out.debug << "subpassLoad";   break;
    case EOpSubpassLoadMS: out.debug << "subpassLoadMS"; break;

    case EOpConstructReference: out.debug << "Construct reference type"; break;

    default: out.debug.message(EPrefixError, "Bad unary op");
    }

    out.debug << " (" << node->getCompleteString() << ")";

    out.debug << "\n";

    return true;
}

bool TOutputTraverser::visitAggregate(TVisit /* visit */, TIntermAggregate* node)
{
    TInfoSink& out = infoSink;

    if (node->getOp() == EOpNull) {
        out.debug.message(EPrefixError, "node is still EOpNull!");
        return true;
    }

    OutputTreeText(out, node, depth);

    switch (node->getOp()) {
    case EOpSequence:      out.debug << "Sequence\n";       return true;
    case EOpLinkerObjects: out.debug << "Linker Objects\n"; return true;
    case EOpComma:         out.debug << "Comma";            break;
    case EOpFunction:      out.debug << "Function Definition: " << node->getName(); break;
    case EOpFunctionCall:  out.debug << "Function Call: "       << node->getName(); break;
    case EOpParameters:    out.debug << "Function Parameters: ";                    break;

    case EOpConstructFloat: out.debug << "Construct float"; break;
    case EOpConstructDouble:out.debug << "Construct double"; break;

    case EOpConstructVec2:  out.debug << "Construct vec2";  break;
    case EOpConstructVec3:  out.debug << "Construct vec3";  break;
    case EOpConstructVec4:  out.debug << "Construct vec4";  break;
    case EOpConstructDVec2: out.debug << "Construct dvec2";  break;
    case EOpConstructDVec3: out.debug << "Construct dvec3";  break;
    case EOpConstructDVec4: out.debug << "Construct dvec4";  break;
    case EOpConstructBool:  out.debug << "Construct bool";  break;
    case EOpConstructBVec2: out.debug << "Construct bvec2"; break;
    case EOpConstructBVec3: out.debug << "Construct bvec3"; break;
    case EOpConstructBVec4: out.debug << "Construct bvec4"; break;
    case EOpConstructInt8:   out.debug << "Construct int8_t";   break;
    case EOpConstructI8Vec2: out.debug << "Construct i8vec2"; break;
    case EOpConstructI8Vec3: out.debug << "Construct i8vec3"; break;
    case EOpConstructI8Vec4: out.debug << "Construct i8vec4"; break;
    case EOpConstructInt:   out.debug << "Construct int";   break;
    case EOpConstructIVec2: out.debug << "Construct ivec2"; break;
    case EOpConstructIVec3: out.debug << "Construct ivec3"; break;
    case EOpConstructIVec4: out.debug << "Construct ivec4"; break;
    case EOpConstructUint8:    out.debug << "Construct uint8_t";    break;
    case EOpConstructU8Vec2:   out.debug << "Construct u8vec2";   break;
    case EOpConstructU8Vec3:   out.debug << "Construct u8vec3";   break;
    case EOpConstructU8Vec4:   out.debug << "Construct u8vec4";   break;
    case EOpConstructUint:    out.debug << "Construct uint";    break;
    case EOpConstructUVec2:   out.debug << "Construct uvec2";   break;
    case EOpConstructUVec3:   out.debug << "Construct uvec3";   break;
    case EOpConstructUVec4:   out.debug << "Construct uvec4";   break;
    case EOpConstructInt64:   out.debug << "Construct int64"; break;
    case EOpConstructI64Vec2: out.debug << "Construct i64vec2"; break;
    case EOpConstructI64Vec3: out.debug << "Construct i64vec3"; break;
    case EOpConstructI64Vec4: out.debug << "Construct i64vec4"; break;
    case EOpConstructUint64:  out.debug << "Construct uint64"; break;
    case EOpConstructU64Vec2: out.debug << "Construct u64vec2"; break;
    case EOpConstructU64Vec3: out.debug << "Construct u64vec3"; break;
    case EOpConstructU64Vec4: out.debug << "Construct u64vec4"; break;
    case EOpConstructInt16:   out.debug << "Construct int16_t"; break;
    case EOpConstructI16Vec2: out.debug << "Construct i16vec2"; break;
    case EOpConstructI16Vec3: out.debug << "Construct i16vec3"; break;
    case EOpConstructI16Vec4: out.debug << "Construct i16vec4"; break;
    case EOpConstructUint16:  out.debug << "Construct uint16_t"; break;
    case EOpConstructU16Vec2: out.debug << "Construct u16vec2"; break;
    case EOpConstructU16Vec3: out.debug << "Construct u16vec3"; break;
    case EOpConstructU16Vec4: out.debug << "Construct u16vec4"; break;
    case EOpConstructMat2x2:  out.debug << "Construct mat2";    break;
    case EOpConstructMat2x3:  out.debug << "Construct mat2x3";  break;
    case EOpConstructMat2x4:  out.debug << "Construct mat2x4";  break;
    case EOpConstructMat3x2:  out.debug << "Construct mat3x2";  break;
    case EOpConstructMat3x3:  out.debug << "Construct mat3";    break;
    case EOpConstructMat3x4:  out.debug << "Construct mat3x4";  break;
    case EOpConstructMat4x2:  out.debug << "Construct mat4x2";  break;
    case EOpConstructMat4x3:  out.debug << "Construct mat4x3";  break;
    case EOpConstructMat4x4:  out.debug << "Construct mat4";    break;
    case EOpConstructDMat2x2: out.debug << "Construct dmat2";   break;
    case EOpConstructDMat2x3: out.debug << "Construct dmat2x3"; break;
    case EOpConstructDMat2x4: out.debug << "Construct dmat2x4"; break;
    case EOpConstructDMat3x2: out.debug << "Construct dmat3x2"; break;
    case EOpConstructDMat3x3: out.debug << "Construct dmat3";   break;
    case EOpConstructDMat3x4: out.debug << "Construct dmat3x4"; break;
    case EOpConstructDMat4x2: out.debug << "Construct dmat4x2"; break;
    case EOpConstructDMat4x3: out.debug << "Construct dmat4x3"; break;
    case EOpConstructDMat4x4: out.debug << "Construct dmat4";   break;
    case EOpConstructIMat2x2: out.debug << "Construct imat2";   break;
    case EOpConstructIMat2x3: out.debug << "Construct imat2x3"; break;
    case EOpConstructIMat2x4: out.debug << "Construct imat2x4"; break;
    case EOpConstructIMat3x2: out.debug << "Construct imat3x2"; break;
    case EOpConstructIMat3x3: out.debug << "Construct imat3";   break;
    case EOpConstructIMat3x4: out.debug << "Construct imat3x4"; break;
    case EOpConstructIMat4x2: out.debug << "Construct imat4x2"; break;
    case EOpConstructIMat4x3: out.debug << "Construct imat4x3"; break;
    case EOpConstructIMat4x4: out.debug << "Construct imat4";   break;
    case EOpConstructUMat2x2: out.debug << "Construct umat2";   break;
    case EOpConstructUMat2x3: out.debug << "Construct umat2x3"; break;
    case EOpConstructUMat2x4: out.debug << "Construct umat2x4"; break;
    case EOpConstructUMat3x2: out.debug << "Construct umat3x2"; break;
    case EOpConstructUMat3x3: out.debug << "Construct umat3";   break;
    case EOpConstructUMat3x4: out.debug << "Construct umat3x4"; break;
    case EOpConstructUMat4x2: out.debug << "Construct umat4x2"; break;
    case EOpConstructUMat4x3: out.debug << "Construct umat4x3"; break;
    case EOpConstructUMat4x4: out.debug << "Construct umat4";   break;
    case EOpConstructBMat2x2: out.debug << "Construct bmat2";   break;
    case EOpConstructBMat2x3: out.debug << "Construct bmat2x3"; break;
    case EOpConstructBMat2x4: out.debug << "Construct bmat2x4"; break;
    case EOpConstructBMat3x2: out.debug << "Construct bmat3x2"; break;
    case EOpConstructBMat3x3: out.debug << "Construct bmat3";   break;
    case EOpConstructBMat3x4: out.debug << "Construct bmat3x4"; break;
    case EOpConstructBMat4x2: out.debug << "Construct bmat4x2"; break;
    case EOpConstructBMat4x3: out.debug << "Construct bmat4x3"; break;
    case EOpConstructBMat4x4: out.debug << "Construct bmat4";   break;
    case EOpConstructFloat16:   out.debug << "Construct float16_t"; break;
    case EOpConstructF16Vec2:   out.debug << "Construct f16vec2";   break;
    case EOpConstructF16Vec3:   out.debug << "Construct f16vec3";   break;
    case EOpConstructF16Vec4:   out.debug << "Construct f16vec4";   break;
    case EOpConstructF16Mat2x2: out.debug << "Construct f16mat2";   break;
    case EOpConstructF16Mat2x3: out.debug << "Construct f16mat2x3"; break;
    case EOpConstructF16Mat2x4: out.debug << "Construct f16mat2x4"; break;
    case EOpConstructF16Mat3x2: out.debug << "Construct f16mat3x2"; break;
    case EOpConstructF16Mat3x3: out.debug << "Construct f16mat3";   break;
    case EOpConstructF16Mat3x4: out.debug << "Construct f16mat3x4"; break;
    case EOpConstructF16Mat4x2: out.debug << "Construct f16mat4x2"; break;
    case EOpConstructF16Mat4x3: out.debug << "Construct f16mat4x3"; break;
    case EOpConstructF16Mat4x4: out.debug << "Construct f16mat4";   break;
    case EOpConstructStruct:  out.debug << "Construct structure";  break;
    case EOpConstructTextureSampler: out.debug << "Construct combined texture-sampler"; break;
    case EOpConstructReference:  out.debug << "Construct reference";  break;
    case EOpConstructCooperativeMatrix:  out.debug << "Construct cooperative matrix";  break;

    case EOpLessThan:         out.debug << "Compare Less Than";             break;
    case EOpGreaterThan:      out.debug << "Compare Greater Than";          break;
    case EOpLessThanEqual:    out.debug << "Compare Less Than or Equal";    break;
    case EOpGreaterThanEqual: out.debug << "Compare Greater Than or Equal"; break;
    case EOpVectorEqual:      out.debug << "Equal";                         break;
    case EOpVectorNotEqual:   out.debug << "NotEqual";                      break;

    case EOpMod:           out.debug << "mod";         break;
    case EOpModf:          out.debug << "modf";        break;
    case EOpPow:           out.debug << "pow";         break;

    case EOpAtan:          out.debug << "arc tangent"; break;

    case EOpMin:           out.debug << "min";         break;
    case EOpMax:           out.debug << "max";         break;
    case EOpClamp:         out.debug << "clamp";       break;
    case EOpMix:           out.debug << "mix";         break;
    case EOpStep:          out.debug << "step";        break;
    case EOpSmoothStep:    out.debug << "smoothstep";  break;

    case EOpDistance:      out.debug << "distance";                break;
    case EOpDot:           out.debug << "dot-product";             break;
    case EOpCross:         out.debug << "cross-product";           break;
    case EOpFaceForward:   out.debug << "face-forward";            break;
    case EOpReflect:       out.debug << "reflect";                 break;
    case EOpRefract:       out.debug << "refract";                 break;
    case EOpMul:           out.debug << "component-wise multiply"; break;
    case EOpOuterProduct:  out.debug << "outer product";           break;

    case EOpEmitVertex:    out.debug << "EmitVertex";              break;
    case EOpEndPrimitive:  out.debug << "EndPrimitive";            break;

    case EOpBarrier:                    out.debug << "Barrier";                    break;
    case EOpMemoryBarrier:              out.debug << "MemoryBarrier";              break;
    case EOpMemoryBarrierAtomicCounter: out.debug << "MemoryBarrierAtomicCounter"; break;
    case EOpMemoryBarrierBuffer:        out.debug << "MemoryBarrierBuffer";        break;
    case EOpMemoryBarrierImage:         out.debug << "MemoryBarrierImage";         break;
    case EOpMemoryBarrierShared:        out.debug << "MemoryBarrierShared";        break;
    case EOpGroupMemoryBarrier:         out.debug << "GroupMemoryBarrier";         break;

    case EOpReadIlwocation:             out.debug << "readIlwocation";        break;

    case EOpSwizzleIlwocations:         out.debug << "swizzleIlwocations";       break;
    case EOpSwizzleIlwocationsMasked:   out.debug << "swizzleIlwocationsMasked"; break;
    case EOpWriteIlwocation:            out.debug << "writeIlwocation";          break;

    case EOpMin3:                       out.debug << "min3";                  break;
    case EOpMax3:                       out.debug << "max3";                  break;
    case EOpMid3:                       out.debug << "mid3";                  break;
    case EOpTime:                       out.debug << "time";                  break;

    case EOpAtomicAdd:                  out.debug << "AtomicAdd";             break;
    case EOpAtomicMin:                  out.debug << "AtomicMin";             break;
    case EOpAtomicMax:                  out.debug << "AtomicMax";             break;
    case EOpAtomicAnd:                  out.debug << "AtomicAnd";             break;
    case EOpAtomicOr:                   out.debug << "AtomicOr";              break;
    case EOpAtomicXor:                  out.debug << "AtomicXor";             break;
    case EOpAtomicExchange:             out.debug << "AtomicExchange";        break;
    case EOpAtomicCompSwap:             out.debug << "AtomicCompSwap";        break;
    case EOpAtomicLoad:                 out.debug << "AtomicLoad";            break;
    case EOpAtomicStore:                out.debug << "AtomicStore";           break;

    case EOpAtomicCounterAdd:           out.debug << "AtomicCounterAdd";      break;
    case EOpAtomicCounterSubtract:      out.debug << "AtomicCounterSubtract"; break;
    case EOpAtomicCounterMin:           out.debug << "AtomicCounterMin";      break;
    case EOpAtomicCounterMax:           out.debug << "AtomicCounterMax";      break;
    case EOpAtomicCounterAnd:           out.debug << "AtomicCounterAnd";      break;
    case EOpAtomicCounterOr:            out.debug << "AtomicCounterOr";       break;
    case EOpAtomicCounterXor:           out.debug << "AtomicCounterXor";      break;
    case EOpAtomicCounterExchange:      out.debug << "AtomicCounterExchange"; break;
    case EOpAtomicCounterCompSwap:      out.debug << "AtomicCounterCompSwap"; break;

    case EOpImageQuerySize:             out.debug << "imageQuerySize";        break;
    case EOpImageQuerySamples:          out.debug << "imageQuerySamples";     break;
    case EOpImageLoad:                  out.debug << "imageLoad";             break;
    case EOpImageStore:                 out.debug << "imageStore";            break;
    case EOpImageAtomicAdd:             out.debug << "imageAtomicAdd";        break;
    case EOpImageAtomicMin:             out.debug << "imageAtomicMin";        break;
    case EOpImageAtomicMax:             out.debug << "imageAtomicMax";        break;
    case EOpImageAtomicAnd:             out.debug << "imageAtomicAnd";        break;
    case EOpImageAtomicOr:              out.debug << "imageAtomicOr";         break;
    case EOpImageAtomicXor:             out.debug << "imageAtomicXor";        break;
    case EOpImageAtomicExchange:        out.debug << "imageAtomicExchange";   break;
    case EOpImageAtomicCompSwap:        out.debug << "imageAtomicCompSwap";   break;
    case EOpImageAtomicLoad:            out.debug << "imageAtomicLoad";       break;
    case EOpImageAtomicStore:           out.debug << "imageAtomicStore";      break;
    case EOpImageLoadLod:               out.debug << "imageLoadLod";          break;
    case EOpImageStoreLod:              out.debug << "imageStoreLod";         break;

    case EOpTextureQuerySize:           out.debug << "textureSize";           break;
    case EOpTextureQueryLod:            out.debug << "textureQueryLod";       break;
    case EOpTextureQueryLevels:         out.debug << "textureQueryLevels";    break;
    case EOpTextureQuerySamples:        out.debug << "textureSamples";        break;
    case EOpTexture:                    out.debug << "texture";               break;
    case EOpTextureProj:                out.debug << "textureProj";           break;
    case EOpTextureLod:                 out.debug << "textureLod";            break;
    case EOpTextureOffset:              out.debug << "textureOffset";         break;
    case EOpTextureFetch:               out.debug << "textureFetch";          break;
    case EOpTextureFetchOffset:         out.debug << "textureFetchOffset";    break;
    case EOpTextureProjOffset:          out.debug << "textureProjOffset";     break;
    case EOpTextureLodOffset:           out.debug << "textureLodOffset";      break;
    case EOpTextureProjLod:             out.debug << "textureProjLod";        break;
    case EOpTextureProjLodOffset:       out.debug << "textureProjLodOffset";  break;
    case EOpTextureGrad:                out.debug << "textureGrad";           break;
    case EOpTextureGradOffset:          out.debug << "textureGradOffset";     break;
    case EOpTextureProjGrad:            out.debug << "textureProjGrad";       break;
    case EOpTextureProjGradOffset:      out.debug << "textureProjGradOffset"; break;
    case EOpTextureGather:              out.debug << "textureGather";         break;
    case EOpTextureGatherOffset:        out.debug << "textureGatherOffset";   break;
    case EOpTextureGatherOffsets:       out.debug << "textureGatherOffsets";  break;
    case EOpTextureClamp:               out.debug << "textureClamp";          break;
    case EOpTextureOffsetClamp:         out.debug << "textureOffsetClamp";    break;
    case EOpTextureGradClamp:           out.debug << "textureGradClamp";      break;
    case EOpTextureGradOffsetClamp:     out.debug << "textureGradOffsetClamp";  break;
    case EOpTextureGatherLod:           out.debug << "textureGatherLod";        break;
    case EOpTextureGatherLodOffset:     out.debug << "textureGatherLodOffset";  break;
    case EOpTextureGatherLodOffsets:    out.debug << "textureGatherLodOffsets"; break;

    case EOpSparseTexture:                  out.debug << "sparseTexture";                   break;
    case EOpSparseTextureOffset:            out.debug << "sparseTextureOffset";             break;
    case EOpSparseTextureLod:               out.debug << "sparseTextureLod";                break;
    case EOpSparseTextureLodOffset:         out.debug << "sparseTextureLodOffset";          break;
    case EOpSparseTextureFetch:             out.debug << "sparseTexelFetch";                break;
    case EOpSparseTextureFetchOffset:       out.debug << "sparseTexelFetchOffset";          break;
    case EOpSparseTextureGrad:              out.debug << "sparseTextureGrad";               break;
    case EOpSparseTextureGradOffset:        out.debug << "sparseTextureGradOffset";         break;
    case EOpSparseTextureGather:            out.debug << "sparseTextureGather";             break;
    case EOpSparseTextureGatherOffset:      out.debug << "sparseTextureGatherOffset";       break;
    case EOpSparseTextureGatherOffsets:     out.debug << "sparseTextureGatherOffsets";      break;
    case EOpSparseImageLoad:                out.debug << "sparseImageLoad";                 break;
    case EOpSparseTextureClamp:             out.debug << "sparseTextureClamp";              break;
    case EOpSparseTextureOffsetClamp:       out.debug << "sparseTextureOffsetClamp";        break;
    case EOpSparseTextureGradClamp:         out.debug << "sparseTextureGradClamp";          break;
    case EOpSparseTextureGradOffsetClamp:   out.debug << "sparseTextureGradOffsetClam";     break;
    case EOpSparseTextureGatherLod:         out.debug << "sparseTextureGatherLod";          break;
    case EOpSparseTextureGatherLodOffset:   out.debug << "sparseTextureGatherLodOffset";    break;
    case EOpSparseTextureGatherLodOffsets:  out.debug << "sparseTextureGatherLodOffsets";   break;
    case EOpSparseImageLoadLod:             out.debug << "sparseImageLoadLod";              break;
    case EOpImageSampleFootprintLW:             out.debug << "imageSampleFootprintLW";          break;
    case EOpImageSampleFootprintClampLW:        out.debug << "imageSampleFootprintClampLW";     break;
    case EOpImageSampleFootprintLodLW:          out.debug << "imageSampleFootprintLodLW";       break;
    case EOpImageSampleFootprintGradLW:         out.debug << "imageSampleFootprintGradLW";      break;
    case EOpImageSampleFootprintGradClampLW:    out.debug << "mageSampleFootprintGradClampLW";  break;
    case EOpAddCarry:                   out.debug << "addCarry";              break;
    case EOpSubBorrow:                  out.debug << "subBorrow";             break;
    case EOpUMulExtended:               out.debug << "uMulExtended";          break;
    case EOpIMulExtended:               out.debug << "iMulExtended";          break;
    case EOpBitfieldExtract:            out.debug << "bitfieldExtract";       break;
    case EOpBitfieldInsert:             out.debug << "bitfieldInsert";        break;

    case EOpFma:                        out.debug << "fma";                   break;
    case EOpFrexp:                      out.debug << "frexp";                 break;
    case EOpLdexp:                      out.debug << "ldexp";                 break;

    case EOpInterpolateAtSample:   out.debug << "interpolateAtSample";    break;
    case EOpInterpolateAtOffset:   out.debug << "interpolateAtOffset";    break;
    case EOpInterpolateAtVertex:   out.debug << "interpolateAtVertex";    break;

    case EOpSinCos:                     out.debug << "sincos";                break;
    case EOpGenMul:                     out.debug << "mul";                   break;

    case EOpAllMemoryBarrierWithGroupSync:    out.debug << "AllMemoryBarrierWithGroupSync";    break;
    case EOpDeviceMemoryBarrier:              out.debug << "DeviceMemoryBarrier";              break;
    case EOpDeviceMemoryBarrierWithGroupSync: out.debug << "DeviceMemoryBarrierWithGroupSync"; break;
    case EOpWorkgroupMemoryBarrier:           out.debug << "WorkgroupMemoryBarrier";           break;
    case EOpWorkgroupMemoryBarrierWithGroupSync: out.debug << "WorkgroupMemoryBarrierWithGroupSync"; break;

    case EOpSubgroupBarrier:                 out.debug << "subgroupBarrier"; break;
    case EOpSubgroupMemoryBarrier:           out.debug << "subgroupMemoryBarrier"; break;
    case EOpSubgroupMemoryBarrierBuffer:     out.debug << "subgroupMemoryBarrierBuffer"; break;
    case EOpSubgroupMemoryBarrierImage:      out.debug << "subgroupMemoryBarrierImage";   break;
    case EOpSubgroupMemoryBarrierShared:     out.debug << "subgroupMemoryBarrierShared"; break;
    case EOpSubgroupElect:                   out.debug << "subgroupElect"; break;
    case EOpSubgroupAll:                     out.debug << "subgroupAll"; break;
    case EOpSubgroupAny:                     out.debug << "subgroupAny"; break;
    case EOpSubgroupAllEqual:                out.debug << "subgroupAllEqual"; break;
    case EOpSubgroupBroadcast:               out.debug << "subgroupBroadcast"; break;
    case EOpSubgroupBroadcastFirst:          out.debug << "subgroupBroadcastFirst"; break;
    case EOpSubgroupBallot:                  out.debug << "subgroupBallot"; break;
    case EOpSubgroupIlwerseBallot:           out.debug << "subgroupIlwerseBallot"; break;
    case EOpSubgroupBallotBitExtract:        out.debug << "subgroupBallotBitExtract"; break;
    case EOpSubgroupBallotBitCount:          out.debug << "subgroupBallotBitCount"; break;
    case EOpSubgroupBallotInclusiveBitCount: out.debug << "subgroupBallotInclusiveBitCount"; break;
    case EOpSubgroupBallotExclusiveBitCount: out.debug << "subgroupBallotExclusiveBitCount"; break;
    case EOpSubgroupBallotFindLSB:           out.debug << "subgroupBallotFindLSB"; break;
    case EOpSubgroupBallotFindMSB:           out.debug << "subgroupBallotFindMSB"; break;
    case EOpSubgroupShuffle:                 out.debug << "subgroupShuffle"; break;
    case EOpSubgroupShuffleXor:              out.debug << "subgroupShuffleXor"; break;
    case EOpSubgroupShuffleUp:               out.debug << "subgroupShuffleUp"; break;
    case EOpSubgroupShuffleDown:             out.debug << "subgroupShuffleDown"; break;
    case EOpSubgroupAdd:                     out.debug << "subgroupAdd"; break;
    case EOpSubgroupMul:                     out.debug << "subgroupMul"; break;
    case EOpSubgroupMin:                     out.debug << "subgroupMin"; break;
    case EOpSubgroupMax:                     out.debug << "subgroupMax"; break;
    case EOpSubgroupAnd:                     out.debug << "subgroupAnd"; break;
    case EOpSubgroupOr:                      out.debug << "subgroupOr"; break;
    case EOpSubgroupXor:                     out.debug << "subgroupXor"; break;
    case EOpSubgroupInclusiveAdd:            out.debug << "subgroupInclusiveAdd"; break;
    case EOpSubgroupInclusiveMul:            out.debug << "subgroupInclusiveMul"; break;
    case EOpSubgroupInclusiveMin:            out.debug << "subgroupInclusiveMin"; break;
    case EOpSubgroupInclusiveMax:            out.debug << "subgroupInclusiveMax"; break;
    case EOpSubgroupInclusiveAnd:            out.debug << "subgroupInclusiveAnd"; break;
    case EOpSubgroupInclusiveOr:             out.debug << "subgroupInclusiveOr"; break;
    case EOpSubgroupInclusiveXor:            out.debug << "subgroupInclusiveXor"; break;
    case EOpSubgroupExclusiveAdd:            out.debug << "subgroupExclusiveAdd"; break;
    case EOpSubgroupExclusiveMul:            out.debug << "subgroupExclusiveMul"; break;
    case EOpSubgroupExclusiveMin:            out.debug << "subgroupExclusiveMin"; break;
    case EOpSubgroupExclusiveMax:            out.debug << "subgroupExclusiveMax"; break;
    case EOpSubgroupExclusiveAnd:            out.debug << "subgroupExclusiveAnd"; break;
    case EOpSubgroupExclusiveOr:             out.debug << "subgroupExclusiveOr"; break;
    case EOpSubgroupExclusiveXor:            out.debug << "subgroupExclusiveXor"; break;
    case EOpSubgroupClusteredAdd:            out.debug << "subgroupClusteredAdd"; break;
    case EOpSubgroupClusteredMul:            out.debug << "subgroupClusteredMul"; break;
    case EOpSubgroupClusteredMin:            out.debug << "subgroupClusteredMin"; break;
    case EOpSubgroupClusteredMax:            out.debug << "subgroupClusteredMax"; break;
    case EOpSubgroupClusteredAnd:            out.debug << "subgroupClusteredAnd"; break;
    case EOpSubgroupClusteredOr:             out.debug << "subgroupClusteredOr"; break;
    case EOpSubgroupClusteredXor:            out.debug << "subgroupClusteredXor"; break;
    case EOpSubgroupQuadBroadcast:           out.debug << "subgroupQuadBroadcast"; break;
    case EOpSubgroupQuadSwapHorizontal:      out.debug << "subgroupQuadSwapHorizontal"; break;
    case EOpSubgroupQuadSwapVertical:        out.debug << "subgroupQuadSwapVertical"; break;
    case EOpSubgroupQuadSwapDiagonal:        out.debug << "subgroupQuadSwapDiagonal"; break;

    case EOpSubgroupPartition:                          out.debug << "subgroupPartitionLW";                          break;
    case EOpSubgroupPartitionedAdd:                     out.debug << "subgroupPartitionedAddLW";                     break;
    case EOpSubgroupPartitionedMul:                     out.debug << "subgroupPartitionedMulLW";                     break;
    case EOpSubgroupPartitionedMin:                     out.debug << "subgroupPartitionedMinLW";                     break;
    case EOpSubgroupPartitionedMax:                     out.debug << "subgroupPartitionedMaxLW";                     break;
    case EOpSubgroupPartitionedAnd:                     out.debug << "subgroupPartitionedAndLW";                     break;
    case EOpSubgroupPartitionedOr:                      out.debug << "subgroupPartitionedOrLW";                      break;
    case EOpSubgroupPartitionedXor:                     out.debug << "subgroupPartitionedXorLW";                     break;
    case EOpSubgroupPartitionedInclusiveAdd:            out.debug << "subgroupPartitionedInclusiveAddLW";            break;
    case EOpSubgroupPartitionedInclusiveMul:            out.debug << "subgroupPartitionedInclusiveMulLW";            break;
    case EOpSubgroupPartitionedInclusiveMin:            out.debug << "subgroupPartitionedInclusiveMinLW";            break;
    case EOpSubgroupPartitionedInclusiveMax:            out.debug << "subgroupPartitionedInclusiveMaxLW";            break;
    case EOpSubgroupPartitionedInclusiveAnd:            out.debug << "subgroupPartitionedInclusiveAndLW";            break;
    case EOpSubgroupPartitionedInclusiveOr:             out.debug << "subgroupPartitionedInclusiveOrLW";             break;
    case EOpSubgroupPartitionedInclusiveXor:            out.debug << "subgroupPartitionedInclusiveXorLW";            break;
    case EOpSubgroupPartitionedExclusiveAdd:            out.debug << "subgroupPartitionedExclusiveAddLW";            break;
    case EOpSubgroupPartitionedExclusiveMul:            out.debug << "subgroupPartitionedExclusiveMulLW";            break;
    case EOpSubgroupPartitionedExclusiveMin:            out.debug << "subgroupPartitionedExclusiveMinLW";            break;
    case EOpSubgroupPartitionedExclusiveMax:            out.debug << "subgroupPartitionedExclusiveMaxLW";            break;
    case EOpSubgroupPartitionedExclusiveAnd:            out.debug << "subgroupPartitionedExclusiveAndLW";            break;
    case EOpSubgroupPartitionedExclusiveOr:             out.debug << "subgroupPartitionedExclusiveOrLW";             break;
    case EOpSubgroupPartitionedExclusiveXor:            out.debug << "subgroupPartitionedExclusiveXorLW";            break;

    case EOpSubpassLoad:   out.debug << "subpassLoad";   break;
    case EOpSubpassLoadMS: out.debug << "subpassLoadMS"; break;

    case EOpTrace:                            out.debug << "traceLW"; break;
    case EOpReportIntersection:               out.debug << "reportIntersectionLW"; break;
    case EOpIgnoreIntersection:               out.debug << "ignoreIntersectionLW"; break;
    case EOpTerminateRay:                     out.debug << "terminateRayLW"; break;
    case EOpExelwteCallable:                  out.debug << "exelwteCallableLW"; break;
    case EOpWritePackedPrimitiveIndices4x8LW: out.debug << "writePackedPrimitiveIndices4x8LW"; break;

    case EOpRayQueryInitialize:                                            out.debug << "rayQueryInitializeEXT"; break;
    case EOpRayQueryTerminate:                                             out.debug << "rayQueryTerminateEXT"; break;
    case EOpRayQueryGenerateIntersection:                                  out.debug << "rayQueryGenerateIntersectionEXT"; break;
    case EOpRayQueryConfirmIntersection:                                   out.debug << "rayQueryConfirmIntersectionEXT"; break;
    case EOpRayQueryProceed:                                               out.debug << "rayQueryProceedEXT"; break;
    case EOpRayQueryGetIntersectionType:                                   out.debug << "rayQueryGetIntersectionTypeEXT"; break;
    case EOpRayQueryGetRayTMin:                                            out.debug << "rayQueryGetRayTMinEXT"; break;
    case EOpRayQueryGetRayFlags:                                           out.debug << "rayQueryGetRayFlagsEXT"; break;
    case EOpRayQueryGetIntersectionT:                                      out.debug << "rayQueryGetIntersectionTEXT"; break;
    case EOpRayQueryGetIntersectionInstanceLwstomIndex:                    out.debug << "rayQueryGetIntersectionInstanceLwstomIndexEXT"; break;
    case EOpRayQueryGetIntersectionInstanceId:                             out.debug << "rayQueryGetIntersectionInstanceIdEXT"; break;
    case EOpRayQueryGetIntersectionInstanceShaderBindingTableRecordOffset: out.debug << "rayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetEXT"; break;
    case EOpRayQueryGetIntersectionGeometryIndex:                          out.debug << "rayQueryGetIntersectionGeometryIndexEXT"; break;
    case EOpRayQueryGetIntersectionPrimitiveIndex:                         out.debug << "rayQueryGetIntersectionPrimitiveIndexEXT"; break;
    case EOpRayQueryGetIntersectionBarycentrics:                           out.debug << "rayQueryGetIntersectionBarycentricsEXT"; break;
    case EOpRayQueryGetIntersectionFrontFace:                              out.debug << "rayQueryGetIntersectionFrontFaceEXT"; break;
    case EOpRayQueryGetIntersectionCandidateAABBOpaque:                    out.debug << "rayQueryGetIntersectionCandidateAABBOpaqueEXT"; break;
    case EOpRayQueryGetIntersectionObjectRayDirection:                     out.debug << "rayQueryGetIntersectionObjectRayDirectionEXT"; break;
    case EOpRayQueryGetIntersectionObjectRayOrigin:                        out.debug << "rayQueryGetIntersectionObjectRayOriginEXT"; break;
    case EOpRayQueryGetWorldRayDirection:                                  out.debug << "rayQueryGetWorldRayDirectionEXT"; break;
    case EOpRayQueryGetWorldRayOrigin:                                     out.debug << "rayQueryGetWorldRayOriginEXT"; break;
    case EOpRayQueryGetIntersectionObjectToWorld:                          out.debug << "rayQueryGetIntersectionObjectToWorldEXT"; break;
    case EOpRayQueryGetIntersectionWorldToObject:                          out.debug << "rayQueryGetIntersectionWorldToObjectEXT"; break;

    case EOpCooperativeMatrixLoad:  out.debug << "Load cooperative matrix";  break;
    case EOpCooperativeMatrixStore:  out.debug << "Store cooperative matrix";  break;
    case EOpCooperativeMatrixMulAdd: out.debug << "MulAdd cooperative matrices"; break;

    case EOpIsHelperIlwocation: out.debug << "IsHelperIlwocation"; break;
    case EOpDebugPrintf:  out.debug << "Debug printf";  break;

    default: out.debug.message(EPrefixError, "Bad aggregation op");
    }

    if (node->getOp() != EOpSequence && node->getOp() != EOpParameters)
        out.debug << " (" << node->getCompleteString() << ")";

    out.debug << "\n";

    return true;
}

bool TOutputTraverser::visitSelection(TVisit /* visit */, TIntermSelection* node)
{
    TInfoSink& out = infoSink;

    OutputTreeText(out, node, depth);

    out.debug << "Test condition and select";
    out.debug << " (" << node->getCompleteString() << ")";

    if (node->getShortCirlwit() == false)
        out.debug << ": no shortcirlwit";
    if (node->getFlatten())
        out.debug << ": Flatten";
    if (node->getDontFlatten())
        out.debug << ": DontFlatten";
    out.debug << "\n";

    ++depth;

    OutputTreeText(out, node, depth);
    out.debug << "Condition\n";
    node->getCondition()->traverse(this);

    OutputTreeText(out, node, depth);
    if (node->getTrueBlock()) {
        out.debug << "true case\n";
        node->getTrueBlock()->traverse(this);
    } else
        out.debug << "true case is null\n";

    if (node->getFalseBlock()) {
        OutputTreeText(out, node, depth);
        out.debug << "false case\n";
        node->getFalseBlock()->traverse(this);
    }

    --depth;

    return false;
}

// Print infinities and NaNs, and numbers in a portable way.
// Goals:
//   - portable (across IEEE 754 platforms)
//   - shows all possible IEEE values
//   - shows simple numbers in a simple way, e.g., no leading/trailing 0s
//   - shows all digits, no premature rounding
static void OutputDouble(TInfoSink& out, double value, TOutputTraverser::EExtraOutput extra)
{
    if (IsInfinity(value)) {
        if (value < 0)
            out.debug << "-1.#INF";
        else
            out.debug << "+1.#INF";
    } else if (IsNan(value))
        out.debug << "1.#IND";
    else {
        const int maxSize = 340;
        char buf[maxSize];
        const char* format = "%f";
        if (fabs(value) > 0.0 && (fabs(value) < 1e-5 || fabs(value) > 1e12))
            format = "%-.13e";
        int len = snprintf(buf, maxSize, format, value);
        assert(len < maxSize);

        // remove a leading zero in the 100s slot in exponent; it is not portable
        // pattern:   XX...XXXe+0XX or XX...XXXe-0XX
        if (len > 5) {
            if (buf[len-5] == 'e' && (buf[len-4] == '+' || buf[len-4] == '-') && buf[len-3] == '0') {
                buf[len-3] = buf[len-2];
                buf[len-2] = buf[len-1];
                buf[len-1] = '\0';
            }
        }

        out.debug << buf;

        switch (extra) {
        case TOutputTraverser::BinaryDoubleOutput:
        {
            uint64_t b;
            static_assert(sizeof(b) == sizeof(value), "sizeof(uint64_t) != sizeof(double)");
            memcpy(&b, &value, sizeof(b));

            out.debug << " : ";
            for (size_t i = 0; i < 8 * sizeof(value); ++i, ++b) {
                out.debug << ((b & 0x8000000000000000) != 0 ? "1" : "0");
                b <<= 1;
            }
            break;
        }
        default:
            break;
        }
    }
}

static void OutputConstantUnion(TInfoSink& out, const TIntermTyped* node, const TConstUnionArray& constUnion,
    TOutputTraverser::EExtraOutput extra, int depth)
{
    int size = node->getType().computeNumComponents();

    for (int i = 0; i < size; i++) {
        OutputTreeText(out, node, depth);
        switch (constUnion[i].getType()) {
        case EbtBool:
            if (constUnion[i].getBConst())
                out.debug << "true";
            else
                out.debug << "false";

            out.debug << " (" << "const bool" << ")";

            out.debug << "\n";
            break;
        case EbtFloat:
        case EbtDouble:
        case EbtFloat16:
            OutputDouble(out, constUnion[i].getDConst(), extra);
            out.debug << "\n";
            break;
        case EbtInt8:
            {
                const int maxSize = 300;
                char buf[maxSize];
                snprintf(buf, maxSize, "%d (%s)", constUnion[i].getI8Const(), "const int8_t");

                out.debug << buf << "\n";
            }
            break;
        case EbtUint8:
            {
                const int maxSize = 300;
                char buf[maxSize];
                snprintf(buf, maxSize, "%u (%s)", constUnion[i].getU8Const(), "const uint8_t");

                out.debug << buf << "\n";
            }
            break;
        case EbtInt16:
            {
                const int maxSize = 300;
                char buf[maxSize];
                snprintf(buf, maxSize, "%d (%s)", constUnion[i].getI16Const(), "const int16_t");

                out.debug << buf << "\n";
            }
            break;
        case EbtUint16:
            {
                const int maxSize = 300;
                char buf[maxSize];
                snprintf(buf, maxSize, "%u (%s)", constUnion[i].getU16Const(), "const uint16_t");

                out.debug << buf << "\n";
            }
            break;
        case EbtInt:
            {
                const int maxSize = 300;
                char buf[maxSize];
                snprintf(buf, maxSize, "%d (%s)", constUnion[i].getIConst(), "const int");

                out.debug << buf << "\n";
            }
            break;
        case EbtUint:
            {
                const int maxSize = 300;
                char buf[maxSize];
                snprintf(buf, maxSize, "%u (%s)", constUnion[i].getUConst(), "const uint");

                out.debug << buf << "\n";
            }
            break;
        case EbtInt64:
            {
                const int maxSize = 300;
                char buf[maxSize];
                snprintf(buf, maxSize, "%lld (%s)", constUnion[i].getI64Const(), "const int64_t");

                out.debug << buf << "\n";
            }
            break;
        case EbtUint64:
            {
                const int maxSize = 300;
                char buf[maxSize];
                snprintf(buf, maxSize, "%llu (%s)", constUnion[i].getU64Const(), "const uint64_t");

                out.debug << buf << "\n";
            }
            break;
        default:
            out.info.message(EPrefixInternalError, "Unknown constant", node->getLoc());
            break;
        }
    }
}

void TOutputTraverser::visitConstantUnion(TIntermConstantUnion* node)
{
    OutputTreeText(infoSink, node, depth);
    infoSink.debug << "Constant:\n";

    OutputConstantUnion(infoSink, node, node->getConstArray(), extraOutput, depth + 1);
}

void TOutputTraverser::visitSymbol(TIntermSymbol* node)
{
    OutputTreeText(infoSink, node, depth);

    infoSink.debug << "'" << node->getName() << "' (" << node->getCompleteString() << ")\n";

    if (! node->getConstArray().empty())
        OutputConstantUnion(infoSink, node, node->getConstArray(), extraOutput, depth + 1);
    else if (node->getConstSubtree()) {
        incrementDepth(node);
        node->getConstSubtree()->traverse(this);
        decrementDepth();
    }
}

bool TOutputTraverser::visitLoop(TVisit /* visit */, TIntermLoop* node)
{
    TInfoSink& out = infoSink;

    OutputTreeText(out, node, depth);

    out.debug << "Loop with condition ";
    if (! node->testFirst())
        out.debug << "not ";
    out.debug << "tested first";

    if (node->getUnroll())
        out.debug << ": Unroll";
    if (node->getDontUnroll())
        out.debug << ": DontUnroll";
    if (node->getLoopDependency()) {
        out.debug << ": Dependency ";
        out.debug << node->getLoopDependency();
    }
    out.debug << "\n";

    ++depth;

    OutputTreeText(infoSink, node, depth);
    if (node->getTest()) {
        out.debug << "Loop Condition\n";
        node->getTest()->traverse(this);
    } else
        out.debug << "No loop condition\n";

    OutputTreeText(infoSink, node, depth);
    if (node->getBody()) {
        out.debug << "Loop Body\n";
        node->getBody()->traverse(this);
    } else
        out.debug << "No loop body\n";

    if (node->getTerminal()) {
        OutputTreeText(infoSink, node, depth);
        out.debug << "Loop Terminal Expression\n";
        node->getTerminal()->traverse(this);
    }

    --depth;

    return false;
}

bool TOutputTraverser::visitBranch(TVisit /* visit*/, TIntermBranch* node)
{
    TInfoSink& out = infoSink;

    OutputTreeText(out, node, depth);

    switch (node->getFlowOp()) {
    case EOpKill:      out.debug << "Branch: Kill";           break;
    case EOpBreak:     out.debug << "Branch: Break";          break;
    case EOpContinue:  out.debug << "Branch: Continue";       break;
    case EOpReturn:    out.debug << "Branch: Return";         break;
    case EOpCase:      out.debug << "case: ";                 break;
    case EOpDemote:    out.debug << "Demote";                 break;
    case EOpDefault:   out.debug << "default: ";              break;
    default:               out.debug << "Branch: Unknown Branch"; break;
    }

    if (node->getExpression()) {
        out.debug << " with expression\n";
        ++depth;
        node->getExpression()->traverse(this);
        --depth;
    } else
        out.debug << "\n";

    return false;
}

bool TOutputTraverser::visitSwitch(TVisit /* visit */, TIntermSwitch* node)
{
    TInfoSink& out = infoSink;

    OutputTreeText(out, node, depth);
    out.debug << "switch";

    if (node->getFlatten())
        out.debug << ": Flatten";
    if (node->getDontFlatten())
        out.debug << ": DontFlatten";
    out.debug << "\n";

    OutputTreeText(out, node, depth);
    out.debug << "condition\n";
    ++depth;
    node->getCondition()->traverse(this);

    --depth;
    OutputTreeText(out, node, depth);
    out.debug << "body\n";
    ++depth;
    node->getBody()->traverse(this);

    --depth;

    return false;
}

//
// This function is the one to call externally to start the traversal.
// Individual functions can be initialized to 0 to skip processing of that
// type of node.  It's children will still be processed.
//
void TIntermediate::output(TInfoSink& infoSink, bool tree)
{
    infoSink.debug << "Shader version: " << version << "\n";
    if (requestedExtensions.size() > 0) {
        for (auto extIt = requestedExtensions.begin(); extIt != requestedExtensions.end(); ++extIt)
            infoSink.debug << "Requested " << extIt->first << "\n";
    }

    if (xfbMode)
        infoSink.debug << "in xfb mode\n";

    switch (language) {
    case EShLangVertex:
        break;

    case EShLangTessControl:
        infoSink.debug << "vertices = " << vertices << "\n";

        if (inputPrimitive != ElgNone)
            infoSink.debug << "input primitive = " << TQualifier::getGeometryString(inputPrimitive) << "\n";
        if (vertexSpacing != EvsNone)
            infoSink.debug << "vertex spacing = " << TQualifier::getVertexSpacingString(vertexSpacing) << "\n";
        if (vertexOrder != EvoNone)
            infoSink.debug << "triangle order = " << TQualifier::getVertexOrderString(vertexOrder) << "\n";
        break;

    case EShLangTessEvaluation:
        infoSink.debug << "input primitive = " << TQualifier::getGeometryString(inputPrimitive) << "\n";
        infoSink.debug << "vertex spacing = " << TQualifier::getVertexSpacingString(vertexSpacing) << "\n";
        infoSink.debug << "triangle order = " << TQualifier::getVertexOrderString(vertexOrder) << "\n";
        if (pointMode)
            infoSink.debug << "using point mode\n";
        break;

    case EShLangGeometry:
        infoSink.debug << "ilwocations = " << ilwocations << "\n";
        infoSink.debug << "max_vertices = " << vertices << "\n";
        infoSink.debug << "input primitive = " << TQualifier::getGeometryString(inputPrimitive) << "\n";
        infoSink.debug << "output primitive = " << TQualifier::getGeometryString(outputPrimitive) << "\n";
        break;

    case EShLangFragment:
        if (pixelCenterInteger)
            infoSink.debug << "gl_FragCoord pixel center is integer\n";
        if (originUpperLeft)
            infoSink.debug << "gl_FragCoord origin is upper left\n";
        if (earlyFragmentTests)
            infoSink.debug << "using early_fragment_tests\n";
        if (postDepthCoverage)
            infoSink.debug << "using post_depth_coverage\n";
        if (depthLayout != EldNone)
            infoSink.debug << "using " << TQualifier::getLayoutDepthString(depthLayout) << "\n";
        if (blendEquations != 0) {
            infoSink.debug << "using";
            // blendEquations is a mask, decode it
            for (TBlendEquationShift be = (TBlendEquationShift)0; be < EBlendCount; be = (TBlendEquationShift)(be + 1)) {
                if (blendEquations & (1 << be))
                    infoSink.debug << " " << TQualifier::getBlendEquationString(be);
            }
            infoSink.debug << "\n";
        }
        if (interlockOrdering != EioNone)
            infoSink.debug << "interlock ordering = " << TQualifier::getInterlockOrderingString(interlockOrdering) << "\n";
        break;

    case EShLangMeshLW:
        infoSink.debug << "max_vertices = " << vertices << "\n";
        infoSink.debug << "max_primitives = " << primitives << "\n";
        infoSink.debug << "output primitive = " << TQualifier::getGeometryString(outputPrimitive) << "\n";
        // Fall through
    case EShLangTaskLW:
        // Fall through
    case EShLangCompute:
        infoSink.debug << "local_size = (" << localSize[0] << ", " << localSize[1] << ", " << localSize[2] << ")\n";
        {
            if (localSizeSpecId[0] != TQualifier::layoutNotSet ||
                localSizeSpecId[1] != TQualifier::layoutNotSet ||
                localSizeSpecId[2] != TQualifier::layoutNotSet) {
                infoSink.debug << "local_size ids = (" <<
                    localSizeSpecId[0] << ", " <<
                    localSizeSpecId[1] << ", " <<
                    localSizeSpecId[2] << ")\n";
            }
        }
        break;

    default:
        break;
    }

    if (treeRoot == 0 || ! tree)
        return;

    TOutputTraverser it(infoSink);
    if (getBinaryDoubleOutput())
        it.setDoubleOutput(TOutputTraverser::BinaryDoubleOutput);
    treeRoot->traverse(&it);
}

} // end namespace glslang

#endif // not GLSLANG_WEB
