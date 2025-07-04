//
// Copyright (C) 2002-2005  3Dlabs Inc. Ltd.
// Copyright (C) 2012-2015 LunarG, Inc.
// Copyright (C) 2015-2020 Google, Inc.
// Copyright (C) 2017 ARM Limited.
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

//
// Build the intermediate representation.
//

#include "localintermediate.h"
#include "RemoveTree.h"
#include "SymbolTable.h"
#include "propagateNoContraction.h"

#include <cfloat>
#include <utility>
#include <tuple>

namespace glslang {

////////////////////////////////////////////////////////////////////////////
//
// First set of functions are to help build the intermediate representation.
// These functions are not member functions of the nodes.
// They are called from parser productions.
//
/////////////////////////////////////////////////////////////////////////////

//
// Add a terminal node for an identifier in an expression.
//
// Returns the added node.
//

TIntermSymbol* TIntermediate::addSymbol(int id, const TString& name, const TType& type, const TConstUnionArray& constArray,
                                        TIntermTyped* constSubtree, const TSourceLoc& loc)
{
    TIntermSymbol* node = new TIntermSymbol(id, name, type);
    node->setLoc(loc);
    node->setConstArray(constArray);
    node->setConstSubtree(constSubtree);

    return node;
}

TIntermSymbol* TIntermediate::addSymbol(const TIntermSymbol& intermSymbol)
{
    return addSymbol(intermSymbol.getId(),
                     intermSymbol.getName(),
                     intermSymbol.getType(),
                     intermSymbol.getConstArray(),
                     intermSymbol.getConstSubtree(),
                     intermSymbol.getLoc());
}

TIntermSymbol* TIntermediate::addSymbol(const TVariable& variable)
{
    glslang::TSourceLoc loc; // just a null location
    loc.init();

    return addSymbol(variable, loc);
}

TIntermSymbol* TIntermediate::addSymbol(const TVariable& variable, const TSourceLoc& loc)
{
    return addSymbol(variable.getUniqueId(), variable.getName(), variable.getType(), variable.getConstArray(), variable.getConstSubtree(), loc);
}

TIntermSymbol* TIntermediate::addSymbol(const TType& type, const TSourceLoc& loc)
{
    TConstUnionArray unionArray;  // just a null constant

    return addSymbol(0, "", type, unionArray, nullptr, loc);
}

//
// Connect two nodes with a new parent that does a binary operation on the nodes.
//
// Returns the added node.
//
// Returns nullptr if the working colwersions and promotions could not be found.
//
TIntermTyped* TIntermediate::addBinaryMath(TOperator op, TIntermTyped* left, TIntermTyped* right, TSourceLoc loc)
{
    // No operations work on blocks
    if (left->getType().getBasicType() == EbtBlock || right->getType().getBasicType() == EbtBlock)
        return nullptr;

    // Colwert "reference +/- int" and "reference - reference" to integer math
    if ((op == EOpAdd || op == EOpSub) && extensionRequested(E_GL_EXT_buffer_reference2)) {

        // No addressing math on struct with unsized array.
        if ((left->isReference() && left->getType().getReferentType()->containsUnsizedArray()) ||
            (right->isReference() && right->getType().getReferentType()->containsUnsizedArray())) {
            return nullptr;
        }

        if (left->isReference() && isTypeInt(right->getBasicType())) {
            const TType& referenceType = left->getType();
            TIntermConstantUnion* size = addConstantUnion((unsigned long long)computeBufferReferenceTypeSize(left->getType()), loc, true);
            left  = addBuiltInFunctionCall(loc, EOpColwPtrToUint64, true, left, TType(EbtUint64));

            right = createColwersion(EbtInt64, right);
            right = addBinaryMath(EOpMul, right, size, loc);

            TIntermTyped *node = addBinaryMath(op, left, right, loc);
            node = addBuiltInFunctionCall(loc, EOpColwUint64ToPtr, true, node, referenceType);
            return node;
        }

        if (op == EOpAdd && right->isReference() && isTypeInt(left->getBasicType())) {
            const TType& referenceType = right->getType();
            TIntermConstantUnion* size = addConstantUnion((unsigned long long)computeBufferReferenceTypeSize(right->getType()), loc, true);
            right = addBuiltInFunctionCall(loc, EOpColwPtrToUint64, true, right, TType(EbtUint64));

            left  = createColwersion(EbtInt64, left);
            left  = addBinaryMath(EOpMul, left, size, loc);

            TIntermTyped *node = addBinaryMath(op, left, right, loc);
            node = addBuiltInFunctionCall(loc, EOpColwUint64ToPtr, true, node, referenceType);
            return node;
        }

        if (op == EOpSub && left->isReference() && right->isReference()) {
            TIntermConstantUnion* size = addConstantUnion((long long)computeBufferReferenceTypeSize(left->getType()), loc, true);

            left = addBuiltInFunctionCall(loc, EOpColwPtrToUint64, true, left, TType(EbtUint64));
            right = addBuiltInFunctionCall(loc, EOpColwPtrToUint64, true, right, TType(EbtUint64));

            left = addBuiltInFunctionCall(loc, EOpColwUint64ToInt64, true, left, TType(EbtInt64));
            right = addBuiltInFunctionCall(loc, EOpColwUint64ToInt64, true, right, TType(EbtInt64));

            left = addBinaryMath(EOpSub, left, right, loc);

            TIntermTyped *node = addBinaryMath(EOpDiv, left, size, loc);
            return node;
        }

        // No other math operators supported on references
        if (left->isReference() || right->isReference()) {
            return nullptr;
        }
    }

    // Try colwerting the children's base types to compatible types.
    auto children = addColwersion(op, left, right);
    left = std::get<0>(children);
    right = std::get<1>(children);

    if (left == nullptr || right == nullptr)
        return nullptr;

    // Colwert the children's type shape to be compatible.
    addBiShapeColwersion(op, left, right);
    if (left == nullptr || right == nullptr)
        return nullptr;

    //
    // Need a new node holding things together.  Make
    // one and promote it to the right type.
    //
    TIntermBinary* node = addBinaryNode(op, left, right, loc);
    if (! promote(node))
        return nullptr;

    node->updatePrecision();

    //
    // If they are both (non-specialization) constants, they must be folded.
    // (Unless it's the sequence (comma) operator, but that's handled in addComma().)
    //
    TIntermConstantUnion *leftTempConstant = node->getLeft()->getAsConstantUnion();
    TIntermConstantUnion *rightTempConstant = node->getRight()->getAsConstantUnion();
    if (leftTempConstant && rightTempConstant) {
        TIntermTyped* folded = leftTempConstant->fold(node->getOp(), rightTempConstant);
        if (folded)
            return folded;
    }

    // If can propagate spec-constantness and if the operation is an allowed
    // specialization-constant operation, make a spec-constant.
    if (specConstantPropagates(*node->getLeft(), *node->getRight()) && isSpecializationOperation(*node))
        node->getWritableType().getQualifier().makeSpecConstant();

    // If must propagate nonuniform, make a nonuniform.
    if ((node->getLeft()->getQualifier().isNonUniform() || node->getRight()->getQualifier().isNonUniform()) &&
            isNonuniformPropagating(node->getOp()))
        node->getWritableType().getQualifier().nonUniform = true;

    return node;
}

//
// Low level: add binary node (no promotions or other argument modifications)
//
TIntermBinary* TIntermediate::addBinaryNode(TOperator op, TIntermTyped* left, TIntermTyped* right, TSourceLoc loc) const
{
    // build the node
    TIntermBinary* node = new TIntermBinary(op);
    if (loc.line == 0)
        loc = left->getLoc();
    node->setLoc(loc);
    node->setLeft(left);
    node->setRight(right);

    return node;
}

//
// like non-type form, but sets node's type.
//
TIntermBinary* TIntermediate::addBinaryNode(TOperator op, TIntermTyped* left, TIntermTyped* right, TSourceLoc loc, const TType& type) const
{
    TIntermBinary* node = addBinaryNode(op, left, right, loc);
    node->setType(type);
    return node;
}

//
// Low level: add unary node (no promotions or other argument modifications)
//
TIntermUnary* TIntermediate::addUnaryNode(TOperator op, TIntermTyped* child, TSourceLoc loc) const
{
    TIntermUnary* node = new TIntermUnary(op);
    if (loc.line == 0)
        loc = child->getLoc();
    node->setLoc(loc);
    node->setOperand(child);

    return node;
}

//
// like non-type form, but sets node's type.
//
TIntermUnary* TIntermediate::addUnaryNode(TOperator op, TIntermTyped* child, TSourceLoc loc, const TType& type) const
{
    TIntermUnary* node = addUnaryNode(op, child, loc);
    node->setType(type);
    return node;
}

//
// Connect two nodes through an assignment.
//
// Returns the added node.
//
// Returns nullptr if the 'right' type could not be colwerted to match the 'left' type,
// or the resulting operation cannot be properly promoted.
//
TIntermTyped* TIntermediate::addAssign(TOperator op, TIntermTyped* left, TIntermTyped* right, TSourceLoc loc)
{
    // No block assignment
    if (left->getType().getBasicType() == EbtBlock || right->getType().getBasicType() == EbtBlock)
        return nullptr;

    // Colwert "reference += int" to "reference = reference + int". We need this because the
    // "reference + int" callwlation ilwolves a cast back to the original type, which makes it
    // not an lvalue.
    if ((op == EOpAddAssign || op == EOpSubAssign) && left->isReference() &&
        extensionRequested(E_GL_EXT_buffer_reference2)) {

        if (!(right->getType().isScalar() && right->getType().isIntegerDomain()))
            return nullptr;

        TIntermTyped* node = addBinaryMath(op == EOpAddAssign ? EOpAdd : EOpSub, left, right, loc);
        if (!node)
            return nullptr;

        TIntermSymbol* symbol = left->getAsSymbolNode();
        left = addSymbol(*symbol);

        node = addAssign(EOpAssign, left, node, loc);
        return node;
    }

    //
    // Like adding binary math, except the colwersion can only go
    // from right to left.
    //

    // colwert base types, nullptr return means not possible
    right = addColwersion(op, left->getType(), right);
    if (right == nullptr)
        return nullptr;

    // colwert shape
    right = addUniShapeColwersion(op, left->getType(), right);

    // build the node
    TIntermBinary* node = addBinaryNode(op, left, right, loc);

    if (! promote(node))
        return nullptr;

    node->updatePrecision();

    return node;
}

//
// Connect two nodes through an index operator, where the left node is the base
// of an array or struct, and the right node is a direct or indirect offset.
//
// Returns the added node.
// The caller should set the type of the returned node.
//
TIntermTyped* TIntermediate::addIndex(TOperator op, TIntermTyped* base, TIntermTyped* index, TSourceLoc loc)
{
    // caller should set the type
    return addBinaryNode(op, base, index, loc);
}

//
// Add one node as the parent of another that it operates on.
//
// Returns the added node.
//
TIntermTyped* TIntermediate::addUnaryMath(TOperator op, TIntermTyped* child, TSourceLoc loc)
{
    if (child == 0)
        return nullptr;

    if (child->getType().getBasicType() == EbtBlock)
        return nullptr;

    switch (op) {
    case EOpLogicalNot:
        if (getSource() == EShSourceHlsl) {
            break; // HLSL can promote logical not
        }

        if (child->getType().getBasicType() != EbtBool || child->getType().isMatrix() || child->getType().isArray() || child->getType().isVector()) {
            return nullptr;
        }
        break;

    case EOpPostIncrement:
    case EOpPreIncrement:
    case EOpPostDecrement:
    case EOpPreDecrement:
    case EOpNegative:
        if (child->getType().getBasicType() == EbtStruct || child->getType().isArray())
            return nullptr;
    default: break; // some compilers want this
    }

    //
    // Do we need to promote the operand?
    //
    TBasicType newType = EbtVoid;
    switch (op) {
    case EOpConstructBool:   newType = EbtBool;   break;
    case EOpConstructFloat:  newType = EbtFloat;  break;
    case EOpConstructInt:    newType = EbtInt;    break;
    case EOpConstructUint:   newType = EbtUint;   break;
#ifndef GLSLANG_WEB
    case EOpConstructInt8:   newType = EbtInt8;   break;
    case EOpConstructUint8:  newType = EbtUint8;  break;
    case EOpConstructInt16:  newType = EbtInt16;  break;
    case EOpConstructUint16: newType = EbtUint16; break;
    case EOpConstructInt64:  newType = EbtInt64;  break;
    case EOpConstructUint64: newType = EbtUint64; break;
    case EOpConstructDouble: newType = EbtDouble; break;
    case EOpConstructFloat16: newType = EbtFloat16; break;
#endif
    default: break; // some compilers want this
    }

    if (newType != EbtVoid) {
        child = addColwersion(op, TType(newType, EvqTemporary, child->getVectorSize(),
                                                               child->getMatrixCols(),
                                                               child->getMatrixRows(),
                                                               child->isVector()),
                              child);
        if (child == nullptr)
            return nullptr;
    }

    //
    // For constructors, we are now done, it was all in the colwersion.
    // TODO: but, did this bypass constant folding?
    //
    switch (op) {
    case EOpConstructInt8:
    case EOpConstructUint8:
    case EOpConstructInt16:
    case EOpConstructUint16:
    case EOpConstructInt:
    case EOpConstructUint:
    case EOpConstructInt64:
    case EOpConstructUint64:
    case EOpConstructBool:
    case EOpConstructFloat:
    case EOpConstructDouble:
    case EOpConstructFloat16:
        return child;
    default: break; // some compilers want this
    }

    //
    // Make a new node for the operator.
    //
    TIntermUnary* node = addUnaryNode(op, child, loc);

    if (! promote(node))
        return nullptr;

    node->updatePrecision();

    // If it's a (non-specialization) constant, it must be folded.
    if (node->getOperand()->getAsConstantUnion())
        return node->getOperand()->getAsConstantUnion()->fold(op, node->getType());

    // If it's a specialization constant, the result is too,
    // if the operation is allowed for specialization constants.
    if (node->getOperand()->getType().getQualifier().isSpecConstant() && isSpecializationOperation(*node))
        node->getWritableType().getQualifier().makeSpecConstant();

    // If must propagate nonuniform, make a nonuniform.
    if (node->getOperand()->getQualifier().isNonUniform() && isNonuniformPropagating(node->getOp()))
        node->getWritableType().getQualifier().nonUniform = true;

    return node;
}

TIntermTyped* TIntermediate::addBuiltInFunctionCall(const TSourceLoc& loc, TOperator op, bool unary,
    TIntermNode* childNode, const TType& returnType)
{
    if (unary) {
        //
        // Treat it like a unary operator.
        // addUnaryMath() should get the type correct on its own;
        // including constness (which would differ from the prototype).
        //
        TIntermTyped* child = childNode->getAsTyped();
        if (child == nullptr)
            return nullptr;

        if (child->getAsConstantUnion()) {
            TIntermTyped* folded = child->getAsConstantUnion()->fold(op, returnType);
            if (folded)
                return folded;
        }

        return addUnaryNode(op, child, child->getLoc(), returnType);
    } else {
        // setAggregateOperater() calls fold() for constant folding
        TIntermTyped* node = setAggregateOperator(childNode, op, returnType, loc);

        return node;
    }
}

//
// This is the safe way to change the operator on an aggregate, as it
// does lots of error checking and fixing.  Especially for establishing
// a function call's operation on its set of parameters.  Sequences
// of instructions are also aggregates, but they just directly set
// their operator to EOpSequence.
//
// Returns an aggregate node, which could be the one passed in if
// it was already an aggregate.
//
TIntermTyped* TIntermediate::setAggregateOperator(TIntermNode* node, TOperator op, const TType& type, TSourceLoc loc)
{
    TIntermAggregate* aggNode;

    //
    // Make sure we have an aggregate.  If not turn it into one.
    //
    if (node != nullptr) {
        aggNode = node->getAsAggregate();
        if (aggNode == nullptr || aggNode->getOp() != EOpNull) {
            //
            // Make an aggregate containing this node.
            //
            aggNode = new TIntermAggregate();
            aggNode->getSequence().push_back(node);
            if (loc.line == 0)
                loc = node->getLoc();
        }
    } else
        aggNode = new TIntermAggregate();

    //
    // Set the operator.
    //
    aggNode->setOperator(op);
    if (loc.line != 0)
        aggNode->setLoc(loc);

    aggNode->setType(type);

    return fold(aggNode);
}

bool TIntermediate::isColwersionAllowed(TOperator op, TIntermTyped* node) const
{
    //
    // Does the base type even allow the operation?
    //
    switch (node->getBasicType()) {
    case EbtVoid:
        return false;
    case EbtAtomilwint:
    case EbtSampler:
    case EbtAccStruct:
        // opaque types can be passed to functions
        if (op == EOpFunction)
            break;

        // HLSL can assign samplers directly (no constructor)
        if (getSource() == EShSourceHlsl && node->getBasicType() == EbtSampler)
            break;

        // samplers can get assigned via a sampler constructor
        // (well, not yet, but code in the rest of this function is ready for it)
        if (node->getBasicType() == EbtSampler && op == EOpAssign &&
            node->getAsOperator() != nullptr && node->getAsOperator()->getOp() == EOpConstructTextureSampler)
            break;

        // otherwise, opaque types can't even be operated on, let alone colwerted
        return false;
    default:
        break;
    }

    return true;
}

bool TIntermediate::buildColwertOp(TBasicType dst, TBasicType src, TOperator& newOp) const
{
    switch (dst) {
#ifndef GLSLANG_WEB
    case EbtDouble:
        switch (src) {
        case EbtUint:    newOp = EOpColwUintToDouble;    break;
        case EbtBool:    newOp = EOpColwBoolToDouble;    break;
        case EbtFloat:   newOp = EOpColwFloatToDouble;   break;
        case EbtInt:     newOp = EOpColwIntToDouble;     break;
        case EbtInt8:    newOp = EOpColwInt8ToDouble;    break;
        case EbtUint8:   newOp = EOpColwUint8ToDouble;   break;
        case EbtInt16:   newOp = EOpColwInt16ToDouble;   break;
        case EbtUint16:  newOp = EOpColwUint16ToDouble;  break;
        case EbtFloat16: newOp = EOpColwFloat16ToDouble; break;
        case EbtInt64:   newOp = EOpColwInt64ToDouble;   break;
        case EbtUint64:  newOp = EOpColwUint64ToDouble;  break;
        default:
            return false;
        }
        break;
#endif
    case EbtFloat:
        switch (src) {
        case EbtInt:     newOp = EOpColwIntToFloat;     break;
        case EbtUint:    newOp = EOpColwUintToFloat;    break;
        case EbtBool:    newOp = EOpColwBoolToFloat;    break;
#ifndef GLSLANG_WEB
        case EbtDouble:  newOp = EOpColwDoubleToFloat;  break;
        case EbtInt8:    newOp = EOpColwInt8ToFloat;    break;
        case EbtUint8:   newOp = EOpColwUint8ToFloat;   break;
        case EbtInt16:   newOp = EOpColwInt16ToFloat;   break;
        case EbtUint16:  newOp = EOpColwUint16ToFloat;  break;
        case EbtFloat16: newOp = EOpColwFloat16ToFloat; break;
        case EbtInt64:   newOp = EOpColwInt64ToFloat;   break;
        case EbtUint64:  newOp = EOpColwUint64ToFloat;  break;
#endif
        default:
            return false;
        }
        break;
#ifndef GLSLANG_WEB
    case EbtFloat16:
        switch (src) {
        case EbtInt8:   newOp = EOpColwInt8ToFloat16;   break;
        case EbtUint8:  newOp = EOpColwUint8ToFloat16;  break;
        case EbtInt16:  newOp = EOpColwInt16ToFloat16;  break;
        case EbtUint16: newOp = EOpColwUint16ToFloat16; break;
        case EbtInt:    newOp = EOpColwIntToFloat16;    break;
        case EbtUint:   newOp = EOpColwUintToFloat16;   break;
        case EbtBool:   newOp = EOpColwBoolToFloat16;   break;
        case EbtFloat:  newOp = EOpColwFloatToFloat16;  break;
        case EbtDouble: newOp = EOpColwDoubleToFloat16; break;
        case EbtInt64:  newOp = EOpColwInt64ToFloat16;  break;
        case EbtUint64: newOp = EOpColwUint64ToFloat16; break;
        default:
            return false;
        }
        break;
#endif
    case EbtBool:
        switch (src) {
        case EbtInt:     newOp = EOpColwIntToBool;     break;
        case EbtUint:    newOp = EOpColwUintToBool;    break;
        case EbtFloat:   newOp = EOpColwFloatToBool;   break;
#ifndef GLSLANG_WEB
        case EbtDouble:  newOp = EOpColwDoubleToBool;  break;
        case EbtInt8:    newOp = EOpColwInt8ToBool;    break;
        case EbtUint8:   newOp = EOpColwUint8ToBool;   break;
        case EbtInt16:   newOp = EOpColwInt16ToBool;   break;
        case EbtUint16:  newOp = EOpColwUint16ToBool;  break;
        case EbtFloat16: newOp = EOpColwFloat16ToBool; break;
        case EbtInt64:   newOp = EOpColwInt64ToBool;   break;
        case EbtUint64:  newOp = EOpColwUint64ToBool;  break;
#endif
        default:
            return false;
        }
        break;
#ifndef GLSLANG_WEB
    case EbtInt8:
        switch (src) {
        case EbtUint8:   newOp = EOpColwUint8ToInt8;   break;
        case EbtInt16:   newOp = EOpColwInt16ToInt8;   break;
        case EbtUint16:  newOp = EOpColwUint16ToInt8;  break;
        case EbtInt:     newOp = EOpColwIntToInt8;     break;
        case EbtUint:    newOp = EOpColwUintToInt8;    break;
        case EbtInt64:   newOp = EOpColwInt64ToInt8;   break;
        case EbtUint64:  newOp = EOpColwUint64ToInt8;  break;
        case EbtBool:    newOp = EOpColwBoolToInt8;    break;
        case EbtFloat:   newOp = EOpColwFloatToInt8;   break;
        case EbtDouble:  newOp = EOpColwDoubleToInt8;  break;
        case EbtFloat16: newOp = EOpColwFloat16ToInt8; break;
        default:
            return false;
        }
        break;
    case EbtUint8:
        switch (src) {
        case EbtInt8:    newOp = EOpColwInt8ToUint8;    break;
        case EbtInt16:   newOp = EOpColwInt16ToUint8;   break;
        case EbtUint16:  newOp = EOpColwUint16ToUint8;  break;
        case EbtInt:     newOp = EOpColwIntToUint8;     break;
        case EbtUint:    newOp = EOpColwUintToUint8;    break;
        case EbtInt64:   newOp = EOpColwInt64ToUint8;   break;
        case EbtUint64:  newOp = EOpColwUint64ToUint8;  break;
        case EbtBool:    newOp = EOpColwBoolToUint8;    break;
        case EbtFloat:   newOp = EOpColwFloatToUint8;   break;
        case EbtDouble:  newOp = EOpColwDoubleToUint8;  break;
        case EbtFloat16: newOp = EOpColwFloat16ToUint8; break;
        default:
            return false;
        }
        break;

    case EbtInt16:
        switch (src) {
        case EbtUint8:   newOp = EOpColwUint8ToInt16;   break;
        case EbtInt8:    newOp = EOpColwInt8ToInt16;    break;
        case EbtUint16:  newOp = EOpColwUint16ToInt16;  break;
        case EbtInt:     newOp = EOpColwIntToInt16;     break;
        case EbtUint:    newOp = EOpColwUintToInt16;    break;
        case EbtInt64:   newOp = EOpColwInt64ToInt16;   break;
        case EbtUint64:  newOp = EOpColwUint64ToInt16;  break;
        case EbtBool:    newOp = EOpColwBoolToInt16;    break;
        case EbtFloat:   newOp = EOpColwFloatToInt16;   break;
        case EbtDouble:  newOp = EOpColwDoubleToInt16;  break;
        case EbtFloat16: newOp = EOpColwFloat16ToInt16; break;
        default:
            return false;
        }
        break;
    case EbtUint16:
        switch (src) {
        case EbtInt8:    newOp = EOpColwInt8ToUint16;    break;
        case EbtUint8:   newOp = EOpColwUint8ToUint16;   break;
        case EbtInt16:   newOp = EOpColwInt16ToUint16;   break;
        case EbtInt:     newOp = EOpColwIntToUint16;     break;
        case EbtUint:    newOp = EOpColwUintToUint16;    break;
        case EbtInt64:   newOp = EOpColwInt64ToUint16;   break;
        case EbtUint64:  newOp = EOpColwUint64ToUint16;  break;
        case EbtBool:    newOp = EOpColwBoolToUint16;    break;
        case EbtFloat:   newOp = EOpColwFloatToUint16;   break;
        case EbtDouble:  newOp = EOpColwDoubleToUint16;  break;
        case EbtFloat16: newOp = EOpColwFloat16ToUint16; break;
        default:
            return false;
        }
        break;
#endif

    case EbtInt:
        switch (src) {
        case EbtUint:    newOp = EOpColwUintToInt;    break;
        case EbtBool:    newOp = EOpColwBoolToInt;    break;
        case EbtFloat:   newOp = EOpColwFloatToInt;   break;
#ifndef GLSLANG_WEB
        case EbtInt8:    newOp = EOpColwInt8ToInt;    break;
        case EbtUint8:   newOp = EOpColwUint8ToInt;   break;
        case EbtInt16:   newOp = EOpColwInt16ToInt;   break;
        case EbtUint16:  newOp = EOpColwUint16ToInt;  break;
        case EbtDouble:  newOp = EOpColwDoubleToInt;  break;
        case EbtFloat16: newOp = EOpColwFloat16ToInt; break;
        case EbtInt64:   newOp = EOpColwInt64ToInt;   break;
        case EbtUint64:  newOp = EOpColwUint64ToInt;  break;
#endif
        default:
            return false;
        }
        break;
    case EbtUint:
        switch (src) {
        case EbtInt:     newOp = EOpColwIntToUint;     break;
        case EbtBool:    newOp = EOpColwBoolToUint;    break;
        case EbtFloat:   newOp = EOpColwFloatToUint;   break;
#ifndef GLSLANG_WEB
        case EbtInt8:    newOp = EOpColwInt8ToUint;    break;
        case EbtUint8:   newOp = EOpColwUint8ToUint;   break;
        case EbtInt16:   newOp = EOpColwInt16ToUint;   break;
        case EbtUint16:  newOp = EOpColwUint16ToUint;  break;
        case EbtDouble:  newOp = EOpColwDoubleToUint;  break;
        case EbtFloat16: newOp = EOpColwFloat16ToUint; break;
        case EbtInt64:   newOp = EOpColwInt64ToUint;   break;
        case EbtUint64:  newOp = EOpColwUint64ToUint;  break;
#endif
        default:
            return false;
        }
        break;
#ifndef GLSLANG_WEB
    case EbtInt64:
        switch (src) {
        case EbtInt8:    newOp = EOpColwInt8ToInt64;    break;
        case EbtUint8:   newOp = EOpColwUint8ToInt64;   break;
        case EbtInt16:   newOp = EOpColwInt16ToInt64;   break;
        case EbtUint16:  newOp = EOpColwUint16ToInt64;  break;
        case EbtInt:     newOp = EOpColwIntToInt64;     break;
        case EbtUint:    newOp = EOpColwUintToInt64;    break;
        case EbtBool:    newOp = EOpColwBoolToInt64;    break;
        case EbtFloat:   newOp = EOpColwFloatToInt64;   break;
        case EbtDouble:  newOp = EOpColwDoubleToInt64;  break;
        case EbtFloat16: newOp = EOpColwFloat16ToInt64; break;
        case EbtUint64:  newOp = EOpColwUint64ToInt64;  break;
        default:
            return false;
        }
        break;
    case EbtUint64:
        switch (src) {
        case EbtInt8:    newOp = EOpColwInt8ToUint64;    break;
        case EbtUint8:   newOp = EOpColwUint8ToUint64;   break;
        case EbtInt16:   newOp = EOpColwInt16ToUint64;   break;
        case EbtUint16:  newOp = EOpColwUint16ToUint64;  break;
        case EbtInt:     newOp = EOpColwIntToUint64;     break;
        case EbtUint:    newOp = EOpColwUintToUint64;    break;
        case EbtBool:    newOp = EOpColwBoolToUint64;    break;
        case EbtFloat:   newOp = EOpColwFloatToUint64;   break;
        case EbtDouble:  newOp = EOpColwDoubleToUint64;  break;
        case EbtFloat16: newOp = EOpColwFloat16ToUint64; break;
        case EbtInt64:   newOp = EOpColwInt64ToUint64;   break;
        default:
            return false;
        }
        break;
#endif
    default:
        return false;
    }
    return true;
}

// This is 'mechanism' here, it does any colwersion told.
// It is about basic type, not about shape.
// The policy comes from the shader or the calling code.
TIntermTyped* TIntermediate::createColwersion(TBasicType colwertTo, TIntermTyped* node) const
{
    //
    // Add a new newNode for the colwersion.
    //

#ifndef GLSLANG_WEB
    bool colwertToIntTypes = (colwertTo == EbtInt8  || colwertTo == EbtUint8  ||
                              colwertTo == EbtInt16 || colwertTo == EbtUint16 ||
                              colwertTo == EbtInt   || colwertTo == EbtUint   ||
                              colwertTo == EbtInt64 || colwertTo == EbtUint64);

    bool colwertFromIntTypes = (node->getBasicType() == EbtInt8  || node->getBasicType() == EbtUint8  ||
                                node->getBasicType() == EbtInt16 || node->getBasicType() == EbtUint16 ||
                                node->getBasicType() == EbtInt   || node->getBasicType() == EbtUint   ||
                                node->getBasicType() == EbtInt64 || node->getBasicType() == EbtUint64);

    bool colwertToFloatTypes = (colwertTo == EbtFloat16 || colwertTo == EbtFloat || colwertTo == EbtDouble);

    bool colwertFromFloatTypes = (node->getBasicType() == EbtFloat16 ||
                                  node->getBasicType() == EbtFloat ||
                                  node->getBasicType() == EbtDouble);

    if (! getArithemeticInt8Enabled()) {
        if (((colwertTo == EbtInt8 || colwertTo == EbtUint8) && ! colwertFromIntTypes) ||
            ((node->getBasicType() == EbtInt8 || node->getBasicType() == EbtUint8) && ! colwertToIntTypes))
            return nullptr;
    }

    if (! getArithemeticInt16Enabled()) {
        if (((colwertTo == EbtInt16 || colwertTo == EbtUint16) && ! colwertFromIntTypes) ||
            ((node->getBasicType() == EbtInt16 || node->getBasicType() == EbtUint16) && ! colwertToIntTypes))
            return nullptr;
    }

    if (! getArithemeticFloat16Enabled()) {
        if ((colwertTo == EbtFloat16 && ! colwertFromFloatTypes) ||
            (node->getBasicType() == EbtFloat16 && ! colwertToFloatTypes))
            return nullptr;
    }
#endif

    TIntermUnary* newNode = nullptr;
    TOperator newOp = EOpNull;
    if (!buildColwertOp(colwertTo, node->getBasicType(), newOp)) {
        return nullptr;
    }

    TType newType(colwertTo, EvqTemporary, node->getVectorSize(), node->getMatrixCols(), node->getMatrixRows());
    newNode = addUnaryNode(newOp, node, node->getLoc(), newType);

    if (node->getAsConstantUnion()) {
#ifndef GLSLANG_WEB
        // 8/16-bit storage extensions don't support 8/16-bit constants, so don't fold colwersions
        // to those types
        if ((getArithemeticInt8Enabled() || !(colwertTo == EbtInt8 || colwertTo == EbtUint8)) &&
            (getArithemeticInt16Enabled() || !(colwertTo == EbtInt16 || colwertTo == EbtUint16)) &&
            (getArithemeticFloat16Enabled() || !(colwertTo == EbtFloat16)))
#endif
        {
            TIntermTyped* folded = node->getAsConstantUnion()->fold(newOp, newType);
            if (folded)
                return folded;
        }
    }

    // Propagate specialization-constant-ness, if allowed
    if (node->getType().getQualifier().isSpecConstant() && isSpecializationOperation(*newNode))
        newNode->getWritableType().getQualifier().makeSpecConstant();

    return newNode;
}

TIntermTyped* TIntermediate::addColwersion(TBasicType colwertTo, TIntermTyped* node) const
{
    return createColwersion(colwertTo, node);
}

// For colwerting a pair of operands to a binary operation to compatible
// types with each other, relative to the operation in 'op'.
// This does not cover assignment operations, which is asymmetric in that the
// left type is not changeable.
// See addColwersion(op, type, node) for assignments and unary operation
// colwersions.
//
// Generally, this is folwsed on basic type colwersion, not shape colwersion.
// See addShapeColwersion() for shape colwersions.
//
// Returns the colwerted pair of nodes.
// Returns <nullptr, nullptr> when there is no colwersion.
std::tuple<TIntermTyped*, TIntermTyped*>
TIntermediate::addColwersion(TOperator op, TIntermTyped* node0, TIntermTyped* node1)
{
    if (!isColwersionAllowed(op, node0) || !isColwersionAllowed(op, node1))
        return std::make_tuple(nullptr, nullptr);

    if (node0->getType() != node1->getType()) {
        // If differing structure, then no colwersions.
        if (node0->isStruct() || node1->isStruct())
            return std::make_tuple(nullptr, nullptr);

        // If differing arrays, then no colwersions.
        if (node0->getType().isArray() || node1->getType().isArray())
            return std::make_tuple(nullptr, nullptr);

        // No implicit colwersions for operations ilwolving cooperative matrices
        if (node0->getType().isCoopMat() || node1->getType().isCoopMat())
            return std::make_tuple(node0, node1);
    }

    auto promoteTo = std::make_tuple(EbtNumTypes, EbtNumTypes);

    switch (op) {
    //
    // List all the binary ops that can implicitly colwert one operand to the other's type;
    // This implements the 'policy' for implicit type colwersion.
    //
    case EOpLessThan:
    case EOpGreaterThan:
    case EOpLessThanEqual:
    case EOpGreaterThanEqual:
    case EOpEqual:
    case EOpNotEqual:

    case EOpAdd:
    case EOpSub:
    case EOpMul:
    case EOpDiv:
    case EOpMod:

    case EOpVectorTimesScalar:
    case EOpVectorTimesMatrix:
    case EOpMatrixTimesVector:
    case EOpMatrixTimesScalar:

    case EOpAnd:
    case EOpInclusiveOr:
    case EOpExclusiveOr:

    case EOpSequence:          // used by ?:

        if (node0->getBasicType() == node1->getBasicType())
            return std::make_tuple(node0, node1);

        promoteTo = getColwersionDestinatonType(node0->getBasicType(), node1->getBasicType(), op);
        if (std::get<0>(promoteTo) == EbtNumTypes || std::get<1>(promoteTo) == EbtNumTypes)
            return std::make_tuple(nullptr, nullptr);

        break;

    case EOpLogicalAnd:
    case EOpLogicalOr:
    case EOpLogicalXor:
        if (getSource() == EShSourceHlsl)
            promoteTo = std::make_tuple(EbtBool, EbtBool);
        else
            return std::make_tuple(node0, node1);
        break;

    // There are no colwersions needed for GLSL; the shift amount just needs to be an
    // integer type, as does the base.
    // HLSL can promote bools to ints to make this work.
    case EOpLeftShift:
    case EOpRightShift:
        if (getSource() == EShSourceHlsl) {
            TBasicType node0BasicType = node0->getBasicType();
            if (node0BasicType == EbtBool)
                node0BasicType = EbtInt;
            if (node1->getBasicType() == EbtBool)
                promoteTo = std::make_tuple(node0BasicType, EbtInt);
            else
                promoteTo = std::make_tuple(node0BasicType, node1->getBasicType());
        } else {
            if (isTypeInt(node0->getBasicType()) && isTypeInt(node1->getBasicType()))
                return std::make_tuple(node0, node1);
            else
                return std::make_tuple(nullptr, nullptr);
        }
        break;

    default:
        if (node0->getType() == node1->getType())
            return std::make_tuple(node0, node1);

        return std::make_tuple(nullptr, nullptr);
    }

    TIntermTyped* newNode0;
    TIntermTyped* newNode1;

    if (std::get<0>(promoteTo) != node0->getType().getBasicType()) {
        if (node0->getAsConstantUnion())
            newNode0 = promoteConstantUnion(std::get<0>(promoteTo), node0->getAsConstantUnion());
        else
            newNode0 = createColwersion(std::get<0>(promoteTo), node0);
    } else
        newNode0 = node0;

    if (std::get<1>(promoteTo) != node1->getType().getBasicType()) {
        if (node1->getAsConstantUnion())
            newNode1 = promoteConstantUnion(std::get<1>(promoteTo), node1->getAsConstantUnion());
        else
            newNode1 = createColwersion(std::get<1>(promoteTo), node1);
    } else
        newNode1 = node1;

    return std::make_tuple(newNode0, newNode1);
}

//
// Colwert the node's type to the given type, as allowed by the operation ilwolved: 'op'.
// For implicit colwersions, 'op' is not the requested colwersion, it is the explicit
// operation requiring the implicit colwersion.
//
// Binary operation colwersions should be handled by addColwersion(op, node, node), not here.
//
// Returns a node representing the colwersion, which could be the same
// node passed in if no colwersion was needed.
//
// Generally, this is folwsed on basic type colwersion, not shape colwersion.
// See addShapeColwersion() for shape colwersions.
//
// Return nullptr if a colwersion can't be done.
//
TIntermTyped* TIntermediate::addColwersion(TOperator op, const TType& type, TIntermTyped* node)
{
    if (!isColwersionAllowed(op, node))
        return nullptr;

    // Otherwise, if types are identical, no problem
    if (type == node->getType())
        return node;

    // If one's a structure, then no colwersions.
    if (type.isStruct() || node->isStruct())
        return nullptr;

    // If one's an array, then no colwersions.
    if (type.isArray() || node->getType().isArray())
        return nullptr;

    // Note: callers are responsible for other aspects of shape,
    // like vector and matrix sizes.

    TBasicType promoteTo;
    // GL_EXT_shader_16bit_storage can't do OpConstantComposite with
    // 16-bit types, so disable promotion for those types.
    bool canPromoteConstant = true;

    switch (op) {
    //
    // Explicit colwersions (unary operations)
    //
    case EOpConstructBool:
        promoteTo = EbtBool;
        break;
    case EOpConstructFloat:
        promoteTo = EbtFloat;
        break;
    case EOpConstructInt:
        promoteTo = EbtInt;
        break;
    case EOpConstructUint:
        promoteTo = EbtUint;
        break;
#ifndef GLSLANG_WEB
    case EOpConstructDouble:
        promoteTo = EbtDouble;
        break;
    case EOpConstructFloat16:
        promoteTo = EbtFloat16;
        canPromoteConstant = extensionRequested(E_GL_EXT_shader_explicit_arithmetic_types) ||
                             extensionRequested(E_GL_EXT_shader_explicit_arithmetic_types_float16);
        break;
    case EOpConstructInt8:
        promoteTo = EbtInt8;
        canPromoteConstant = extensionRequested(E_GL_EXT_shader_explicit_arithmetic_types) ||
                             extensionRequested(E_GL_EXT_shader_explicit_arithmetic_types_int8);
        break;
    case EOpConstructUint8:
        promoteTo = EbtUint8;
        canPromoteConstant = extensionRequested(E_GL_EXT_shader_explicit_arithmetic_types) ||
                             extensionRequested(E_GL_EXT_shader_explicit_arithmetic_types_int8);
        break;
    case EOpConstructInt16:
        promoteTo = EbtInt16;
        canPromoteConstant = extensionRequested(E_GL_EXT_shader_explicit_arithmetic_types) ||
                             extensionRequested(E_GL_EXT_shader_explicit_arithmetic_types_int16);
        break;
    case EOpConstructUint16:
        promoteTo = EbtUint16;
        canPromoteConstant = extensionRequested(E_GL_EXT_shader_explicit_arithmetic_types) ||
                             extensionRequested(E_GL_EXT_shader_explicit_arithmetic_types_int16);
        break;
    case EOpConstructInt64:
        promoteTo = EbtInt64;
        break;
    case EOpConstructUint64:
        promoteTo = EbtUint64;
        break;
#endif

    case EOpLogicalNot:

    case EOpFunctionCall:

    case EOpReturn:
    case EOpAssign:
    case EOpAddAssign:
    case EOpSubAssign:
    case EOpMulAssign:
    case EOpVectorTimesScalarAssign:
    case EOpMatrixTimesScalarAssign:
    case EOpDivAssign:
    case EOpModAssign:
    case EOpAndAssign:
    case EOpInclusiveOrAssign:
    case EOpExclusiveOrAssign:

    case EOpAtan:
    case EOpClamp:
    case EOpCross:
    case EOpDistance:
    case EOpDot:
    case EOpDst:
    case EOpFaceForward:
    case EOpFma:
    case EOpFrexp:
    case EOpLdexp:
    case EOpMix:
    case EOpLit:
    case EOpMax:
    case EOpMin:
    case EOpMod:
    case EOpModf:
    case EOpPow:
    case EOpReflect:
    case EOpRefract:
    case EOpSmoothStep:
    case EOpStep:

    case EOpSequence:
    case EOpConstructStruct:
    case EOpConstructCooperativeMatrix:

        if (type.isReference() || node->getType().isReference()) {
            // types must match to assign a reference
            if (type == node->getType())
                return node;
            else
                return nullptr;
        }

        if (type.getBasicType() == node->getType().getBasicType())
            return node;

        if (canImplicitlyPromote(node->getBasicType(), type.getBasicType(), op))
            promoteTo = type.getBasicType();
        else
            return nullptr;
        break;

    // For GLSL, there are no colwersions needed; the shift amount just needs to be an
    // integer type, as do the base/result.
    // HLSL can colwert the shift from a bool to an int.
    case EOpLeftShiftAssign:
    case EOpRightShiftAssign:
    {
        if (getSource() == EShSourceHlsl && node->getType().getBasicType() == EbtBool)
            promoteTo = type.getBasicType();
        else {
            if (isTypeInt(type.getBasicType()) && isTypeInt(node->getBasicType()))
                return node;
            else
                return nullptr;
        }
        break;
    }

    default:
        // default is to require a match; all exceptions should have case statements above

        if (type.getBasicType() == node->getType().getBasicType())
            return node;
        else
            return nullptr;
    }

    if (canPromoteConstant && node->getAsConstantUnion())
        return promoteConstantUnion(promoteTo, node->getAsConstantUnion());

    //
    // Add a new newNode for the colwersion.
    //
    TIntermTyped* newNode = createColwersion(promoteTo, node);

    return newNode;
}

// Colwert the node's shape of type for the given type, as allowed by the
// operation ilwolved: 'op'.  This is for situations where there is only one
// direction to consider doing the shape colwersion.
//
// This implements policy, it call addShapeColwersion() for the mechanism.
//
// Generally, the AST represents allowed GLSL shapes, so this isn't needed
// for GLSL.  Bad shapes are caught in colwersion or promotion.
//
// Return 'node' if no colwersion was done. Promotion handles final shape
// checking.
//
TIntermTyped* TIntermediate::addUniShapeColwersion(TOperator op, const TType& type, TIntermTyped* node)
{
    // some source languages don't do this
    switch (getSource()) {
    case EShSourceHlsl:
        break;
    case EShSourceGlsl:
    default:
        return node;
    }

    // some operations don't do this
    switch (op) {
    case EOpFunctionCall:
    case EOpReturn:
        break;

    case EOpMulAssign:
        // want to support vector *= scalar native ops in AST and lower, not smear, similarly for
        // matrix *= scalar, etc.

    case EOpAddAssign:
    case EOpSubAssign:
    case EOpDivAssign:
    case EOpAndAssign:
    case EOpInclusiveOrAssign:
    case EOpExclusiveOrAssign:
    case EOpRightShiftAssign:
    case EOpLeftShiftAssign:
        if (node->getVectorSize() == 1)
            return node;
        break;

    case EOpAssign:
        break;

    case EOpMix:
        break;

    default:
        return node;
    }

    return addShapeColwersion(type, node);
}

// Colwert the nodes' shapes to be compatible for the operation 'op'.
//
// This implements policy, it call addShapeColwersion() for the mechanism.
//
// Generally, the AST represents allowed GLSL shapes, so this isn't needed
// for GLSL.  Bad shapes are caught in colwersion or promotion.
//
void TIntermediate::addBiShapeColwersion(TOperator op, TIntermTyped*& lhsNode, TIntermTyped*& rhsNode)
{
    // some source languages don't do this
    switch (getSource()) {
    case EShSourceHlsl:
        break;
    case EShSourceGlsl:
    default:
        return;
    }

    // some operations don't do this
    // 'break' will mean attempt bidirectional colwersion
    switch (op) {
    case EOpMulAssign:
    case EOpAssign:
    case EOpAddAssign:
    case EOpSubAssign:
    case EOpDivAssign:
    case EOpAndAssign:
    case EOpInclusiveOrAssign:
    case EOpExclusiveOrAssign:
    case EOpRightShiftAssign:
    case EOpLeftShiftAssign:
        // switch to unidirectional colwersion (the lhs can't change)
        rhsNode = addUniShapeColwersion(op, lhsNode->getType(), rhsNode);
        return;

    case EOpMul:
        // matrix multiply does not change shapes
        if (lhsNode->isMatrix() && rhsNode->isMatrix())
            return;
    case EOpAdd:
    case EOpSub:
    case EOpDiv:
        // want to support vector * scalar native ops in AST and lower, not smear, similarly for
        // matrix * vector, etc.
        if (lhsNode->getVectorSize() == 1 || rhsNode->getVectorSize() == 1)
            return;
        break;

    case EOpRightShift:
    case EOpLeftShift:
        // can natively support the right operand being a scalar and the left a vector,
        // but not the reverse
        if (rhsNode->getVectorSize() == 1)
            return;
        break;

    case EOpLessThan:
    case EOpGreaterThan:
    case EOpLessThanEqual:
    case EOpGreaterThanEqual:

    case EOpEqual:
    case EOpNotEqual:

    case EOpLogicalAnd:
    case EOpLogicalOr:
    case EOpLogicalXor:

    case EOpAnd:
    case EOpInclusiveOr:
    case EOpExclusiveOr:

    case EOpMix:
        break;

    default:
        return;
    }

    // Do bidirectional colwersions
    if (lhsNode->getType().isScalarOrVec1() || rhsNode->getType().isScalarOrVec1()) {
        if (lhsNode->getType().isScalarOrVec1())
            lhsNode = addShapeColwersion(rhsNode->getType(), lhsNode);
        else
            rhsNode = addShapeColwersion(lhsNode->getType(), rhsNode);
    }
    lhsNode = addShapeColwersion(rhsNode->getType(), lhsNode);
    rhsNode = addShapeColwersion(lhsNode->getType(), rhsNode);
}

// Colwert the node's shape of type for the given type, as allowed by the
// operation ilwolved: 'op'.
//
// Generally, the AST represents allowed GLSL shapes, so this isn't needed
// for GLSL.  Bad shapes are caught in colwersion or promotion.
//
// Return 'node' if no colwersion was done. Promotion handles final shape
// checking.
//
TIntermTyped* TIntermediate::addShapeColwersion(const TType& type, TIntermTyped* node)
{
    // no colwersion needed
    if (node->getType() == type)
        return node;

    // structures and arrays don't change shape, either to or from
    if (node->getType().isStruct() || node->getType().isArray() ||
        type.isStruct() || type.isArray())
        return node;

    // The new node that handles the colwersion
    TOperator constructorOp = mapTypeToConstructorOp(type);

    if (getSource() == EShSourceHlsl) {
        // HLSL rules for scalar, vector and matrix colwersions:
        // 1) scalar can become anything, initializing every component with its value
        // 2) vector and matrix can become scalar, first element is used (warning: truncation)
        // 3) matrix can become matrix with less rows and/or columns (warning: truncation)
        // 4) vector can become vector with less rows size (warning: truncation)
        // 5a) vector 4 can become 2x2 matrix (special case) (same packing layout, its a reinterpret)
        // 5b) 2x2 matrix can become vector 4 (special case) (same packing layout, its a reinterpret)

        const TType &sourceType = node->getType();

        // rule 1 for scalar to matrix is special
        if (sourceType.isScalarOrVec1() && type.isMatrix()) {

            // HLSL semantics: the scalar (or vec1) is replicated to every component of the matrix.  Left to its
            // own devices, the constructor from a scalar would populate the diagonal.  This forces replication
            // to every matrix element.

            // Note that if the node is complex (e.g, a function call), we don't want to duplicate it here
            // repeatedly, so we copy it to a temp, then use the temp.
            const int matSize = type.computeNumComponents();
            TIntermAggregate* rhsAggregate = new TIntermAggregate();

            const bool isSimple = (node->getAsSymbolNode() != nullptr) || (node->getAsConstantUnion() != nullptr);

            if (!isSimple) {
                assert(0); // TODO: use node replicator service when available.
            }

            for (int x = 0; x < matSize; ++x)
                rhsAggregate->getSequence().push_back(node);

            return setAggregateOperator(rhsAggregate, constructorOp, type, node->getLoc());
        }

        // rule 1 and 2
        if ((sourceType.isScalar() && !type.isScalar()) || (!sourceType.isScalar() && type.isScalar()))
            return setAggregateOperator(makeAggregate(node), constructorOp, type, node->getLoc());

        // rule 3 and 5b
        if (sourceType.isMatrix()) {
            // rule 3
            if (type.isMatrix()) {
                if ((sourceType.getMatrixCols() != type.getMatrixCols() || sourceType.getMatrixRows() != type.getMatrixRows()) &&
                    sourceType.getMatrixCols() >= type.getMatrixCols() && sourceType.getMatrixRows() >= type.getMatrixRows())
                    return setAggregateOperator(makeAggregate(node), constructorOp, type, node->getLoc());
            // rule 5b
            } else if (type.isVector()) {
                if (type.getVectorSize() == 4 && sourceType.getMatrixCols() == 2 && sourceType.getMatrixRows() == 2)
                    return setAggregateOperator(makeAggregate(node), constructorOp, type, node->getLoc());
            }
        }

        // rule 4 and 5a
        if (sourceType.isVector()) {
            // rule 4
            if (type.isVector())
            {
                if (sourceType.getVectorSize() > type.getVectorSize())
                    return setAggregateOperator(makeAggregate(node), constructorOp, type, node->getLoc());
            // rule 5a
            } else if (type.isMatrix()) {
                if (sourceType.getVectorSize() == 4 && type.getMatrixCols() == 2 && type.getMatrixRows() == 2)
                    return setAggregateOperator(makeAggregate(node), constructorOp, type, node->getLoc());
            }
        }
    }

    // scalar -> vector or vec1 -> vector or
    // vector -> scalar or
    // bigger vector -> smaller vector
    if ((node->getType().isScalarOrVec1() && type.isVector()) ||
        (node->getType().isVector() && type.isScalar()) ||
        (node->isVector() && type.isVector() && node->getVectorSize() > type.getVectorSize()))
        return setAggregateOperator(makeAggregate(node), constructorOp, type, node->getLoc());

    return node;
}

bool TIntermediate::isIntegralPromotion(TBasicType from, TBasicType to) const
{
    // integral promotions
    if (to == EbtInt) {
        switch(from) {
        case EbtInt8:
        case EbtInt16:
        case EbtUint8:
        case EbtUint16:
            return true;
        default:
            break;
        }
    }
    return false;
}

bool TIntermediate::isFPPromotion(TBasicType from, TBasicType to) const
{
    // floating-point promotions
    if (to == EbtDouble) {
        switch(from) {
        case EbtFloat16:
        case EbtFloat:
            return true;
        default:
            break;
        }
    }
    return false;
}

bool TIntermediate::isIntegralColwersion(TBasicType from, TBasicType to) const
{
#ifdef GLSLANG_WEB
    return false;
#endif

    switch (from) {
    case EbtInt:
        switch(to) {
        case EbtUint:
            return version >= 400 || getSource() == EShSourceHlsl;
        case EbtInt64:
        case EbtUint64:
            return true;
        default:
            break;
        }
        break;
    case EbtUint:
        switch(to) {
        case EbtInt64:
        case EbtUint64:
            return true;
        default:
            break;
        }
        break;
    case EbtInt8:
        switch (to) {
        case EbtUint8:
        case EbtInt16:
        case EbtUint16:
        case EbtUint:
        case EbtInt64:
        case EbtUint64:
            return true;
        default:
            break;
        }
        break;
    case EbtUint8:
        switch (to) {
        case EbtInt16:
        case EbtUint16:
        case EbtUint:
        case EbtInt64:
        case EbtUint64:
            return true;
        default:
            break;
        }
        break;
    case EbtInt16:
        switch(to) {
        case EbtUint16:
        case EbtUint:
        case EbtInt64:
        case EbtUint64:
            return true;
        default:
            break;
        }
        break;
    case EbtUint16:
        switch(to) {
        case EbtUint:
        case EbtInt64:
        case EbtUint64:
            return true;
        default:
            break;
        }
        break;
    case EbtInt64:
        if (to == EbtUint64) {
            return true;
        }
        break;
    default:
        break;
    }
    return false;
}

bool TIntermediate::isFPColwersion(TBasicType from, TBasicType to) const
{
#ifdef GLSLANG_WEB
    return false;
#endif

    if (to == EbtFloat && from == EbtFloat16) {
        return true;
    } else {
        return false;
    }
}

bool TIntermediate::isFPIntegralColwersion(TBasicType from, TBasicType to) const
{
    switch (from) {
    case EbtInt:
    case EbtUint:
        switch(to) {
        case EbtFloat:
        case EbtDouble:
            return true;
        default:
            break;
        }
        break;
#ifndef GLSLANG_WEB
    case EbtInt8:
    case EbtUint8:
    case EbtInt16:
    case EbtUint16:
        switch (to) {
        case EbtFloat16:
        case EbtFloat:
        case EbtDouble:
            return true;
        default:
            break;
        }
        break;
    case EbtInt64:
    case EbtUint64:
        if (to == EbtDouble) {
            return true;
        }
        break;
#endif
    default:
        break;
    }
    return false;
}

//
// See if the 'from' type is allowed to be implicitly colwerted to the
// 'to' type.  This is not about vector/array/struct, only about basic type.
//
bool TIntermediate::canImplicitlyPromote(TBasicType from, TBasicType to, TOperator op) const
{
    if ((isEsProfile() && version < 310 ) || version == 110)
        return false;

    if (from == to)
        return true;

    // TODO: Move more policies into language-specific handlers.
    // Some languages allow more general (or potentially, more specific) colwersions under some conditions.
    if (getSource() == EShSourceHlsl) {
        const bool fromColwertable = (from == EbtFloat || from == EbtDouble || from == EbtInt || from == EbtUint || from == EbtBool);
        const bool toColwertable = (to == EbtFloat || to == EbtDouble || to == EbtInt || to == EbtUint || to == EbtBool);

        if (fromColwertable && toColwertable) {
            switch (op) {
            case EOpAndAssign:               // assignments can perform arbitrary colwersions
            case EOpInclusiveOrAssign:       // ...
            case EOpExclusiveOrAssign:       // ...
            case EOpAssign:                  // ...
            case EOpAddAssign:               // ...
            case EOpSubAssign:               // ...
            case EOpMulAssign:               // ...
            case EOpVectorTimesScalarAssign: // ...
            case EOpMatrixTimesScalarAssign: // ...
            case EOpDivAssign:               // ...
            case EOpModAssign:               // ...
            case EOpReturn:                  // function returns can also perform arbitrary colwersions
            case EOpFunctionCall:            // colwersion of a calling parameter
            case EOpLogicalNot:
            case EOpLogicalAnd:
            case EOpLogicalOr:
            case EOpLogicalXor:
            case EOpConstructStruct:
                return true;
            default:
                break;
            }
        }
    }

    bool explicitTypesEnabled = extensionRequested(E_GL_EXT_shader_explicit_arithmetic_types) ||
                                extensionRequested(E_GL_EXT_shader_explicit_arithmetic_types_int8) ||
                                extensionRequested(E_GL_EXT_shader_explicit_arithmetic_types_int16) ||
                                extensionRequested(E_GL_EXT_shader_explicit_arithmetic_types_int32) ||
                                extensionRequested(E_GL_EXT_shader_explicit_arithmetic_types_int64) ||
                                extensionRequested(E_GL_EXT_shader_explicit_arithmetic_types_float16) ||
                                extensionRequested(E_GL_EXT_shader_explicit_arithmetic_types_float32) ||
                                extensionRequested(E_GL_EXT_shader_explicit_arithmetic_types_float64);
    
    if (explicitTypesEnabled) {
        // integral promotions
        if (isIntegralPromotion(from, to)) {
            return true;
        }

        // floating-point promotions
        if (isFPPromotion(from, to)) {
            return true;
        }

        // integral colwersions
        if (isIntegralColwersion(from, to)) {
            return true;
        }

        // floating-point colwersions
        if (isFPColwersion(from, to)) {
           return true;
        }

        // floating-integral colwersions
        if (isFPIntegralColwersion(from, to)) {
           return true;
        }

        // hlsl supported colwersions
        if (getSource() == EShSourceHlsl) {
            if (from == EbtBool && (to == EbtInt || to == EbtUint || to == EbtFloat))
                return true;
        }
    } else if (isEsProfile()) {
        switch (to) {
            case EbtFloat:
                switch (from) {
                case EbtInt:
                case EbtUint:
                    return extensionRequested(E_GL_EXT_shader_implicit_colwersions);
                case EbtFloat:
                    return true;
                default:
                    return false;
                }
            case EbtUint:
                switch (from) {
                case EbtInt:
                    return extensionRequested(E_GL_EXT_shader_implicit_colwersions);
                case EbtUint:
                    return true;
                default:
                    return false;
                }
            default:
                return false;
        }        
    } else {
        switch (to) {
        case EbtDouble:
            switch (from) {
            case EbtInt:
            case EbtUint:
            case EbtInt64:
            case EbtUint64:
            case EbtFloat:
            case EbtDouble:
                return true;
            case EbtInt16:
            case EbtUint16:
                return extensionRequested(E_GL_AMD_gpu_shader_int16);
            case EbtFloat16:
                return extensionRequested(E_GL_AMD_gpu_shader_half_float);
            default:
                return false;
           }
        case EbtFloat:
            switch (from) {
            case EbtInt:
            case EbtUint:
            case EbtFloat:
                 return true;
            case EbtBool:
                 return getSource() == EShSourceHlsl;
            case EbtInt16:
            case EbtUint16:
                return extensionRequested(E_GL_AMD_gpu_shader_int16);
            case EbtFloat16:
                return 
                    extensionRequested(E_GL_AMD_gpu_shader_half_float) || getSource() == EShSourceHlsl;
            default:
                 return false;
            }
        case EbtUint:
            switch (from) {
            case EbtInt:
                return version >= 400 || getSource() == EShSourceHlsl;
            case EbtUint:
                return true;
            case EbtBool:
                return getSource() == EShSourceHlsl;
            case EbtInt16:
            case EbtUint16:
                return extensionRequested(E_GL_AMD_gpu_shader_int16);
            default:
                return false;
            }
        case EbtInt:
            switch (from) {
            case EbtInt:
                return true;
            case EbtBool:
                return getSource() == EShSourceHlsl;
            case EbtInt16:
                return extensionRequested(E_GL_AMD_gpu_shader_int16);
            default:
                return false;
            }
        case EbtUint64:
            switch (from) {
            case EbtInt:
            case EbtUint:
            case EbtInt64:
            case EbtUint64:
                return true;
            case EbtInt16:
            case EbtUint16:
                return extensionRequested(E_GL_AMD_gpu_shader_int16);
            default:
                return false;
            }
        case EbtInt64:
            switch (from) {
            case EbtInt:
            case EbtInt64:
                return true;
            case EbtInt16:
                return extensionRequested(E_GL_AMD_gpu_shader_int16);
            default:
                return false;
            }
        case EbtFloat16:
            switch (from) {
            case EbtInt16:
            case EbtUint16:
                return extensionRequested(E_GL_AMD_gpu_shader_int16);
            case EbtFloat16:
                return extensionRequested(E_GL_AMD_gpu_shader_half_float);
            default:
                break;
            }
            return false;
        case EbtUint16:
            switch (from) {
            case EbtInt16:
            case EbtUint16:
                return extensionRequested(E_GL_AMD_gpu_shader_int16);
            default:
                break;
            }
            return false;
        default:
            return false;
        }
    }

    return false;
}

static bool canSignedIntTypeRepresentAllUnsignedValues(TBasicType sintType, TBasicType uintType)
{
#ifdef GLSLANG_WEB
    return false;
#endif

    switch(sintType) {
    case EbtInt8:
        switch(uintType) {
        case EbtUint8:
        case EbtUint16:
        case EbtUint:
        case EbtUint64:
            return false;
        default:
            assert(false);
            return false;
        }
        break;
    case EbtInt16:
        switch(uintType) {
        case EbtUint8:
            return true;
        case EbtUint16:
        case EbtUint:
        case EbtUint64:
            return false;
        default:
            assert(false);
            return false;
        }
        break;
    case EbtInt:
        switch(uintType) {
        case EbtUint8:
        case EbtUint16:
            return true;
        case EbtUint:
            return false;
        default:
            assert(false);
            return false;
        }
        break;
    case EbtInt64:
        switch(uintType) {
        case EbtUint8:
        case EbtUint16:
        case EbtUint:
            return true;
        case EbtUint64:
            return false;
        default:
            assert(false);
            return false;
        }
        break;
    default:
        assert(false);
        return false;
    }
}


static TBasicType getCorrespondingUnsignedType(TBasicType type)
{
#ifdef GLSLANG_WEB
    assert(type == EbtInt);
    return EbtUint;
#endif

    switch(type) {
    case EbtInt8:
        return EbtUint8;
    case EbtInt16:
        return EbtUint16;
    case EbtInt:
        return EbtUint;
    case EbtInt64:
        return EbtUint64;
    default:
        assert(false);
        return EbtNumTypes;
    }
}

// Implements the following rules
//    - If either operand has type float64_t or derived from float64_t,
//      the other shall be colwerted to float64_t or derived type.
//    - Otherwise, if either operand has type float32_t or derived from
//      float32_t, the other shall be colwerted to float32_t or derived type.
//    - Otherwise, if either operand has type float16_t or derived from
//      float16_t, the other shall be colwerted to float16_t or derived type.
//    - Otherwise, if both operands have integer types the following rules
//      shall be applied to the operands:
//      - If both operands have the same type, no further colwersion
//        is needed.
//      - Otherwise, if both operands have signed integer types or both
//        have unsigned integer types, the operand with the type of lesser
//        integer colwersion rank shall be colwerted to the type of the
//        operand with greater rank.
//      - Otherwise, if the operand that has unsigned integer type has rank
//        greater than or equal to the rank of the type of the other
//        operand, the operand with signed integer type shall be colwerted
//        to the type of the operand with unsigned integer type.
//      - Otherwise, if the type of the operand with signed integer type can
//        represent all of the values of the type of the operand with
//        unsigned integer type, the operand with unsigned integer type
//        shall be colwerted to the type of the operand with signed
//        integer type.
//      - Otherwise, both operands shall be colwerted to the unsigned
//        integer type corresponding to the type of the operand with signed
//        integer type.

std::tuple<TBasicType, TBasicType> TIntermediate::getColwersionDestinatonType(TBasicType type0, TBasicType type1, TOperator op) const
{
    TBasicType res0 = EbtNumTypes;
    TBasicType res1 = EbtNumTypes;

    if ((isEsProfile() && 
        (version < 310 || !extensionRequested(E_GL_EXT_shader_implicit_colwersions))) || 
        version == 110)
        return std::make_tuple(res0, res1);

    if (getSource() == EShSourceHlsl) {
        if (canImplicitlyPromote(type1, type0, op)) {
            res0 = type0;
            res1 = type0;
        } else if (canImplicitlyPromote(type0, type1, op)) {
            res0 = type1;
            res1 = type1;
        }
        return std::make_tuple(res0, res1);
    }

    if ((type0 == EbtDouble && canImplicitlyPromote(type1, EbtDouble, op)) ||
        (type1 == EbtDouble && canImplicitlyPromote(type0, EbtDouble, op)) ) {
        res0 = EbtDouble;
        res1 = EbtDouble;
    } else if ((type0 == EbtFloat && canImplicitlyPromote(type1, EbtFloat, op)) ||
               (type1 == EbtFloat && canImplicitlyPromote(type0, EbtFloat, op)) ) {
        res0 = EbtFloat;
        res1 = EbtFloat;
    } else if ((type0 == EbtFloat16 && canImplicitlyPromote(type1, EbtFloat16, op)) ||
               (type1 == EbtFloat16 && canImplicitlyPromote(type0, EbtFloat16, op)) ) {
        res0 = EbtFloat16;
        res1 = EbtFloat16;
    } else if (isTypeInt(type0) && isTypeInt(type1) &&
               (canImplicitlyPromote(type0, type1, op) || canImplicitlyPromote(type1, type0, op))) {
        if ((isTypeSignedInt(type0) && isTypeSignedInt(type1)) ||
            (isTypeUnsignedInt(type0) && isTypeUnsignedInt(type1))) {
            if (getTypeRank(type0) < getTypeRank(type1)) {
                res0 = type1;
                res1 = type1;
            } else {
                res0 = type0;
                res1 = type0;
            }
        } else if (isTypeUnsignedInt(type0) && (getTypeRank(type0) > getTypeRank(type1))) {
            res0 = type0;
            res1 = type0;
        } else if (isTypeUnsignedInt(type1) && (getTypeRank(type1) > getTypeRank(type0))) {
            res0 = type1;
            res1 = type1;
        } else if (isTypeSignedInt(type0)) {
            if (canSignedIntTypeRepresentAllUnsignedValues(type0, type1)) {
                res0 = type0;
                res1 = type0;
            } else {
                res0 = getCorrespondingUnsignedType(type0);
                res1 = getCorrespondingUnsignedType(type0);
            }
        } else if (isTypeSignedInt(type1)) {
            if (canSignedIntTypeRepresentAllUnsignedValues(type1, type0)) {
                res0 = type1;
                res1 = type1;
            } else {
                res0 = getCorrespondingUnsignedType(type1);
                res1 = getCorrespondingUnsignedType(type1);
            }
        }
    }

    return std::make_tuple(res0, res1);
}

//
// Given a type, find what operation would fully construct it.
//
TOperator TIntermediate::mapTypeToConstructorOp(const TType& type) const
{
    TOperator op = EOpNull;

    if (type.getQualifier().isNonUniform())
        return EOpConstructNonuniform;

    if (type.isCoopMat())
        return EOpConstructCooperativeMatrix;

    switch (type.getBasicType()) {
    case EbtStruct:
        op = EOpConstructStruct;
        break;
    case EbtSampler:
        if (type.getSampler().isCombined())
            op = EOpConstructTextureSampler;
        break;
    case EbtFloat:
        if (type.isMatrix()) {
            switch (type.getMatrixCols()) {
            case 2:
                switch (type.getMatrixRows()) {
                case 2: op = EOpConstructMat2x2; break;
                case 3: op = EOpConstructMat2x3; break;
                case 4: op = EOpConstructMat2x4; break;
                default: break; // some compilers want this
                }
                break;
            case 3:
                switch (type.getMatrixRows()) {
                case 2: op = EOpConstructMat3x2; break;
                case 3: op = EOpConstructMat3x3; break;
                case 4: op = EOpConstructMat3x4; break;
                default: break; // some compilers want this
                }
                break;
            case 4:
                switch (type.getMatrixRows()) {
                case 2: op = EOpConstructMat4x2; break;
                case 3: op = EOpConstructMat4x3; break;
                case 4: op = EOpConstructMat4x4; break;
                default: break; // some compilers want this
                }
                break;
            default: break; // some compilers want this
            }
        } else {
            switch(type.getVectorSize()) {
            case 1: op = EOpConstructFloat; break;
            case 2: op = EOpConstructVec2;  break;
            case 3: op = EOpConstructVec3;  break;
            case 4: op = EOpConstructVec4;  break;
            default: break; // some compilers want this
            }
        }
        break;
    case EbtInt:
        if (type.getMatrixCols()) {
            switch (type.getMatrixCols()) {
            case 2:
                switch (type.getMatrixRows()) {
                case 2: op = EOpConstructIMat2x2; break;
                case 3: op = EOpConstructIMat2x3; break;
                case 4: op = EOpConstructIMat2x4; break;
                default: break; // some compilers want this
                }
                break;
            case 3:
                switch (type.getMatrixRows()) {
                case 2: op = EOpConstructIMat3x2; break;
                case 3: op = EOpConstructIMat3x3; break;
                case 4: op = EOpConstructIMat3x4; break;
                default: break; // some compilers want this
                }
                break;
            case 4:
                switch (type.getMatrixRows()) {
                case 2: op = EOpConstructIMat4x2; break;
                case 3: op = EOpConstructIMat4x3; break;
                case 4: op = EOpConstructIMat4x4; break;
                default: break; // some compilers want this
                }
                break;
            }
        } else {
            switch(type.getVectorSize()) {
            case 1: op = EOpConstructInt;   break;
            case 2: op = EOpConstructIVec2; break;
            case 3: op = EOpConstructIVec3; break;
            case 4: op = EOpConstructIVec4; break;
            default: break; // some compilers want this
            }
        }
        break;
    case EbtUint:
        if (type.getMatrixCols()) {
            switch (type.getMatrixCols()) {
            case 2:
                switch (type.getMatrixRows()) {
                case 2: op = EOpConstructUMat2x2; break;
                case 3: op = EOpConstructUMat2x3; break;
                case 4: op = EOpConstructUMat2x4; break;
                default: break; // some compilers want this
                }
                break;
            case 3:
                switch (type.getMatrixRows()) {
                case 2: op = EOpConstructUMat3x2; break;
                case 3: op = EOpConstructUMat3x3; break;
                case 4: op = EOpConstructUMat3x4; break;
                default: break; // some compilers want this
                }
                break;
            case 4:
                switch (type.getMatrixRows()) {
                case 2: op = EOpConstructUMat4x2; break;
                case 3: op = EOpConstructUMat4x3; break;
                case 4: op = EOpConstructUMat4x4; break;
                default: break; // some compilers want this
                }
                break;
            }
        } else {
            switch(type.getVectorSize()) {
            case 1: op = EOpConstructUint;  break;
            case 2: op = EOpConstructUVec2; break;
            case 3: op = EOpConstructUVec3; break;
            case 4: op = EOpConstructUVec4; break;
            default: break; // some compilers want this
            }
        }
        break;
    case EbtBool:
        if (type.getMatrixCols()) {
            switch (type.getMatrixCols()) {
            case 2:
                switch (type.getMatrixRows()) {
                case 2: op = EOpConstructBMat2x2; break;
                case 3: op = EOpConstructBMat2x3; break;
                case 4: op = EOpConstructBMat2x4; break;
                default: break; // some compilers want this
                }
                break;
            case 3:
                switch (type.getMatrixRows()) {
                case 2: op = EOpConstructBMat3x2; break;
                case 3: op = EOpConstructBMat3x3; break;
                case 4: op = EOpConstructBMat3x4; break;
                default: break; // some compilers want this
                }
                break;
            case 4:
                switch (type.getMatrixRows()) {
                case 2: op = EOpConstructBMat4x2; break;
                case 3: op = EOpConstructBMat4x3; break;
                case 4: op = EOpConstructBMat4x4; break;
                default: break; // some compilers want this
                }
                break;
            }
        } else {
            switch(type.getVectorSize()) {
            case 1:  op = EOpConstructBool;  break;
            case 2:  op = EOpConstructBVec2; break;
            case 3:  op = EOpConstructBVec3; break;
            case 4:  op = EOpConstructBVec4; break;
            default: break; // some compilers want this
            }
        }
        break;
#ifndef GLSLANG_WEB
    case EbtDouble:
        if (type.getMatrixCols()) {
            switch (type.getMatrixCols()) {
            case 2:
                switch (type.getMatrixRows()) {
                case 2: op = EOpConstructDMat2x2; break;
                case 3: op = EOpConstructDMat2x3; break;
                case 4: op = EOpConstructDMat2x4; break;
                default: break; // some compilers want this
                }
                break;
            case 3:
                switch (type.getMatrixRows()) {
                case 2: op = EOpConstructDMat3x2; break;
                case 3: op = EOpConstructDMat3x3; break;
                case 4: op = EOpConstructDMat3x4; break;
                default: break; // some compilers want this
                }
                break;
            case 4:
                switch (type.getMatrixRows()) {
                case 2: op = EOpConstructDMat4x2; break;
                case 3: op = EOpConstructDMat4x3; break;
                case 4: op = EOpConstructDMat4x4; break;
                default: break; // some compilers want this
                }
                break;
            }
        } else {
            switch(type.getVectorSize()) {
            case 1: op = EOpConstructDouble; break;
            case 2: op = EOpConstructDVec2;  break;
            case 3: op = EOpConstructDVec3;  break;
            case 4: op = EOpConstructDVec4;  break;
            default: break; // some compilers want this
            }
        }
        break;
    case EbtFloat16:
        if (type.getMatrixCols()) {
            switch (type.getMatrixCols()) {
            case 2:
                switch (type.getMatrixRows()) {
                case 2: op = EOpConstructF16Mat2x2; break;
                case 3: op = EOpConstructF16Mat2x3; break;
                case 4: op = EOpConstructF16Mat2x4; break;
                default: break; // some compilers want this
                }
                break;
            case 3:
                switch (type.getMatrixRows()) {
                case 2: op = EOpConstructF16Mat3x2; break;
                case 3: op = EOpConstructF16Mat3x3; break;
                case 4: op = EOpConstructF16Mat3x4; break;
                default: break; // some compilers want this
                }
                break;
            case 4:
                switch (type.getMatrixRows()) {
                case 2: op = EOpConstructF16Mat4x2; break;
                case 3: op = EOpConstructF16Mat4x3; break;
                case 4: op = EOpConstructF16Mat4x4; break;
                default: break; // some compilers want this
                }
                break;
            }
        }
        else {
            switch (type.getVectorSize()) {
            case 1: op = EOpConstructFloat16;  break;
            case 2: op = EOpConstructF16Vec2;  break;
            case 3: op = EOpConstructF16Vec3;  break;
            case 4: op = EOpConstructF16Vec4;  break;
            default: break; // some compilers want this
            }
        }
        break;
    case EbtInt8:
        switch(type.getVectorSize()) {
        case 1: op = EOpConstructInt8;   break;
        case 2: op = EOpConstructI8Vec2; break;
        case 3: op = EOpConstructI8Vec3; break;
        case 4: op = EOpConstructI8Vec4; break;
        default: break; // some compilers want this
        }
        break;
    case EbtUint8:
        switch(type.getVectorSize()) {
        case 1: op = EOpConstructUint8;  break;
        case 2: op = EOpConstructU8Vec2; break;
        case 3: op = EOpConstructU8Vec3; break;
        case 4: op = EOpConstructU8Vec4; break;
        default: break; // some compilers want this
        }
        break;
    case EbtInt16:
        switch(type.getVectorSize()) {
        case 1: op = EOpConstructInt16;   break;
        case 2: op = EOpConstructI16Vec2; break;
        case 3: op = EOpConstructI16Vec3; break;
        case 4: op = EOpConstructI16Vec4; break;
        default: break; // some compilers want this
        }
        break;
    case EbtUint16:
        switch(type.getVectorSize()) {
        case 1: op = EOpConstructUint16;  break;
        case 2: op = EOpConstructU16Vec2; break;
        case 3: op = EOpConstructU16Vec3; break;
        case 4: op = EOpConstructU16Vec4; break;
        default: break; // some compilers want this
        }
        break;
    case EbtInt64:
        switch(type.getVectorSize()) {
        case 1: op = EOpConstructInt64;   break;
        case 2: op = EOpConstructI64Vec2; break;
        case 3: op = EOpConstructI64Vec3; break;
        case 4: op = EOpConstructI64Vec4; break;
        default: break; // some compilers want this
        }
        break;
    case EbtUint64:
        switch(type.getVectorSize()) {
        case 1: op = EOpConstructUint64;  break;
        case 2: op = EOpConstructU64Vec2; break;
        case 3: op = EOpConstructU64Vec3; break;
        case 4: op = EOpConstructU64Vec4; break;
        default: break; // some compilers want this
        }
        break;
    case EbtReference:
        op = EOpConstructReference;
        break;
#endif
    default:
        break;
    }

    return op;
}

//
// Safe way to combine two nodes into an aggregate.  Works with null pointers,
// a node that's not a aggregate yet, etc.
//
// Returns the resulting aggregate, unless nullptr was passed in for
// both existing nodes.
//
TIntermAggregate* TIntermediate::growAggregate(TIntermNode* left, TIntermNode* right)
{
    if (left == nullptr && right == nullptr)
        return nullptr;

    TIntermAggregate* aggNode = nullptr;
    if (left != nullptr)
        aggNode = left->getAsAggregate();
    if (aggNode == nullptr || aggNode->getOp() != EOpNull) {
        aggNode = new TIntermAggregate;
        if (left != nullptr)
            aggNode->getSequence().push_back(left);
    }

    if (right != nullptr)
        aggNode->getSequence().push_back(right);

    return aggNode;
}

TIntermAggregate* TIntermediate::growAggregate(TIntermNode* left, TIntermNode* right, const TSourceLoc& loc)
{
    TIntermAggregate* aggNode = growAggregate(left, right);
    if (aggNode)
        aggNode->setLoc(loc);

    return aggNode;
}

//
// Turn an existing node into an aggregate.
//
// Returns an aggregate, unless nullptr was passed in for the existing node.
//
TIntermAggregate* TIntermediate::makeAggregate(TIntermNode* node)
{
    if (node == nullptr)
        return nullptr;

    TIntermAggregate* aggNode = new TIntermAggregate;
    aggNode->getSequence().push_back(node);
    aggNode->setLoc(node->getLoc());

    return aggNode;
}

TIntermAggregate* TIntermediate::makeAggregate(TIntermNode* node, const TSourceLoc& loc)
{
    if (node == nullptr)
        return nullptr;

    TIntermAggregate* aggNode = new TIntermAggregate;
    aggNode->getSequence().push_back(node);
    aggNode->setLoc(loc);

    return aggNode;
}

//
// Make an aggregate with an empty sequence.
//
TIntermAggregate* TIntermediate::makeAggregate(const TSourceLoc& loc)
{
    TIntermAggregate* aggNode = new TIntermAggregate;
    aggNode->setLoc(loc);

    return aggNode;
}

//
// For "if" test nodes.  There are three children; a condition,
// a true path, and a false path.  The two paths are in the
// nodePair.
//
// Returns the selection node created.
//
TIntermSelection* TIntermediate::addSelection(TIntermTyped* cond, TIntermNodePair nodePair, const TSourceLoc& loc)
{
    //
    // Don't prune the false path for compile-time constants; it's needed
    // for static access analysis.
    //

    TIntermSelection* node = new TIntermSelection(cond, nodePair.node1, nodePair.node2);
    node->setLoc(loc);

    return node;
}

TIntermTyped* TIntermediate::addComma(TIntermTyped* left, TIntermTyped* right, const TSourceLoc& loc)
{
    // However, the lowest precedence operators of the sequence operator ( , ) and the assignment operators
    // ... are not included in the operators that can create a constant expression.
    //
    // if (left->getType().getQualifier().storage == EvqConst &&
    //    right->getType().getQualifier().storage == EvqConst) {

    //    return right;
    //}

    TIntermTyped *commaAggregate = growAggregate(left, right, loc);
    commaAggregate->getAsAggregate()->setOperator(EOpComma);
    commaAggregate->setType(right->getType());
    commaAggregate->getWritableType().getQualifier().makeTemporary();

    return commaAggregate;
}

TIntermTyped* TIntermediate::addMethod(TIntermTyped* object, const TType& type, const TString* name, const TSourceLoc& loc)
{
    TIntermMethod* method = new TIntermMethod(object, type, *name);
    method->setLoc(loc);

    return method;
}

//
// For "?:" test nodes.  There are three children; a condition,
// a true path, and a false path.  The two paths are specified
// as separate parameters. For vector 'cond', the true and false
// are not paths, but vectors to mix.
//
// Specialization constant operations include
//     - The ternary operator ( ? : )
//
// Returns the selection node created, or nullptr if one could not be.
//
TIntermTyped* TIntermediate::addSelection(TIntermTyped* cond, TIntermTyped* trueBlock, TIntermTyped* falseBlock,
                                          const TSourceLoc& loc)
{
    // If it's void, go to the if-then-else selection()
    if (trueBlock->getBasicType() == EbtVoid && falseBlock->getBasicType() == EbtVoid) {
        TIntermNodePair pair = { trueBlock, falseBlock };
        TIntermSelection* selection = addSelection(cond, pair, loc);
        if (getSource() == EShSourceHlsl)
            selection->setNoShortCirlwit();

        return selection;
    }

    //
    // Get compatible types.
    //
    auto children = addColwersion(EOpSequence, trueBlock, falseBlock);
    trueBlock = std::get<0>(children);
    falseBlock = std::get<1>(children);

    if (trueBlock == nullptr || falseBlock == nullptr)
        return nullptr;

    // Handle a vector condition as a mix
    if (!cond->getType().isScalarOrVec1()) {
        TType targetVectorType(trueBlock->getType().getBasicType(), EvqTemporary,
                               cond->getType().getVectorSize());
        // smear true/false operands as needed
        trueBlock = addUniShapeColwersion(EOpMix, targetVectorType, trueBlock);
        falseBlock = addUniShapeColwersion(EOpMix, targetVectorType, falseBlock);

        // After colwersion, types have to match.
        if (falseBlock->getType() != trueBlock->getType())
            return nullptr;

        // make the mix operation
        TIntermAggregate* mix = makeAggregate(loc);
        mix = growAggregate(mix, falseBlock);
        mix = growAggregate(mix, trueBlock);
        mix = growAggregate(mix, cond);
        mix->setType(targetVectorType);
        mix->setOp(EOpMix);

        return mix;
    }

    // Now have a scalar condition...

    // Colwert true and false expressions to matching types
    addBiShapeColwersion(EOpMix, trueBlock, falseBlock);

    // After colwersion, types have to match.
    if (falseBlock->getType() != trueBlock->getType())
        return nullptr;

    // Eliminate the selection when the condition is a scalar and all operands are constant.
    if (cond->getAsConstantUnion() && trueBlock->getAsConstantUnion() && falseBlock->getAsConstantUnion()) {
        if (cond->getAsConstantUnion()->getConstArray()[0].getBConst())
            return trueBlock;
        else
            return falseBlock;
    }

    //
    // Make a selection node.
    //
    TIntermSelection* node = new TIntermSelection(cond, trueBlock, falseBlock, trueBlock->getType());
    node->setLoc(loc);
    node->getQualifier().precision = std::max(trueBlock->getQualifier().precision, falseBlock->getQualifier().precision);

    if ((cond->getQualifier().isConstant() && specConstantPropagates(*trueBlock, *falseBlock)) ||
        (cond->getQualifier().isSpecConstant() && trueBlock->getQualifier().isConstant() &&
                                                 falseBlock->getQualifier().isConstant()))
        node->getQualifier().makeSpecConstant();
    else
        node->getQualifier().makeTemporary();

    if (getSource() == EShSourceHlsl)
        node->setNoShortCirlwit();

    return node;
}

//
// Constant terminal nodes.  Has a union that contains bool, float or int constants
//
// Returns the constant union node created.
//

TIntermConstantUnion* TIntermediate::addConstantUnion(const TConstUnionArray& unionArray, const TType& t, const TSourceLoc& loc, bool literal) const
{
    TIntermConstantUnion* node = new TIntermConstantUnion(unionArray, t);
    node->getQualifier().storage = EvqConst;
    node->setLoc(loc);
    if (literal)
        node->setLiteral();

    return node;
}
TIntermConstantUnion* TIntermediate::addConstantUnion(signed char i8, const TSourceLoc& loc, bool literal) const
{
    TConstUnionArray unionArray(1);
    unionArray[0].setI8Const(i8);

    return addConstantUnion(unionArray, TType(EbtInt8, EvqConst), loc, literal);
}

TIntermConstantUnion* TIntermediate::addConstantUnion(unsigned char u8, const TSourceLoc& loc, bool literal) const
{
    TConstUnionArray unionArray(1);
    unionArray[0].setUConst(u8);

    return addConstantUnion(unionArray, TType(EbtUint8, EvqConst), loc, literal);
}

TIntermConstantUnion* TIntermediate::addConstantUnion(signed short i16, const TSourceLoc& loc, bool literal) const
{
    TConstUnionArray unionArray(1);
    unionArray[0].setI16Const(i16);

    return addConstantUnion(unionArray, TType(EbtInt16, EvqConst), loc, literal);
}

TIntermConstantUnion* TIntermediate::addConstantUnion(unsigned short u16, const TSourceLoc& loc, bool literal) const
{
    TConstUnionArray unionArray(1);
    unionArray[0].setU16Const(u16);

    return addConstantUnion(unionArray, TType(EbtUint16, EvqConst), loc, literal);
}

TIntermConstantUnion* TIntermediate::addConstantUnion(int i, const TSourceLoc& loc, bool literal) const
{
    TConstUnionArray unionArray(1);
    unionArray[0].setIConst(i);

    return addConstantUnion(unionArray, TType(EbtInt, EvqConst), loc, literal);
}

TIntermConstantUnion* TIntermediate::addConstantUnion(unsigned int u, const TSourceLoc& loc, bool literal) const
{
    TConstUnionArray unionArray(1);
    unionArray[0].setUConst(u);

    return addConstantUnion(unionArray, TType(EbtUint, EvqConst), loc, literal);
}

TIntermConstantUnion* TIntermediate::addConstantUnion(long long i64, const TSourceLoc& loc, bool literal) const
{
    TConstUnionArray unionArray(1);
    unionArray[0].setI64Const(i64);

    return addConstantUnion(unionArray, TType(EbtInt64, EvqConst), loc, literal);
}

TIntermConstantUnion* TIntermediate::addConstantUnion(unsigned long long u64, const TSourceLoc& loc, bool literal) const
{
    TConstUnionArray unionArray(1);
    unionArray[0].setU64Const(u64);

    return addConstantUnion(unionArray, TType(EbtUint64, EvqConst), loc, literal);
}

TIntermConstantUnion* TIntermediate::addConstantUnion(bool b, const TSourceLoc& loc, bool literal) const
{
    TConstUnionArray unionArray(1);
    unionArray[0].setBConst(b);

    return addConstantUnion(unionArray, TType(EbtBool, EvqConst), loc, literal);
}

TIntermConstantUnion* TIntermediate::addConstantUnion(double d, TBasicType baseType, const TSourceLoc& loc, bool literal) const
{
    assert(baseType == EbtFloat || baseType == EbtDouble || baseType == EbtFloat16);

    TConstUnionArray unionArray(1);
    unionArray[0].setDConst(d);

    return addConstantUnion(unionArray, TType(baseType, EvqConst), loc, literal);
}

TIntermConstantUnion* TIntermediate::addConstantUnion(const TString* s, const TSourceLoc& loc, bool literal) const
{
    TConstUnionArray unionArray(1);
    unionArray[0].setSConst(s);

    return addConstantUnion(unionArray, TType(EbtString, EvqConst), loc, literal);
}

// Put vector swizzle selectors onto the given sequence
void TIntermediate::pushSelector(TIntermSequence& sequence, const TVectorSelector& selector, const TSourceLoc& loc)
{
    TIntermConstantUnion* constIntNode = addConstantUnion(selector, loc);
    sequence.push_back(constIntNode);
}

// Put matrix swizzle selectors onto the given sequence
void TIntermediate::pushSelector(TIntermSequence& sequence, const TMatrixSelector& selector, const TSourceLoc& loc)
{
    TIntermConstantUnion* constIntNode = addConstantUnion(selector.coord1, loc);
    sequence.push_back(constIntNode);
    constIntNode = addConstantUnion(selector.coord2, loc);
    sequence.push_back(constIntNode);
}

// Make an aggregate node that has a sequence of all selectors.
template TIntermTyped* TIntermediate::addSwizzle<TVectorSelector>(TSwizzleSelectors<TVectorSelector>& selector, const TSourceLoc& loc);
template TIntermTyped* TIntermediate::addSwizzle<TMatrixSelector>(TSwizzleSelectors<TMatrixSelector>& selector, const TSourceLoc& loc);
template<typename selectorType>
TIntermTyped* TIntermediate::addSwizzle(TSwizzleSelectors<selectorType>& selector, const TSourceLoc& loc)
{
    TIntermAggregate* node = new TIntermAggregate(EOpSequence);

    node->setLoc(loc);
    TIntermSequence &sequenceVector = node->getSequence();

    for (int i = 0; i < selector.size(); i++)
        pushSelector(sequenceVector, selector[i], loc);

    return node;
}

//
// Follow the left branches down to the root of an l-value
// expression (just "." and []).
//
// Return the base of the l-value (where following indexing quits working).
// Return nullptr if a chain following dereferences cannot be followed.
//
// 'swizzleOkay' says whether or not it is okay to consider a swizzle
// a valid part of the dereference chain.
//
const TIntermTyped* TIntermediate::findLValueBase(const TIntermTyped* node, bool swizzleOkay)
{
    do {
        const TIntermBinary* binary = node->getAsBinaryNode();
        if (binary == nullptr)
            return node;
        TOperator op = binary->getOp();
        if (op != EOpIndexDirect && op != EOpIndexIndirect && op != EOpIndexDirectStruct && op != EOpVectorSwizzle && op != EOpMatrixSwizzle)
            return nullptr;
        if (! swizzleOkay) {
            if (op == EOpVectorSwizzle || op == EOpMatrixSwizzle)
                return nullptr;
            if ((op == EOpIndexDirect || op == EOpIndexIndirect) &&
                (binary->getLeft()->getType().isVector() || binary->getLeft()->getType().isScalar()) &&
                ! binary->getLeft()->getType().isArray())
                return nullptr;
        }
        node = node->getAsBinaryNode()->getLeft();
    } while (true);
}

//
// Create while and do-while loop nodes.
//
TIntermLoop* TIntermediate::addLoop(TIntermNode* body, TIntermTyped* test, TIntermTyped* terminal, bool testFirst,
    const TSourceLoc& loc)
{
    TIntermLoop* node = new TIntermLoop(body, test, terminal, testFirst);
    node->setLoc(loc);

    return node;
}

//
// Create a for-loop sequence.
//
TIntermAggregate* TIntermediate::addForLoop(TIntermNode* body, TIntermNode* initializer, TIntermTyped* test,
    TIntermTyped* terminal, bool testFirst, const TSourceLoc& loc, TIntermLoop*& node)
{
    node = new TIntermLoop(body, test, terminal, testFirst);
    node->setLoc(loc);

    // make a sequence of the initializer and statement, but try to reuse the
    // aggregate already created for whatever is in the initializer, if there is one
    TIntermAggregate* loopSequence = (initializer == nullptr ||
                                      initializer->getAsAggregate() == nullptr) ? makeAggregate(initializer, loc)
                                                                                : initializer->getAsAggregate();
    if (loopSequence != nullptr && loopSequence->getOp() == EOpSequence)
        loopSequence->setOp(EOpNull);
    loopSequence = growAggregate(loopSequence, node);
    loopSequence->setOperator(EOpSequence);

    return loopSequence;
}

//
// Add branches.
//
TIntermBranch* TIntermediate::addBranch(TOperator branchOp, const TSourceLoc& loc)
{
    return addBranch(branchOp, nullptr, loc);
}

TIntermBranch* TIntermediate::addBranch(TOperator branchOp, TIntermTyped* expression, const TSourceLoc& loc)
{
    TIntermBranch* node = new TIntermBranch(branchOp, expression);
    node->setLoc(loc);

    return node;
}

//
// This is to be exelwted after the final root is put on top by the parsing
// process.
//
bool TIntermediate::postProcess(TIntermNode* root, EShLanguage /*language*/)
{
    if (root == nullptr)
        return true;

    // Finish off the top-level sequence
    TIntermAggregate* aggRoot = root->getAsAggregate();
    if (aggRoot && aggRoot->getOp() == EOpNull)
        aggRoot->setOperator(EOpSequence);

#ifndef GLSLANG_WEB
    // Propagate 'noContraction' label in backward from 'precise' variables.
    glslang::PropagateNoContraction(*this);

    switch (textureSamplerTransformMode) {
    case EShTexSampTransKeep:
        break;
    case EShTexSampTransUpgradeTextureRemoveSampler:
        performTextureUpgradeAndSamplerRemovalTransformation(root);
        break;
    case EShTexSampTransCount:
        assert(0);
        break;
    }
#endif

    return true;
}

void TIntermediate::addSymbolLinkageNodes(TIntermAggregate*& linkage, EShLanguage language, TSymbolTable& symbolTable)
{
    // Add top-level nodes for declarations that must be checked cross
    // compilation unit by a linker, yet might not have been referenced
    // by the AST.
    //
    // Almost entirely, translation of symbols is driven by what's present
    // in the AST traversal, not by translating the symbol table.
    //
    // However, there are some special cases:
    //  - From the specification: "Special built-in inputs gl_VertexID and
    //    gl_InstanceID are also considered active vertex attributes."
    //  - Linker-based type mismatch error reporting needs to see all
    //    uniforms/ins/outs variables and blocks.
    //  - ftransform() can make gl_Vertex and gl_ModelViewProjectionMatrix active.
    //

    // if (ftransformUsed) {
        // TODO: 1.1 lowering functionality: track ftransform() usage
    //    addSymbolLinkageNode(root, symbolTable, "gl_Vertex");
    //    addSymbolLinkageNode(root, symbolTable, "gl_ModelViewProjectionMatrix");
    //}

    if (language == EShLangVertex) {
        // the names won't be found in the symbol table unless the versions are right,
        // so version logic does not need to be repeated here
        addSymbolLinkageNode(linkage, symbolTable, "gl_VertexID");
        addSymbolLinkageNode(linkage, symbolTable, "gl_InstanceID");
    }

    // Add a child to the root node for the linker objects
    linkage->setOperator(EOpLinkerObjects);
    treeRoot = growAggregate(treeRoot, linkage);
}

//
// Add the given name or symbol to the list of nodes at the end of the tree used
// for link-time checking and external linkage.
//

void TIntermediate::addSymbolLinkageNode(TIntermAggregate*& linkage, TSymbolTable& symbolTable, const TString& name)
{
    TSymbol* symbol = symbolTable.find(name);
    if (symbol)
        addSymbolLinkageNode(linkage, *symbol->getAsVariable());
}

void TIntermediate::addSymbolLinkageNode(TIntermAggregate*& linkage, const TSymbol& symbol)
{
    const TVariable* variable = symbol.getAsVariable();
    if (! variable) {
        // This must be a member of an anonymous block, and we need to add the whole block
        const TAnonMember* anon = symbol.getAsAnonMember();
        variable = &anon->getAnonContainer();
    }
    TIntermSymbol* node = addSymbol(*variable);
    linkage = growAggregate(linkage, node);
}

//
// Add a caller->callee relationship to the call graph.
// Assumes the strings are unique per signature.
//
void TIntermediate::addToCallGraph(TInfoSink& /*infoSink*/, const TString& caller, const TString& callee)
{
    // Duplicates are okay, but faster to not keep them, and they come grouped by caller,
    // as long as new ones are push on the same end we check on for duplicates
    for (TGraph::const_iterator call = callGraph.begin(); call != callGraph.end(); ++call) {
        if (call->caller != caller)
            break;
        if (call->callee == callee)
            return;
    }

    callGraph.push_front(TCall(caller, callee));
}

//
// This deletes the tree.
//
void TIntermediate::removeTree()
{
    if (treeRoot)
        RemoveAllTreeNodes(treeRoot);
}

//
// Implement the part of KHR_vulkan_glsl that lists the set of operations
// that can result in a specialization constant operation.
//
// "5.x Specialization Constant Operations"
//
//    Only some operations dislwssed in this section may be applied to a
//    specialization constant and still yield a result that is as
//    specialization constant.  The operations allowed are listed below.
//    When a specialization constant is operated on with one of these
//    operators and with another constant or specialization constant, the
//    result is implicitly a specialization constant.
//
//     - int(), uint(), and bool() constructors for type colwersions
//       from any of the following types to any of the following types:
//         * int
//         * uint
//         * bool
//     - vector versions of the above colwersion constructors
//     - allowed implicit colwersions of the above
//     - swizzles (e.g., foo.yx)
//     - The following when applied to integer or unsigned integer types:
//         * unary negative ( - )
//         * binary operations ( + , - , * , / , % )
//         * shift ( <<, >> )
//         * bitwise operations ( & , | , ^ )
//     - The following when applied to integer or unsigned integer scalar types:
//         * comparison ( == , != , > , >= , < , <= )
//     - The following when applied to the Boolean scalar type:
//         * not ( ! )
//         * logical operations ( && , || , ^^ )
//         * comparison ( == , != )"
//
// This function just handles binary and unary nodes.  Construction
// rules are handled in construction paths that are not covered by the unary
// and binary paths, while required colwersions will still show up here
// as unary colwerters in the from a construction operator.
//
bool TIntermediate::isSpecializationOperation(const TIntermOperator& node) const
{
    // The operations resulting in floating point are quite limited
    // (However, some floating-point operations result in bool, like ">",
    // so are handled later.)
    if (node.getType().isFloatingDomain()) {
        switch (node.getOp()) {
        case EOpIndexDirect:
        case EOpIndexIndirect:
        case EOpIndexDirectStruct:
        case EOpVectorSwizzle:
        case EOpColwFloatToDouble:
        case EOpColwDoubleToFloat:
        case EOpColwFloat16ToFloat:
        case EOpColwFloatToFloat16:
        case EOpColwFloat16ToDouble:
        case EOpColwDoubleToFloat16:
            return true;
        default:
            return false;
        }
    }

    // Check for floating-point arguments
    if (const TIntermBinary* bin = node.getAsBinaryNode())
        if (bin->getLeft() ->getType().isFloatingDomain() ||
            bin->getRight()->getType().isFloatingDomain())
            return false;

    // So, for now, we can assume everything left is non-floating-point...

    // Now check for integer/bool-based operations
    switch (node.getOp()) {

    // dereference/swizzle
    case EOpIndexDirect:
    case EOpIndexIndirect:
    case EOpIndexDirectStruct:
    case EOpVectorSwizzle:

    // (u)int* -> bool
    case EOpColwInt8ToBool:
    case EOpColwInt16ToBool:
    case EOpColwIntToBool:
    case EOpColwInt64ToBool:
    case EOpColwUint8ToBool:
    case EOpColwUint16ToBool:
    case EOpColwUintToBool:
    case EOpColwUint64ToBool:

    // bool -> (u)int*
    case EOpColwBoolToInt8:
    case EOpColwBoolToInt16:
    case EOpColwBoolToInt:
    case EOpColwBoolToInt64:
    case EOpColwBoolToUint8:
    case EOpColwBoolToUint16:
    case EOpColwBoolToUint:
    case EOpColwBoolToUint64:

    // int8_t -> (u)int*
    case EOpColwInt8ToInt16:
    case EOpColwInt8ToInt:
    case EOpColwInt8ToInt64:
    case EOpColwInt8ToUint8:
    case EOpColwInt8ToUint16:
    case EOpColwInt8ToUint:
    case EOpColwInt8ToUint64:

    // int16_t -> (u)int*
    case EOpColwInt16ToInt8:
    case EOpColwInt16ToInt:
    case EOpColwInt16ToInt64:
    case EOpColwInt16ToUint8:
    case EOpColwInt16ToUint16:
    case EOpColwInt16ToUint:
    case EOpColwInt16ToUint64:

    // int32_t -> (u)int*
    case EOpColwIntToInt8:
    case EOpColwIntToInt16:
    case EOpColwIntToInt64:
    case EOpColwIntToUint8:
    case EOpColwIntToUint16:
    case EOpColwIntToUint:
    case EOpColwIntToUint64:

    // int64_t -> (u)int*
    case EOpColwInt64ToInt8:
    case EOpColwInt64ToInt16:
    case EOpColwInt64ToInt:
    case EOpColwInt64ToUint8:
    case EOpColwInt64ToUint16:
    case EOpColwInt64ToUint:
    case EOpColwInt64ToUint64:

    // uint8_t -> (u)int*
    case EOpColwUint8ToInt8:
    case EOpColwUint8ToInt16:
    case EOpColwUint8ToInt:
    case EOpColwUint8ToInt64:
    case EOpColwUint8ToUint16:
    case EOpColwUint8ToUint:
    case EOpColwUint8ToUint64:

    // uint16_t -> (u)int*
    case EOpColwUint16ToInt8:
    case EOpColwUint16ToInt16:
    case EOpColwUint16ToInt:
    case EOpColwUint16ToInt64:
    case EOpColwUint16ToUint8:
    case EOpColwUint16ToUint:
    case EOpColwUint16ToUint64:

    // uint32_t -> (u)int*
    case EOpColwUintToInt8:
    case EOpColwUintToInt16:
    case EOpColwUintToInt:
    case EOpColwUintToInt64:
    case EOpColwUintToUint8:
    case EOpColwUintToUint16:
    case EOpColwUintToUint64:

    // uint64_t -> (u)int*
    case EOpColwUint64ToInt8:
    case EOpColwUint64ToInt16:
    case EOpColwUint64ToInt:
    case EOpColwUint64ToInt64:
    case EOpColwUint64ToUint8:
    case EOpColwUint64ToUint16:
    case EOpColwUint64ToUint:

    // unary operations
    case EOpNegative:
    case EOpLogicalNot:
    case EOpBitwiseNot:

    // binary operations
    case EOpAdd:
    case EOpSub:
    case EOpMul:
    case EOpVectorTimesScalar:
    case EOpDiv:
    case EOpMod:
    case EOpRightShift:
    case EOpLeftShift:
    case EOpAnd:
    case EOpInclusiveOr:
    case EOpExclusiveOr:
    case EOpLogicalOr:
    case EOpLogicalXor:
    case EOpLogicalAnd:
    case EOpEqual:
    case EOpNotEqual:
    case EOpLessThan:
    case EOpGreaterThan:
    case EOpLessThanEqual:
    case EOpGreaterThanEqual:
        return true;
    default:
        return false;
    }
}

// Is the operation one that must propagate nonuniform?
bool TIntermediate::isNonuniformPropagating(TOperator op) const
{
    // "* All Operators in Section 5.1 (Operators), except for assignment,
    //    arithmetic assignment, and sequence
    //  * Component selection in Section 5.5
    //  * Matrix components in Section 5.6
    //  * Structure and Array Operations in Section 5.7, except for the length
    //    method."
    switch (op) {
    case EOpPostIncrement:
    case EOpPostDecrement:
    case EOpPreIncrement:
    case EOpPreDecrement:

    case EOpNegative:
    case EOpLogicalNot:
    case EOpVectorLogicalNot:
    case EOpBitwiseNot:

    case EOpAdd:
    case EOpSub:
    case EOpMul:
    case EOpDiv:
    case EOpMod:
    case EOpRightShift:
    case EOpLeftShift:
    case EOpAnd:
    case EOpInclusiveOr:
    case EOpExclusiveOr:
    case EOpEqual:
    case EOpNotEqual:
    case EOpLessThan:
    case EOpGreaterThan:
    case EOpLessThanEqual:
    case EOpGreaterThanEqual:
    case EOpVectorTimesScalar:
    case EOpVectorTimesMatrix:
    case EOpMatrixTimesVector:
    case EOpMatrixTimesScalar:

    case EOpLogicalOr:
    case EOpLogicalXor:
    case EOpLogicalAnd:

    case EOpIndexDirect:
    case EOpIndexIndirect:
    case EOpIndexDirectStruct:
    case EOpVectorSwizzle:
        return true;

    default:
        break;
    }

    return false;
}

////////////////////////////////////////////////////////////////
//
// Member functions of the nodes used for building the tree.
//
////////////////////////////////////////////////////////////////

//
// Say whether or not an operation node changes the value of a variable.
//
// Returns true if state is modified.
//
bool TIntermOperator::modifiesState() const
{
    switch (op) {
    case EOpPostIncrement:
    case EOpPostDecrement:
    case EOpPreIncrement:
    case EOpPreDecrement:
    case EOpAssign:
    case EOpAddAssign:
    case EOpSubAssign:
    case EOpMulAssign:
    case EOpVectorTimesMatrixAssign:
    case EOpVectorTimesScalarAssign:
    case EOpMatrixTimesScalarAssign:
    case EOpMatrixTimesMatrixAssign:
    case EOpDivAssign:
    case EOpModAssign:
    case EOpAndAssign:
    case EOpInclusiveOrAssign:
    case EOpExclusiveOrAssign:
    case EOpLeftShiftAssign:
    case EOpRightShiftAssign:
        return true;
    default:
        return false;
    }
}

//
// returns true if the operator is for one of the constructors
//
bool TIntermOperator::isConstructor() const
{
    return op > EOpConstructGuardStart && op < EOpConstructGuardEnd;
}

//
// Make sure the type of an operator is appropriate for its
// combination of operation and operand type.  This will ilwoke
// promoteUnary, promoteBinary, etc as needed.
//
// Returns false if nothing makes sense.
//
bool TIntermediate::promote(TIntermOperator* node)
{
    if (node == nullptr)
        return false;

    if (node->getAsUnaryNode())
        return promoteUnary(*node->getAsUnaryNode());

    if (node->getAsBinaryNode())
        return promoteBinary(*node->getAsBinaryNode());

    if (node->getAsAggregate())
        return promoteAggregate(*node->getAsAggregate());

    return false;
}

//
// See TIntermediate::promote
//
bool TIntermediate::promoteUnary(TIntermUnary& node)
{
    const TOperator op    = node.getOp();
    TIntermTyped* operand = node.getOperand();

    switch (op) {
    case EOpLogicalNot:
        // Colwert operand to a boolean type
        if (operand->getBasicType() != EbtBool) {
            // Add constructor to boolean type. If that fails, we can't do it, so return false.
            TIntermTyped* colwerted = addColwersion(op, TType(EbtBool), operand);
            if (colwerted == nullptr)
                return false;

            // Use the result of colwerting the node to a bool.
            node.setOperand(operand = colwerted); // also updates stack variable
        }
        break;
    case EOpBitwiseNot:
        if (!isTypeInt(operand->getBasicType()))
            return false;
        break;
    case EOpNegative:
    case EOpPostIncrement:
    case EOpPostDecrement:
    case EOpPreIncrement:
    case EOpPreDecrement:
        if (!isTypeInt(operand->getBasicType()) &&
            operand->getBasicType() != EbtFloat &&
            operand->getBasicType() != EbtFloat16 &&
            operand->getBasicType() != EbtDouble)

            return false;
        break;
    default:
        // HLSL uses this path for initial function signature finding for built-ins
        // taking a single argument, which generally don't participate in
        // operator-based type promotion (type colwersion will occur later).
        // For now, scalar argument cases are relying on the setType() call below.
        if (getSource() == EShSourceHlsl)
            break;

        // GLSL only allows integer arguments for the cases identified above in the
        // case statements.
        if (operand->getBasicType() != EbtFloat)
            return false;
    }

    node.setType(operand->getType());
    node.getWritableType().getQualifier().makeTemporary();

    return true;
}

void TIntermUnary::updatePrecision()
{
    if (getBasicType() == EbtInt || getBasicType() == EbtUint || getBasicType() == EbtFloat || getBasicType() == EbtFloat16) {
        if (operand->getQualifier().precision > getQualifier().precision)
            getQualifier().precision = operand->getQualifier().precision;
    }
}

//
// See TIntermediate::promote
//
bool TIntermediate::promoteBinary(TIntermBinary& node)
{
    TOperator     op    = node.getOp();
    TIntermTyped* left  = node.getLeft();
    TIntermTyped* right = node.getRight();

    // Arrays and structures have to be exact matches.
    if ((left->isArray() || right->isArray() || left->getBasicType() == EbtStruct || right->getBasicType() == EbtStruct)
        && left->getType() != right->getType())
        return false;

    // Base assumption:  just make the type the same as the left
    // operand.  Only deviations from this will be coded.
    node.setType(left->getType());
    node.getWritableType().getQualifier().clear();

    // Composite and opaque types don't having pending operator changes, e.g.,
    // array, structure, and samplers.  Just establish final type and correctness.
    if (left->isArray() || left->getBasicType() == EbtStruct || left->getBasicType() == EbtSampler) {
        switch (op) {
        case EOpEqual:
        case EOpNotEqual:
            if (left->getBasicType() == EbtSampler) {
                // can't compare samplers
                return false;
            } else {
                // Promote to conditional
                node.setType(TType(EbtBool));
            }

            return true;

        case EOpAssign:
            // Keep type from above

            return true;

        default:
            return false;
        }
    }

    //
    // We now have only scalars, vectors, and matrices to worry about.
    //

    // HLSL implicitly promotes bool -> int for numeric operations.
    // (Implicit colwersions to make the operands match each other's types were already done.)
    if (getSource() == EShSourceHlsl &&
        (left->getBasicType() == EbtBool || right->getBasicType() == EbtBool)) {
        switch (op) {
        case EOpLessThan:
        case EOpGreaterThan:
        case EOpLessThanEqual:
        case EOpGreaterThanEqual:

        case EOpRightShift:
        case EOpLeftShift:

        case EOpMod:

        case EOpAnd:
        case EOpInclusiveOr:
        case EOpExclusiveOr:

        case EOpAdd:
        case EOpSub:
        case EOpDiv:
        case EOpMul:
            if (left->getBasicType() == EbtBool)
                left  = createColwersion(EbtInt, left);
            if (right->getBasicType() == EbtBool)
                right = createColwersion(EbtInt, right);
            if (left == nullptr || right == nullptr)
                return false;
            node.setLeft(left);
            node.setRight(right);

            // Update the original base assumption on result type..
            node.setType(left->getType());
            node.getWritableType().getQualifier().clear();

            break;

        default:
            break;
        }
    }

    // Do general type checks against individual operands (comparing left and right is coming up, checking mixed shapes after that)
    switch (op) {
    case EOpLessThan:
    case EOpGreaterThan:
    case EOpLessThanEqual:
    case EOpGreaterThanEqual:
        // Relational comparisons need numeric types and will promote to scalar Boolean.
        if (left->getBasicType() == EbtBool)
            return false;

        node.setType(TType(EbtBool, EvqTemporary, left->getVectorSize()));
        break;

    case EOpEqual:
    case EOpNotEqual:
        if (getSource() == EShSourceHlsl) {
            const int resultWidth = std::max(left->getVectorSize(), right->getVectorSize());

            // In HLSL, == or != on vectors means component-wise comparison.
            if (resultWidth > 1) {
                op = (op == EOpEqual) ? EOpVectorEqual : EOpVectorNotEqual;
                node.setOp(op);
            }

            node.setType(TType(EbtBool, EvqTemporary, resultWidth));
        } else {
            // All the above comparisons result in a bool (but not the vector compares)
            node.setType(TType(EbtBool));
        }
        break;

    case EOpLogicalAnd:
    case EOpLogicalOr:
    case EOpLogicalXor:
        // logical ops operate only on Booleans or vectors of Booleans.
        if (left->getBasicType() != EbtBool || left->isMatrix())
                return false;

        if (getSource() == EShSourceGlsl) {
            // logical ops operate only on scalar Booleans and will promote to scalar Boolean.
            if (left->isVector())
                return false;
        }

        node.setType(TType(EbtBool, EvqTemporary, left->getVectorSize()));
        break;

    case EOpRightShift:
    case EOpLeftShift:
    case EOpRightShiftAssign:
    case EOpLeftShiftAssign:

    case EOpMod:
    case EOpModAssign:

    case EOpAnd:
    case EOpInclusiveOr:
    case EOpExclusiveOr:
    case EOpAndAssign:
    case EOpInclusiveOrAssign:
    case EOpExclusiveOrAssign:
        if (getSource() == EShSourceHlsl)
            break;

        // Check for integer-only operands.
        if (!isTypeInt(left->getBasicType()) && !isTypeInt(right->getBasicType()))
            return false;
        if (left->isMatrix() || right->isMatrix())
            return false;

        break;

    case EOpAdd:
    case EOpSub:
    case EOpDiv:
    case EOpMul:
    case EOpAddAssign:
    case EOpSubAssign:
    case EOpMulAssign:
    case EOpDivAssign:
        // check for non-Boolean operands
        if (left->getBasicType() == EbtBool || right->getBasicType() == EbtBool)
            return false;

    default:
        break;
    }

    // Compare left and right, and finish with the cases where the operand types must match
    switch (op) {
    case EOpLessThan:
    case EOpGreaterThan:
    case EOpLessThanEqual:
    case EOpGreaterThanEqual:

    case EOpEqual:
    case EOpNotEqual:
    case EOpVectorEqual:
    case EOpVectorNotEqual:

    case EOpLogicalAnd:
    case EOpLogicalOr:
    case EOpLogicalXor:
        return left->getType() == right->getType();

    case EOpMod:
    case EOpModAssign:

    case EOpAnd:
    case EOpInclusiveOr:
    case EOpExclusiveOr:
    case EOpAndAssign:
    case EOpInclusiveOrAssign:
    case EOpExclusiveOrAssign:

    case EOpAdd:
    case EOpSub:
    case EOpDiv:

    case EOpAddAssign:
    case EOpSubAssign:
    case EOpDivAssign:
        // Quick out in case the types do match
        if (left->getType() == right->getType())
            return true;

        // Fall through

    case EOpMul:
    case EOpMulAssign:
        // At least the basic type has to match
        if (left->getBasicType() != right->getBasicType())
            return false;

    default:
        break;
    }

    if (left->getType().isCoopMat() || right->getType().isCoopMat()) {
        if (left->getType().isCoopMat() && right->getType().isCoopMat() &&
            *left->getType().getTypeParameters() != *right->getType().getTypeParameters()) {
            return false;
        }
        switch (op) {
        case EOpMul:
        case EOpMulAssign:
            if (left->getType().isCoopMat() && right->getType().isCoopMat()) {
                return false;
            }
            if (op == EOpMulAssign && right->getType().isCoopMat()) {
                return false;
            }
            node.setOp(op == EOpMulAssign ? EOpMatrixTimesScalarAssign : EOpMatrixTimesScalar);
            if (right->getType().isCoopMat()) {
                node.setType(right->getType());
            }
            return true;
        case EOpAdd:
        case EOpSub:
        case EOpDiv:
        case EOpAssign:
            // These require both to be cooperative matrices
            if (!left->getType().isCoopMat() || !right->getType().isCoopMat()) {
                return false;
            }
            return true;
        default:
            break;
        }
        return false;
    }

    // Finish handling the case, for all ops, where both operands are scalars.
    if (left->isScalar() && right->isScalar())
        return true;

    // Finish handling the case, for all ops, where there are two vectors of different sizes
    if (left->isVector() && right->isVector() && left->getVectorSize() != right->getVectorSize() && right->getVectorSize() > 1)
        return false;

    //
    // We now have a mix of scalars, vectors, or matrices, for non-relational operations.
    //

    // Can these two operands be combined, what is the resulting type?
    TBasicType basicType = left->getBasicType();
    switch (op) {
    case EOpMul:
        if (!left->isMatrix() && right->isMatrix()) {
            if (left->isVector()) {
                if (left->getVectorSize() != right->getMatrixRows())
                    return false;
                node.setOp(op = EOpVectorTimesMatrix);
                node.setType(TType(basicType, EvqTemporary, right->getMatrixCols()));
            } else {
                node.setOp(op = EOpMatrixTimesScalar);
                node.setType(TType(basicType, EvqTemporary, 0, right->getMatrixCols(), right->getMatrixRows()));
            }
        } else if (left->isMatrix() && !right->isMatrix()) {
            if (right->isVector()) {
                if (left->getMatrixCols() != right->getVectorSize())
                    return false;
                node.setOp(op = EOpMatrixTimesVector);
                node.setType(TType(basicType, EvqTemporary, left->getMatrixRows()));
            } else {
                node.setOp(op = EOpMatrixTimesScalar);
            }
        } else if (left->isMatrix() && right->isMatrix()) {
            if (left->getMatrixCols() != right->getMatrixRows())
                return false;
            node.setOp(op = EOpMatrixTimesMatrix);
            node.setType(TType(basicType, EvqTemporary, 0, right->getMatrixCols(), left->getMatrixRows()));
        } else if (! left->isMatrix() && ! right->isMatrix()) {
            if (left->isVector() && right->isVector()) {
                ; // leave as component product
            } else if (left->isVector() || right->isVector()) {
                node.setOp(op = EOpVectorTimesScalar);
                if (right->isVector())
                    node.setType(TType(basicType, EvqTemporary, right->getVectorSize()));
            }
        } else {
            return false;
        }
        break;
    case EOpMulAssign:
        if (! left->isMatrix() && right->isMatrix()) {
            if (left->isVector()) {
                if (left->getVectorSize() != right->getMatrixRows() || left->getVectorSize() != right->getMatrixCols())
                    return false;
                node.setOp(op = EOpVectorTimesMatrixAssign);
            } else {
                return false;
            }
        } else if (left->isMatrix() && !right->isMatrix()) {
            if (right->isVector()) {
                return false;
            } else {
                node.setOp(op = EOpMatrixTimesScalarAssign);
            }
        } else if (left->isMatrix() && right->isMatrix()) {
            if (left->getMatrixCols() != right->getMatrixCols() || left->getMatrixCols() != right->getMatrixRows())
                return false;
            node.setOp(op = EOpMatrixTimesMatrixAssign);
        } else if (!left->isMatrix() && !right->isMatrix()) {
            if (left->isVector() && right->isVector()) {
                // leave as component product
            } else if (left->isVector() || right->isVector()) {
                if (! left->isVector())
                    return false;
                node.setOp(op = EOpVectorTimesScalarAssign);
            }
        } else {
            return false;
        }
        break;

    case EOpRightShift:
    case EOpLeftShift:
    case EOpRightShiftAssign:
    case EOpLeftShiftAssign:
        if (right->isVector() && (! left->isVector() || right->getVectorSize() != left->getVectorSize()))
            return false;
        break;

    case EOpAssign:
        if (left->getVectorSize() != right->getVectorSize() || left->getMatrixCols() != right->getMatrixCols() || left->getMatrixRows() != right->getMatrixRows())
            return false;
        // fall through

    case EOpAdd:
    case EOpSub:
    case EOpDiv:
    case EOpMod:
    case EOpAnd:
    case EOpInclusiveOr:
    case EOpExclusiveOr:
    case EOpAddAssign:
    case EOpSubAssign:
    case EOpDivAssign:
    case EOpModAssign:
    case EOpAndAssign:
    case EOpInclusiveOrAssign:
    case EOpExclusiveOrAssign:

        if ((left->isMatrix() && right->isVector()) ||
            (left->isVector() && right->isMatrix()) ||
            left->getBasicType() != right->getBasicType())
            return false;
        if (left->isMatrix() && right->isMatrix() && (left->getMatrixCols() != right->getMatrixCols() || left->getMatrixRows() != right->getMatrixRows()))
            return false;
        if (left->isVector() && right->isVector() && left->getVectorSize() != right->getVectorSize())
            return false;
        if (right->isVector() || right->isMatrix()) {
            node.getWritableType().shallowCopy(right->getType());
            node.getWritableType().getQualifier().makeTemporary();
        }
        break;

    default:
        return false;
    }

    //
    // One more check for assignment.
    //
    switch (op) {
    // The resulting type has to match the left operand.
    case EOpAssign:
    case EOpAddAssign:
    case EOpSubAssign:
    case EOpMulAssign:
    case EOpDivAssign:
    case EOpModAssign:
    case EOpAndAssign:
    case EOpInclusiveOrAssign:
    case EOpExclusiveOrAssign:
    case EOpLeftShiftAssign:
    case EOpRightShiftAssign:
        if (node.getType() != left->getType())
            return false;
        break;
    default:
        break;
    }

    return true;
}

//
// See TIntermediate::promote
//
bool TIntermediate::promoteAggregate(TIntermAggregate& node)
{
    TOperator op = node.getOp();
    TIntermSequence& args = node.getSequence();
    const int numArgs = static_cast<int>(args.size());

    // Presently, only hlsl does intrinsic promotions.
    if (getSource() != EShSourceHlsl)
        return true;

    // set of opcodes that can be promoted in this manner.
    switch (op) {
    case EOpAtan:
    case EOpClamp:
    case EOpCross:
    case EOpDistance:
    case EOpDot:
    case EOpDst:
    case EOpFaceForward:
    // case EOpFindMSB: TODO:
    // case EOpFindLSB: TODO:
    case EOpFma:
    case EOpMod:
    case EOpFrexp:
    case EOpLdexp:
    case EOpMix:
    case EOpLit:
    case EOpMax:
    case EOpMin:
    case EOpModf:
    // case EOpGenMul: TODO:
    case EOpPow:
    case EOpReflect:
    case EOpRefract:
    // case EOpSinCos: TODO:
    case EOpSmoothStep:
    case EOpStep:
        break;
    default:
        return true;
    }

    // TODO: array and struct behavior

    // Try colwerting all nodes to the given node's type
    TIntermSequence colwertedArgs(numArgs, nullptr);

    // Try to colwert all types to the nonColwArg type.
    for (int nonColwArg = 0; nonColwArg < numArgs; ++nonColwArg) {
        // Try colwerting all args to this arg's type
        for (int colwArg = 0; colwArg < numArgs; ++colwArg) {
            colwertedArgs[colwArg] = addColwersion(op, args[nonColwArg]->getAsTyped()->getType(),
                                                   args[colwArg]->getAsTyped());
        }

        // If we successfully colwerted all the args, use the result.
        if (std::all_of(colwertedArgs.begin(), colwertedArgs.end(),
                        [](const TIntermNode* node) { return node != nullptr; })) {

            std::swap(args, colwertedArgs);
            return true;
        }
    }

    return false;
}

void TIntermBinary::updatePrecision()
{
    if (getBasicType() == EbtInt || getBasicType() == EbtUint || getBasicType() == EbtFloat || getBasicType() == EbtFloat16) {
        getQualifier().precision = std::max(right->getQualifier().precision, left->getQualifier().precision);
        if (getQualifier().precision != EpqNone) {
            left->propagatePrecision(getQualifier().precision);
            right->propagatePrecision(getQualifier().precision);
        }
    }
}

void TIntermTyped::propagatePrecision(TPrecisionQualifier newPrecision)
{
    if (getQualifier().precision != EpqNone || (getBasicType() != EbtInt && getBasicType() != EbtUint && getBasicType() != EbtFloat && getBasicType() != EbtFloat16))
        return;

    getQualifier().precision = newPrecision;

    TIntermBinary* binaryNode = getAsBinaryNode();
    if (binaryNode) {
        binaryNode->getLeft()->propagatePrecision(newPrecision);
        binaryNode->getRight()->propagatePrecision(newPrecision);

        return;
    }

    TIntermUnary* unaryNode = getAsUnaryNode();
    if (unaryNode) {
        unaryNode->getOperand()->propagatePrecision(newPrecision);

        return;
    }

    TIntermAggregate* aggregateNode = getAsAggregate();
    if (aggregateNode) {
        TIntermSequence operands = aggregateNode->getSequence();
        for (unsigned int i = 0; i < operands.size(); ++i) {
            TIntermTyped* typedNode = operands[i]->getAsTyped();
            if (! typedNode)
                break;
            typedNode->propagatePrecision(newPrecision);
        }

        return;
    }

    TIntermSelection* selectionNode = getAsSelectionNode();
    if (selectionNode) {
        TIntermTyped* typedNode = selectionNode->getTrueBlock()->getAsTyped();
        if (typedNode) {
            typedNode->propagatePrecision(newPrecision);
            typedNode = selectionNode->getFalseBlock()->getAsTyped();
            if (typedNode)
                typedNode->propagatePrecision(newPrecision);
        }

        return;
    }
}

TIntermTyped* TIntermediate::promoteConstantUnion(TBasicType promoteTo, TIntermConstantUnion* node) const
{
    const TConstUnionArray& rightUnionArray = node->getConstArray();
    int size = node->getType().computeNumComponents();

    TConstUnionArray leftUnionArray(size);

    for (int i=0; i < size; i++) {

#define PROMOTE(Set, CType, Get) leftUnionArray[i].Set(static_cast<CType>(rightUnionArray[i].Get()))
#define PROMOTE_TO_BOOL(Get) leftUnionArray[i].setBConst(rightUnionArray[i].Get() != 0)

#ifdef GLSLANG_WEB
#define TO_ALL(Get)   \
        switch (promoteTo) { \
        case EbtFloat: PROMOTE(setDConst, double, Get); break; \
        case EbtInt: PROMOTE(setIConst, int, Get); break; \
        case EbtUint: PROMOTE(setUConst, unsigned int, Get); break; \
        case EbtBool: PROMOTE_TO_BOOL(Get); break; \
        default: return node; \
        }
#else
#define TO_ALL(Get)   \
        switch (promoteTo) { \
        case EbtFloat16: PROMOTE(setDConst, double, Get); break; \
        case EbtFloat: PROMOTE(setDConst, double, Get); break; \
        case EbtDouble: PROMOTE(setDConst, double, Get); break; \
        case EbtInt8: PROMOTE(setI8Const, char, Get); break; \
        case EbtInt16: PROMOTE(setI16Const, short, Get); break; \
        case EbtInt: PROMOTE(setIConst, int, Get); break; \
        case EbtInt64: PROMOTE(setI64Const, long long, Get); break; \
        case EbtUint8: PROMOTE(setU8Const, unsigned char, Get); break; \
        case EbtUint16: PROMOTE(setU16Const, unsigned short, Get); break; \
        case EbtUint: PROMOTE(setUConst, unsigned int, Get); break; \
        case EbtUint64: PROMOTE(setU64Const, unsigned long long, Get); break; \
        case EbtBool: PROMOTE_TO_BOOL(Get); break; \
        default: return node; \
        }
#endif

        switch (node->getType().getBasicType()) {
        case EbtFloat: TO_ALL(getDConst); break;
        case EbtInt: TO_ALL(getIConst); break;
        case EbtUint: TO_ALL(getUConst); break;
        case EbtBool: TO_ALL(getBConst); break;
#ifndef GLSLANG_WEB
        case EbtFloat16: TO_ALL(getDConst); break;
        case EbtDouble: TO_ALL(getDConst); break;
        case EbtInt8: TO_ALL(getI8Const); break;
        case EbtInt16: TO_ALL(getI16Const); break;
        case EbtInt64: TO_ALL(getI64Const); break;
        case EbtUint8: TO_ALL(getU8Const); break;
        case EbtUint16: TO_ALL(getU16Const); break;
        case EbtUint64: TO_ALL(getU64Const); break;
#endif
        default: return node;
        }
    }

    const TType& t = node->getType();

    return addConstantUnion(leftUnionArray, TType(promoteTo, t.getQualifier().storage, t.getVectorSize(), t.getMatrixCols(), t.getMatrixRows()),
                            node->getLoc());
}

void TIntermAggregate::setPragmaTable(const TPragmaTable& pTable)
{
    assert(pragmaTable == nullptr);
    pragmaTable = new TPragmaTable;
    *pragmaTable = pTable;
}

// If either node is a specialization constant, while the other is
// a constant (or specialization constant), the result is still
// a specialization constant.
bool TIntermediate::specConstantPropagates(const TIntermTyped& node1, const TIntermTyped& node2)
{
    return (node1.getType().getQualifier().isSpecConstant() && node2.getType().getQualifier().isConstant()) ||
           (node2.getType().getQualifier().isSpecConstant() && node1.getType().getQualifier().isConstant());
}

struct TextureUpgradeAndSamplerRemovalTransform : public TIntermTraverser {
    void visitSymbol(TIntermSymbol* symbol) override {
        if (symbol->getBasicType() == EbtSampler && symbol->getType().getSampler().isTexture()) {
            symbol->getWritableType().getSampler().setCombined(true);
        }
    }
    bool visitAggregate(TVisit, TIntermAggregate* ag) override {
        using namespace std;
        TIntermSequence& seq = ag->getSequence();
        TQualifierList& qual = ag->getQualifierList();

        // qual and seq are indexed using the same indices, so we have to modify both in lock-step
        assert(seq.size() == qual.size() || qual.empty());

        size_t write = 0;
        for (size_t i = 0; i < seq.size(); ++i) {
            TIntermSymbol* symbol = seq[i]->getAsSymbolNode();
            if (symbol && symbol->getBasicType() == EbtSampler && symbol->getType().getSampler().isPureSampler()) {
                // remove pure sampler variables
                continue;
            }

            TIntermNode* result = seq[i];

            // replace constructors with sampler/textures
            TIntermAggregate *constructor = seq[i]->getAsAggregate();
            if (constructor && constructor->getOp() == EOpConstructTextureSampler) {
                if (!constructor->getSequence().empty())
                    result = constructor->getSequence()[0];
            }

            // write new node & qualifier
            seq[write] = result;
            if (!qual.empty())
                qual[write] = qual[i];
            write++;
        }

        seq.resize(write);
        if (!qual.empty())
            qual.resize(write);

        return true;
    }
};

void TIntermediate::performTextureUpgradeAndSamplerRemovalTransformation(TIntermNode* root)
{
    TextureUpgradeAndSamplerRemovalTransform transform;
    root->traverse(&transform);
}

const char* TIntermediate::getResourceName(TResourceType res)
{
    switch (res) {
    case EResSampler: return "shift-sampler-binding";
    case EResTexture: return "shift-texture-binding";
    case EResImage:   return "shift-image-binding";
    case EResUbo:     return "shift-UBO-binding";
    case EResSsbo:    return "shift-ssbo-binding";
    case EResUav:     return "shift-uav-binding";
    default:
        assert(0); // internal error: should only be called with valid resource types.
        return nullptr;
    }
}


} // end namespace glslang
