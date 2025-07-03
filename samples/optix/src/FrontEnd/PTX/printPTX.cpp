// Copyright (c) 2017, LWPU CORPORATION.
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#include <FrontEnd/PTX/printPTX.h>

#include <string>

#include <FrontEnd/PTX/PTXStitch/ptxparse/ptxConstructors.h>
#include <FrontEnd/PTX/PTXStitch/ptxparse/ptxIR.h>
#include <FrontEnd/PTX/PTXStitch/ptxparse/ptxMacroUtils.h>
#include "ptxObfuscatedIRdefs.h"

namespace optix {

static const char* const getStorage[] = {"",        ".code",       ".register", ".sregister", ".const",
                                         ".global", ".local",      ".param",    ".shared",    ".surf",
                                         "",        ".texsampler", ""};

static const char* const getCompare[]  = {PTX_COMPARE_TABLE( GET_NAME )};
static const char* const getOperator[] = {PTX_OPERATOR_TABLE( GET_NAME )};

static const char* const getModifier_APRX[]      = {PTX_APRXMOD_TABLE( GET_NAME )};
static const char* const getModifier_FTZ[]       = {PTX_FTZMOD_TABLE( GET_NAME )};
static const char* const getModifier_SAT[]       = {PTX_SATMOD_TABLE( GET_NAME )};
static const char* const getModifier_CC[]        = {PTX_CCMOD_TABLE( GET_NAME )};
static const char* const getModifier_SHAMT[]     = {PTX_SHAMTMOD_TABLE( GET_NAME )};
static const char* const getModifier_SYNC[]      = {PTX_SYNCMOD_TABLE( GET_NAME )};
static const char* const getModifier_BAR[]       = {PTX_BARMOD_TABLE( GET_NAME )};
static const char* const getModifier_TADDR[]     = {PTX_TADDRMOD_TABLE( GET_NAME )};
static const char* const getModifier_ORDER[]     = {PTX_ORDERMOD_TABLE( GET_NAME )};
static const char* const getModifier_NC[]        = {PTX_NCMOD_TABLE( GET_NAME )};
static const char* const getModifier_ROUND[]     = {PTX_ROUNDMOD_TABLE( GET_NAME )};
static const char* const getModifier_TESTP[]     = {PTX_TESTPMOD_TABLE( GET_NAME )};
static const char* const getModifier_CACHEOP[]   = {PTX_CACHEOPMOD_TABLE( GET_NAME )};
static const char* const getModifier_LEVEL[]     = {PTX_LEVELMOD_TABLE( GET_NAME )};
static const char* const getModifier_FLOW[]      = {"", "", ".uni"};  // We do not want the .div modifier
static const char* const getModifier_BRANCH[]    = {PTX_BRANCHMOD_TABLE( GET_NAME )};
static const char* const getModifier_VECTOR[]    = {PTX_VECTORMOD_TABLE( GET_NAME )};
static const char* const getModifier_TEXTURE[]   = {PTX_TEXTUREMOD_TABLE( GET_NAME )};
static const char* const getModifier_COMPONENT[] = {PTX_COMPONENTMOD_TABLE( GET_NAME )};
static const char* const getModifier_QUERY[]     = {PTX_QUERYMOD_TABLE( GET_NAME )};
static const char* const getModifier_VOTE[]      = {PTX_VOTEMOD_TABLE( GET_NAME )};
static const char* const getModifier_CLAMP[]     = {PTX_CLAMPMOD_TABLE( GET_NAME )};
static const char* const getModifier_SHR[]       = {PTX_SHRMOD_TABLE( GET_NAME )};
static const char* const getModifier_VMAD[]      = {PTX_VMADMOD_TABLE( GET_NAME )};
static const char* const getModifier_PRMT[]      = {PTX_PRMTMOD_TABLE( GET_NAME )};
static const char* const getModifier_SHFL[]      = {PTX_SHFLMOD_TABLE( GET_NAME )};
static const char* const getModifier_ENDIS[]     = {PTX_ENDISMOD_TABLE( GET_NAME )};

static const char* const getVideoSpecifier_SEL[] = {PTX_VIDEOSELECTOR_TABLE( GET_NAME )};

static void printPtxType( std::string& result, ptxType t, ptxParsingState parseState )
{
    result.append( getTypeEnumAsString( parseState->parseData->deobfuscatedStringMapPtr, t->kind ) );
}

bool isAtomInstruction( const std::string& name )
{
    return name.find( "optix.ptx.atom" ) == 0;
}

bool isAtomAddInstruction( const std::string& name )
{
    if( isAtomInstruction( name ) )
        return name.find( "add" ) == ( name.size() - 3 );
    return false;
}

void printPtxOpCode( std::string& result, ptxInstruction instr, ptxParsingState parseState )
{
    ptxModifier& mod = instr->modifiers;
    // Emit instruction name and storage type
    result.append( instr->tmplate->name );

    // TODO: ptxRELAXED_MOD is lwrrently suppressed since it is added automatically as a default
    //       for .atom after updating the parser to ISA 6.4 and this confuses the LLVMtoPTX
    //       colwersion and cause a compilation error when compiling to SASS. This needs to be
    //       fixed to be able to support the new .sem modifiers for .atom (.relaxed, .acquire,
    //       .release, .acq_rel)
    // clang-format off
  if( mod.ORDER  && mod.ORDER != ptxRELAXED_MOD ) result.append( getORDERAsString(parseState->parseData->deobfuscatedStringMapPtr, mod.ORDER) ); // Volatile must be before storage kind
  // not sure about other cases, hence limiting to 'atom' only
  if( mod.SCOPE && isAtomInstruction( result ) )
  {
      const char* scope = getSCOPEAsString( mod.SCOPE );
      // As shown in Bug 3525576 this scope modification breaks O6 handling of instructions. Hence skipping default, ie ".gpu".
      if( strcmp( scope, ".gpu" ) != 0 )
          result.append( scope );
  }
  if( instr->storage && instr->storage->kind )    result.append( getStorage[instr->storage->kind] );

  // Emit modifiers, as ordered by the ptxModifier struct
  if( mod.APRX )       result.append( getAPRXAsString( mod.APRX) );
  if( mod.CC )         result.append( getCCAsString( mod.CC) );
  if( mod.SHAMT )      result.append( getSHAMTAsString( mod.SHAMT) );
  if( mod.SYNC )       result.append( getSYNCAsString( mod.SYNC) );
  if( mod.BAR )        result.append( getBARAsString( mod.BAR) );
  if( mod.TADDR )      result.append( getModifier_TADDR[mod.TADDR] );
  if( mod.NC )         result.append( getNCAsString( mod.NC) );
  if( mod.ROUND )      result.append( getROUNDAsString( mod.ROUND) );
  if( mod.TESTP )      result.append( getTESTPAsString( mod.TESTP) );
  if( mod.CACHEOP )    result.append( getCACHEOPAsString(parseState->parseData->deobfuscatedStringMapPtr, mod.CACHEOP) );
  if( mod.LEVEL )      result.append( getLEVELAsString( mod.LEVEL) );
  if( mod.FLOW )       result.append( getFLOWAsString( mod.FLOW) );
  if( mod.BRANCH )     result.append( getBRANCHAsString(parseState->parseData->deobfuscatedStringMapPtr, mod.BRANCH) );
  if( mod.TEXTURE )    result.append( getTEXTUREAsString( mod.TEXTURE) );
  if( mod.COMPONENT )  result.append( getCOMPONENTAsString( mod.COMPONENT) );
  if( mod.QUERY )      result.append( getQUERYAsString( mod.QUERY) );
  if( mod.VOTE )       result.append( getVOTEAsString( mod.VOTE) );
  if( mod.CLAMP )      result.append( getCLAMPAsString( mod.CLAMP) );
  if( mod.SHR )        result.append( getSHRAsString( mod.SHR) );
  if( mod.VMAD )       result.append( getVMADAsString( mod.VMAD) );
  if( mod.PRMT )       result.append( getPRMTAsString( mod.PRMT) );
  if( mod.SHFL )       result.append( getSHFLAsString( mod.SHFL) );
  if( mod.ENDIS )      result.append( getENDISAsString(parseState->parseData->deobfuscatedStringMapPtr, mod.ENDIS) );

  // Vector needs to be last
  if( mod.VECTOR )     result.append( getVECTORAsString( mod.VECTOR) );

  // Emit comparison operator and postoperator
  if( instr->cmp )     result.append( getCOMPAREAsString(instr->cmp) );
  if( instr->postop )  result.append( getOperator[instr->postop] );

  // FTZ must be after cmpop/postop
  if( mod.FTZ )        result.append( getFTZAsString(mod.FTZ) );
  if( mod.SAT )        result.append( getModifier_SAT[mod.SAT] ); // SAT must be after ftz
  // special case handling: atom.add.f16 and atom.add.f16x2 operations require the .noftz qualifier;
  // see Bug 3522212
  if( isAtomAddInstruction( result ) && (int)instr->tmplate->nrofInstrTypes == 1 &&
      ( instr->type[0]->kind == ptxTypeF16 || instr->type[0]->kind == ptxTypeF16x2 ) )
  {
      result.append( getNOFTZAsString( mod.NOFTZ ) );
  }

  // Emit types
  for( int i = 0; i < (int)instr->tmplate->nrofInstrTypes; i++ )
      printPtxType( result, instr->type[i], parseState );

    // clang-format on
}

// Builds a specifier vector for each video selector (e.g. a vector of ".b2356" for a v4 video SIMD instruction argument)
// fill with ".noSel" if there's no video selector at all
void printVideoSelectSpecifier( std::string& result, ptxInstruction instr )
{
    std::string temp_specifier;
    // c operands doesn't have a selector so it should always be ".noSel", and max number of arguments is 2 (a,b) + destination register + c
    for( unsigned int j = 0; j < instr->tmplate->nrofArguments; ++j )
    {
        if( instr->arguments[j]->kind != ptxVideoSelectExpression )
        {
            temp_specifier = ".noSel";
        }
        else
        {
            temp_specifier.clear();
            for( unsigned int i = 0; i < instr->arguments[j]->cases.VideoSelect->N; ++i )
            {
                if( i == 0 )
                {
                    temp_specifier.append( optix::getVideoSpecifier_SEL[instr->arguments[j]->cases.VideoSelect->selector[i]] );
                }
                else
                {
                    const char* tmp = optix::getVideoSpecifier_SEL[instr->arguments[j]->cases.VideoSelect->selector[i]];
                    if( tmp[0] != '\0' )
                        temp_specifier.append( tmp + 2 );
                }
            }
        }
        result.append( temp_specifier );
    }
}

}  // namespace optix
