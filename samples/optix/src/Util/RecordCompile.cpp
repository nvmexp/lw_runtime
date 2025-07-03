// Copyright LWPU Corporation 2018
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#include <Util/RecordCompile.h>

#include <Context/RTCore.h>

#include <support/rapidjson/include/rapidjson/document.h>
#include <support/rapidjson/include/rapidjson/filereadstream.h>
#include <support/rapidjson/include/rapidjson/filewritestream.h>
#include <support/rapidjson/include/rapidjson/stringbuffer.h>
#include <support/rapidjson/include/rapidjson/writer.h>

#include <cstdio>
#include <fstream>
#include <iostream>

using namespace rapidjson;

void recordcompile::recordCompileCall( const std::string& moduleName, const RtcCompileOptions& compileOptions, const std::string& bitcode )
{
    Document d;
    d.SetObject();

    Value compileOptionsObject( kObjectType );

    Value abiVariant( compileOptions.abiVariant );
    compileOptionsObject.AddMember( "abiVariant", abiVariant, d.GetAllocator() );

    Value numPayloadRegisters( compileOptions.numPayloadRegisters );
    compileOptionsObject.AddMember( "numPayloadRegisters", numPayloadRegisters, d.GetAllocator() );

    Value numAttributeRegisters( compileOptions.numAttributeRegisters );
    compileOptionsObject.AddMember( "numAttributeRegisters", numAttributeRegisters, d.GetAllocator() );

    Value numCallableParamRegisters( compileOptions.numCallableParamRegisters );
    compileOptionsObject.AddMember( "numCallableParamRegisters", numCallableParamRegisters, d.GetAllocator() );

    Value numMemoryAttributeScalars( compileOptions.numMemoryAttributeScalars );
    compileOptionsObject.AddMember( "numMemoryAttributeScalars", numMemoryAttributeScalars, d.GetAllocator() );

    Value smVersion( compileOptions.smVersion );
    compileOptionsObject.AddMember( "smVersion", smVersion, d.GetAllocator() );

    Value maxRegisterCount( compileOptions.maxRegisterCount );
    compileOptionsObject.AddMember( "maxRegisterCount", maxRegisterCount, d.GetAllocator() );

    Value optLevel( compileOptions.optLevel );
    compileOptionsObject.AddMember( "optLevel", optLevel, d.GetAllocator() );

    Value debugLevel( compileOptions.debugLevel );
    compileOptionsObject.AddMember( "debugLevel", debugLevel, d.GetAllocator() );

    Value enabledTools( compileOptions.enabledTools );
    compileOptionsObject.AddMember( "enabledTools", enabledTools, d.GetAllocator() );

    Value exceptionFlags( compileOptions.exceptionFlags );
    compileOptionsObject.AddMember( "exceptionFlags", exceptionFlags, d.GetAllocator() );

    Value traversableGraphFlags( compileOptions.traversableGraphFlags );
    compileOptionsObject.AddMember( "traversableGraphFlags", traversableGraphFlags, d.GetAllocator() );

    Value targetSharedMemoryBytesPerSM( compileOptions.targetSharedMemoryBytesPerSM );
    compileOptionsObject.AddMember( "targetSharedMemoryBytesPerSM", targetSharedMemoryBytesPerSM, d.GetAllocator() );

    Value compileForLwda( compileOptions.compileForLwda );
    compileOptionsObject.AddMember( "compileForLwda", compileForLwda, d.GetAllocator() );

    d.AddMember( "compileOptions", compileOptionsObject, d.GetAllocator() );

    Value name;
    name.SetString( moduleName, d.GetAllocator() );
    d.AddMember( "moduleName", name, d.GetAllocator() );

    // Write file out.
    std::string filename = moduleName + ".json";
    FILE*       fp       = fopen( filename.c_str(), "wb" );
    if( fp )
    {
        char            writeBuffer[65536];
        FileWriteStream os( fp, writeBuffer, sizeof( writeBuffer ) );

        Writer<FileWriteStream> writer( os );
        d.Accept( writer );

        fclose( fp );
    }
    else
    {
        std::cerr << "Unable to open file " << filename << " for writing.\n";
    }

    // Write bytes to separate file.
    std::string   binPath = moduleName + ".bc";
    std::ofstream fout;
    fout.open( binPath, std::ios_base::binary | std::ios_base::out );
    fout.write( (char*)bitcode.c_str(), bitcode.size() );
    fout.close();
}

void recordcompile::loadCompileCall( const std::string& jsonPath,
                                     const std::string& binPath,
                                     std::string&       moduleName,
                                     RtcCompileOptions& compileOptions,
                                     std::string&       bitcode )
{
    FILE* fp = fopen( jsonPath.c_str(), "rb" );
    if( !fp )
    {
        std::cerr << "Unable to open file " << jsonPath << " for reading.\n";
        return;
    }

    char           readBuffer[65536];
    FileReadStream is( fp, readBuffer, sizeof( readBuffer ) );

    Document d;
    d.ParseStream( is );

    fclose( fp );

    std::string name = d["moduleName"].GetString();
    moduleName.assign( name );

    // Load bitcode
    std::ifstream fin( binPath );

    fin.seekg( 0, fin.end );
    size_t length = fin.tellg();
    fin.seekg( 0, fin.beg );

    std::string code( length, '\0' );
    fin.read( &code[0], length );

    fin.close();

    bitcode.assign( code );

    // Load compile options.
    Value& compileObj                        = d["compileOptions"];
    compileOptions.abiVariant                = (RtcAbiVariant)compileObj["abiVariant"].GetInt();
    compileOptions.numPayloadRegisters       = compileObj["numPayloadRegisters"].GetInt();
    compileOptions.numAttributeRegisters     = compileObj["numAttributeRegisters"].GetInt();
    compileOptions.numCallableParamRegisters = compileObj["numCallableParamRegisters"].GetInt();
    compileOptions.numMemoryAttributeScalars = compileObj["numMemoryAttributeScalars"].GetInt();
    compileOptions.traversableGraphFlags     = compileObj["traversableGraphFlags"].GetBool();
    compileOptions.smVersion                 = compileObj["smVersion"].GetInt();
    compileOptions.maxRegisterCount          = compileObj["maxRegisterCount"].GetInt();
    compileOptions.optLevel                  = compileObj["optLevel"].GetInt();
    compileOptions.debugLevel                = compileObj["debugLevel"].GetInt();
    compileOptions.enabledTools              = compileObj["enabledTools"].GetUint();
    compileOptions.exceptionFlags            = compileObj["exceptionFlags"].GetUint();
    compileOptions.compileForLwda            = compileObj["compileForLwda"].GetUint();
}
