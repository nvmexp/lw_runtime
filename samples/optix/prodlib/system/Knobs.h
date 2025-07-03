// Copyright LWPU Corporation 2008
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#pragma once

#include <optix_types.h>

#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iosfwd>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include <corelib/misc/String.h>
#include <prodlib/exceptions/Assert.h>

#define RT_PUBLIC_DSTRING( s ) s

#if defined( DEBUG ) || defined( DEVELOP )
#if !defined( OPTIX_ENABLE_KNOBS )
#define OPTIX_ENABLE_KNOBS
#endif
#endif

#if defined( OPTIX_ENABLE_KNOBS )
#define RT_DSTRING( s ) s
#define RT_PRIVATE_DSTRING( s ) s
#else
#define RT_DSTRING( s ) ""
#define RT_PRIVATE_DSTRING( s ) ""
#endif

// A knob is an Optix configuration value used for debugging and development. It is externally controlled through a
// config file, environment variables, or (in OptiX 7) API functions. Config files and environment variables are
// ignored in release builds.
//
// To introduce a new knob, simply define it as a static variable at file scope, for example:
//
//     static Knob<bool> savePTX("compile.savePTX",false,"Save PTX kernel to file");
//
// Then use the knob's value:
//
//     if( savePTX.get() ) { ... }
//
// It is important that the knob is initialized statically, so that by the time the knob registry is finalized, all
// knobs are registered. The exception to this are dynamic rtcore knobs, whose name/values are cached as string pairs
// and later concatenated and appended to the value of the rtcore.knobs" knob.
//
// Note that multiple knobs with the same name can be used as long as their default value and description match, so
// it's OK to put a knob definition in a header file, even if it's included in multiple compilation units.
//
// Also note that knobs with empty names and descriptions are OK. If the name is missing, a knob will simply always
// keep its default value. Therefore, it's OK to use e.g. the RT_DSTRING macro to hide the string name in release
// builds.
//
// Knobs do not have dependencies on any other part of Optix (such as the Logger or Exceptions etc), so it's OK to use
// them from pretty much anywhere. In particular, it's OK to read a knob's value *before* the knob registry has been
// finalized with KnobRegistry::finalizeKnobs().
//
// Note that only a subset of types for knobs are explicitly instantiated at the end of Knobs.cpp.
//
// Thread-safety: the various knob classes are NOT thread-safe. If this is really required, one could add a mutex that
// protects all setters and getters of m_value and m_updateCallback, but this might impact performance. However, it is
// believed that we do not need full thread-safety at the moment:
//
// - write/write collisions: All value changes are supposed to be done via the knob registry (which uses its own lock
//   to serialize all such requests). KnobBase::setUntyped() and Knob<T>::set() are not supposed to be called directly.
//   Thus, there should be no write/write collisions.
//
// - write/read collisions: All value changes (via the knob registry) need to happen before the first device context is
//   created. During that time there should not be any other threads that could read knob values. The only exception is
//   KnobRegistry::setKnobTyped(), which is used by ScopedKnobSetter in white-box unit tests. This class should only be
//   used in such way that during its construction and destruction no other threads are in OptiX code, potentially
//   reading knob values.
class KnobBase
{
  public:
    enum class Kind
    {
        DEVELOPER,     // Available only in debug or developer builds
        PUBLIC,        // Also available in release builds
        HIDDEN_PUBLIC  // Also available in release builds, but hidden from exported optix.props files
    };

    enum class Source
    {
        DEFAULT,              // knob definition in source file
        FILE,                 // optix.props
        ENVIRONMENT,          // environment variable (e.g. optix.compile.savePTX or optix_log_file)
        API,                  // From O7 knobs API
        FILE_OR_ELWIRONMENT,  // From O6 wrapper
        SCOPED_KNOB_SETTER,   // ScopedKnobSetter
        MIXED                 // In the case of rtcore knobs there can be more than one source
    };


    KnobBase( const std::string& name, const std::string& description, Kind kind )
        : m_name( name )
        , m_description( description )
        , m_kind( kind )
        , m_source( Source::DEFAULT )
    {
    }

    virtual ~KnobBase() {}

    // Returns the name of a knob.
    std::string getName() const { return m_name; }

    // Returns the kind of the knob.
    Kind getKind() const { return m_kind; }

    Source getSource() const { return m_source; }

    void setSource( Source source ) { m_source = source; }

    bool isSet() const { return m_source != Source::DEFAULT; }

    static std::string sourceToString( Source s );

    // Indicates whether the current value of the knob matches the default value set in its declaration.
    virtual bool isDefault() const = 0;

    // Prints a knob to specified stream. If verbose is true will print the default value and description.
    virtual void print( std::ostream& out, bool align = true, bool verbose = false, bool printSource = false ) const = 0;

    // Indicates whether the knob is equal to another knob (not just the value, but also name, kind, default, and
    // description).
    virtual bool isEqual( const KnobBase* other ) const = 0;

    // Sets the value of the knob.
    //
    // Not supposed to be called directly, only via the KnobRegistry or ScopedKnobSetter (see dislwssion about
    // thread-safety above).
    virtual OptixResult setUntyped( const std::string& val, Source source ) = 0;

  protected:
    // Alignment widths
    static const int SOURCE_WIDTH = 20;
    static const int NAME_WIDTH   = 40;
    static const int VALUE_WIDTH  = 20;

    const std::string m_name;
    const std::string m_description;
    const Kind        m_kind;
    Source            m_source;
};

// Regular (developer) knob with a value of any type T.
template <typename T>
class Knob : public KnobBase
{
  public:
    // non-copyable
    Knob( const Knob& ) = delete;
    Knob& operator=( const Knob& ) = delete;

    Knob( const char* name, const T& defaultval, const char* description, Kind kind = Kind::DEVELOPER );

    const T& get() const { return m_value; }

    bool isDefault() const override { return m_value == m_default; }

    void print( std::ostream& out, bool align, bool verbose, bool printSource ) const override;

    bool isEqual( const KnobBase* other ) const override;

    // Registers a callback which will be called when the value of the knob is set.
    void registerUpdateCB( std::function<void( T )> callback );

    // Unregisters the callback above.
    void unregisterUpdateCB();

    // Sets the value of the knob.
    //
    // Not supposed to be called directly, only via the KnobRegistry or ScopedKnobSetter (see dislwssion about
    // thread-safety above).
    void set( const T& val );

  protected:
    OptixResult setUntyped( const std::string& val, Source source ) override;

    // The default of the knob value.
    const T m_default;

    // The knob value.
    T m_value;

    // Callback to be ilwoked whenever the knob value is set.
    std::function<void( T )> m_updateCallback;
};

// A public knob keeps its name in release builds and is discoverable from the knobs dumped in the optix.props file.
template <typename T>
class PublicKnob : public Knob<T>
{
  public:
    // non-copyable
    PublicKnob( const PublicKnob& ) = delete;
    PublicKnob& operator=( const PublicKnob& ) = delete;

    PublicKnob( const char* name, const T& defaultVal, const char* description )
        : Knob<T>( name, defaultVal, description, KnobBase::Kind::PUBLIC )
    {
    }
};

// A hidden public knob keeps its name in release builds but is not dumped in the optix.props file.
template <typename T>
class HiddenPublicKnob : public Knob<T>
{
  public:
    // non-copyable
    HiddenPublicKnob( const HiddenPublicKnob& ) = delete;
    HiddenPublicKnob& operator=( const HiddenPublicKnob& ) = delete;

    HiddenPublicKnob( const char* name, const T& defaultVal, const char* description )
        : Knob<T>( name, defaultVal, description, KnobBase::Kind::HIDDEN_PUBLIC )
    {
    }
};


// Sets the value of a knob and restores it when destroyed. Intended to be used with test fixtures.
//
// This class is NOT thread-safe. It should only be used in such way that during its construction and destruction no
// other threads are in OptiX code, potentially reading knob values. See documentation about thread-safety for
// KnobBase.
class ScopedKnobSetter
{
  public:
    template <typename T>
    explicit ScopedKnobSetter( const std::string& name, const T& value );

  private:
    // Using this helper class and its derived template class allows to make ScopedKnobSetter a
    // non-template, which is more colwenient to use.
    class KnobSaver
    {
      public:
        virtual ~KnobSaver() {}
    };

    template <typename T>
    class KnobSaverT : public KnobSaver
    {
      public:
        KnobSaverT( const std::string& name, const T& value );
        ~KnobSaverT() override;

        std::string      m_name;
        T                m_oldValue;
        KnobBase::Source m_oldSource;
    };

    std::unique_ptr<KnobSaver> m_knobSaver;
};


// The knob registry holds a map of all registered knobs with methods to import/export knobs.
class KnobRegistry
{
  public:
    // Returns the location of the optix.props file.
    //
    // The directory is the value of OPTIX_PROPS_PATH if set, and the current working directory otherwise.
    // Appends a slash if the directory is non-empty, and "optix.props" as the filename.
    static std::string getOptixPropsLocation();


    // Registers a knob. Can only be used before finalizeKnobs(). Used by Knob constructor.
    void registerKnob( KnobBase* knob );


    // Initializes knob values from optix.props and environment variables.
    //
    // This method is supposed to be called very early, but after registration of knobs (and can be only be used before
    // finalizeKnobs()). Does nothing in release builds. Used by O7.
    OptixResult initializeKnobs();

    // Initializes knob values from a string.
    //
    // This method is supposed to be called very early, but after registration of knobs (and can be only be used before
    // finalizeKnobs()). Used by O6.
    //
    // The format of the string is the same as for the content of optix.props.
    OptixResult initializeKnobs( const std::string& nameValues );


    // Sets knob values from a file, e.g., optix.props.
    //
    // Can only be used before finalizeKnobs(). Used by O7 extension API for knobs.
    OptixResult setKnobsFromFile( const std::string& filename, bool IOFailureIsError );

    // Sets a knob value, which is given as string.
    //
    // Can only be used before finalizeKnobs(). Used by O7 extension API for knobs.
    OptixResult setKnob( const std::string& name, const std::string& value, KnobBase::Source source );

    // Sets a knob value, which is given as instance of the corresponding type.
    //
    // Can be used after finalizeKnobs(). Does not support the dynamic rtcore knobs (except for the static
    // "rtcore.knobs" knob itself). Used by ScopedKnobSetter.
    //
    // In an ideal world it would not be possible to change knob values after finalization. The fact that this method
    // can be used after finalizeKnobs() is a concession to white-box unit tests that use ScopedKnobSetter. In
    // particular, its destructor will run after finalization.
    template <typename T>
    OptixResult setKnobTyped( const std::string& name, const T& value, T& oldValue, KnobBase::Source source, KnobBase::Source& oldSource );


    // Performs a couple of tasks after knobs have been registered and their values set.
    //
    // This method is supposed to be called early during context creation. Afterwards, knobs can no longer be
    // registered or their values be changed (with the exception of using setKnobTyped()).
    //
    // The method transforms the dynamic rtcore knobs (see transformDynamicRtcoreKnobs()) and exports knobs to an
    // optix.props file if requested (see exportKnobsToFile()).
    //
    // The first call to finalizeKnobs returns the string of errors that have been aclwmulated so far. (After the first
    // call, no more errors can occur.)
    void finalizeKnobs( std::string& errorString );


    // Prints a list of all knobs, their current values, and descriptions.
    //
    // Knobs at their default value are commented out. Developer and public hidden knobs are skipped in release builds.
    void printKnobs( std::ostream& out ) const;

    // Prints a list of all non-default knobs and their current values.
    //
    // Includes a header line. Suitable for output during context creation.
    void printNonDefaultKnobs( std::ostream& out ) const;

    // Returns a list of all non-default knobs and their vurrent values.
    //
    // Similar to printNonDefaultKnobs(), but without a header-line and suitable for post-processing.
    std::vector<std::string> getNonDefaultKnobs() const;


  private:
    // Sets knob values from a file, e.g., optix.props.
    //
    // Used by O7 API and for optix.props file in debug/develop builds. Can only be used before finalizeKnobs().
    // Caller needs to lock m_mutex.
    OptixResult setKnobsFromFileLocked( const std::string& filename, bool IOFailureIsError );

    // Sets knob values from the environment.
    //
    // Used only in debug/develop builds. Can only be used before finalizeKnobs(). Caller needs to lock m_mutex.
    OptixResult setKnobsFromElwironmentLocked();

    // Low-level function for all setters except for setKnobTyped().
    //
    // Caller needs to lock m_mutex.
    OptixResult setKnobLocked( const std::string& name, const std::string& value, KnobBase::Source source );

    // Prints a list of all knobs, their current values, and descriptions.
    //
    // Knobs at their default value are commented out. Developer and public hidden knobs are skipped in release builds.
    // Caller needs to lock m_mutex.
    void printKnobsLocked( std::ostream& out ) const;

    // Indicates whether "name" starts with "rtcore." but is not equal to "rtcore.knobs".
    static bool isDynamicRtcoreKnob( const std::string& name );

    // Combines all name/values of knobs in m_rtcoreKnobs into a single string and appends it to the value of the
    // "rtcore.knobs" knob.
    void transformDynamicRtcoreKnobs();

    // Creates a new optix.props file if the size of the existing file is 0, or the environment variable OPTIX_PROPS is
    // set. Does nothing in release builds.
    void exportKnobsToFile() const;


    // Lock that protects all members.
    mutable std::mutex m_mutex;

    // Indicates whether finalizeKnobs() has been called.
    bool m_finalized = false;

    // Multimap of knobs by name. Multiple instances of a knob with the same name are equal as determined by
    // KnobBase::isEqual().
    std::multimap<std::string, KnobBase*> m_knobs;

    // Map of all dynamic rtcore knobs. Handled by transformRtcoreKnobs().
    std::map<std::string, std::pair<std::string, KnobBase::Source>> m_rtcoreKnobs;

    // String that aclwmulates all error messages since there might be no logger yet (and we do not want such a
    // dependency). Can be queried as part of finalizeKnobs(),
    std::string m_errorString;
};


//
// Instance access
//

KnobRegistry& knobRegistry();


//
// Knob implementation
//

template <typename T>
Knob<T>::Knob( const char* name, const T& defaultval, const char* description, Kind kind )
    : KnobBase( name, description, kind )
    , m_default( defaultval )
    , m_value( defaultval )
{
    knobRegistry().registerKnob( this );
}

template <typename T>
void Knob<T>::set( const T& val )
{
    m_value = val;
    if( m_updateCallback )
        m_updateCallback( m_value );
}

template <typename T>
OptixResult Knob<T>::setUntyped( const std::string& val, Source source )
{
    bool    ok;
    const T parsedVal = corelib::from_string<T>( val, &ok );
    if( !ok )
        return OPTIX_ERROR_ILWALID_VALUE;

    set( parsedVal );
    m_source = source;
    return OPTIX_SUCCESS;
}

template <typename T>
void Knob<T>::registerUpdateCB( std::function<void( T )> callback )
{
    m_updateCallback = callback;
}

template <typename T>
void Knob<T>::unregisterUpdateCB()
{
    m_updateCallback = std::function<void( T )>();
}


template <typename T>
void Knob<T>::print( std::ostream& out, bool align, bool verbose, bool printSource ) const
{
    if( printSource )
        out << std::left << std::setw( SOURCE_WIDTH ) << sourceToString( m_source );
    const int KW = align ? NAME_WIDTH : 0;
    const int VW = align ? VALUE_WIDTH : 0;
    out << std::left << std::setw( KW ) << m_name << " " << std::setw( VW ) << m_value;
    if( verbose )
        out << "  // [" << m_default << "] " << m_description;
}

template <typename T>
bool Knob<T>::isEqual( const KnobBase* other ) const
{
    const Knob<T>* concrete = dynamic_cast<const Knob<T>*>( other );

    return concrete && concrete->m_name == m_name && concrete->m_description == m_description
           && concrete->m_default == m_default && concrete->m_value == m_value && concrete->m_kind == m_kind;
}


//
// ScopedKnobSetter implementation
//

template <typename T>
ScopedKnobSetter::ScopedKnobSetter( const std::string& name, const T& value )
{
    m_knobSaver.reset( new KnobSaverT<T>( name, value ) );
}

template <typename T>
ScopedKnobSetter::KnobSaverT<T>::KnobSaverT( const std::string& name, const T& value )
    : m_name( name )
{
    OptixResult result = knobRegistry().setKnobTyped( m_name, value, m_oldValue, KnobBase::Source::SCOPED_KNOB_SETTER, m_oldSource );
    if( result != OPTIX_SUCCESS )
    {
        std::string message =
            std::string( "ScopedKnobSetter for knob \"" ) + name + "\" has invalid name or invalid value";
        RT_ASSERT_MSG( false, message.c_str() );
    }
}

template <typename T>
ScopedKnobSetter::KnobSaverT<T>::~KnobSaverT()
{
    T                lwrrentValue;
    KnobBase::Source source;
    knobRegistry().setKnobTyped( m_name, m_oldValue, lwrrentValue, m_oldSource, source );
}
