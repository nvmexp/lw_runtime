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

#include <Objects/Transform.h>

#include <Context/BindingManager.h>
#include <Context/Context.h>
#include <Context/ProgramManager.h>
#include <Context/RTCore.h>
#include <Context/SharedProgramManager.h>
#include <Context/TableManager.h>
#include <Context/UpdateManager.h>
#include <Device/LWDADevice.h>
#include <Device/DeviceManager.h>
#include <Exceptions/VariableNotFound.h>
#include <ExelwtionStrategy/CORTTypes.h>
#include <FrontEnd/Canonical/CanonicalProgram.h>
#include <FrontEnd/Canonical/VariableReference.h>
#include <KernelUtils/Transform_ptx_bin.h>
#include <Memory/MemoryManager.h>
#include <Objects/Group.h>
#include <Objects/Selector.h>
#include <Util/BufferUtil.h>
#include <Util/ContainerAlgorithm.h>
#include <Util/LinkedPtrHelpers.h>
#include <Util/MotionAabb.h>
#include <Util/ResampleMotion.h>
#include <Util/TransformHelper.h>

#include <prodlib/exceptions/IlwalidValue.h>
#include <prodlib/exceptions/ValidationError.h>

#include <cmath>

using namespace optix;
using namespace prodlib;

static inline float3 transform( const float3& v, const float m[16] );
static inline Aabb transform( const Aabb& aabb, const float m[16] );
static void transform( const Aabb& inAabb0, const Aabb& inAabb1, const float* key0, const float* key1, Aabb& outaabb0, Aabb& outaabb1 );
static inline Matrix4x4 getScaleMatrixFromSrtKey( const float key[] );

static inline Aabb transformSRT( const Aabb& aabb, const float* key );
static inline void transformSRT( const Aabb& aabb, const float* key0, const float* key1, Aabb& outAabb0, Aabb& outAabb1 );
static inline void transformSRT( const Aabb& aabb, const float* key0, const float* key1, float t0, float t1, Aabb& outAabb0, Aabb& outAabb1 );
static inline void transformSRT( const Aabb& aabb0, const Aabb& aabb1, const float* key0, const float* key1, float t0, float t1, Aabb& outAabb0, Aabb& outAabb1 );
static Aabb transformSR( const Aabb& aabb, const float* key0, const float* key1, float t0, float t1 );
static Aabb transformSR( const Aabb& aabb0, const Aabb& aabb1, const float* key0, const float* key1, float t0, float t1 );

//------------------------------------------------------------------------
// CTOR/DTOR
//------------------------------------------------------------------------

Transform::Transform( Context* context )
    : GraphNode( context, RT_OBJECT_TRANSFORM )
{
    m_transform         = Matrix<4, 4>::identity();
    m_ilwerse_transform = Matrix<4, 4>::identity();
    if( !m_context->useRtxDataModel() )
        LexicalScope::reallocateRecord();

    // Transforms always have a height of 1
    m_context->getBindingManager()->receivePropertyDidChange_TransformHeight( 1, true );

    // Set visit program after record is allocated
    setVisitProgram( context->getSharedProgramManager()->getProgram( data::getTransformSources(), "transform" ) );

    // Set bounding box program
    std::string bbname = context->useRtxDataModel() ? "bounds_rtctransform" : "bounds_transform";
    Program*    bounds = m_context->getSharedProgramManager()->getBoundsRuntimeProgram( bbname, false, false );
    setBoundingBoxProgram( bounds );
}

Transform::~Transform()
{
    setChild( nullptr );
    setVisitProgram( nullptr );
    setBoundingBoxProgram( nullptr );
    m_context->getBindingManager()->receivePropertyDidChange_TransformHeight( 1, false );
    deleteVariables();
    for( auto& traversable : m_traversables )
        if( traversable )
            traversable->removeListener( this );
}


//------------------------------------------------------------------------
// Public API
//------------------------------------------------------------------------

void Transform::setChild( GraphNode* child )
{
    checkForRelwrsiveGraph( child );

    // Transform child properties:
    //
    // Direct caller:                   originates from visit program to child
    // Trace Caller:                    propagates from parent (must be attached before / removed after attachment)
    // Attachment:                      propagates from parent
    // Transform height:                propagates from child
    // Acceleration height:             propagates from child
    // Has Motion Aabbs:                propagates from child
    // Instance Transform:              propagates from child
    // SBT Index:                       propagates from child
    // Requires traversable:            originates from parent
    // Direct traversable:              propagates from child
    // Transform requires traversable:   originates from parent

    GraphNode* oldChild = m_child.get();
    if( oldChild )
    {
        // Remove old properties
        this->attachOrDetachProperty_RequiresTraversable( oldChild, false );
        this->attachOrDetachProperty_TransformRequiresTraversable( oldChild, false );
        oldChild->attachOrDetachProperty_HasMotionAabbs( this, false );
        oldChild->attachOrDetachProperty_AccelerationHeight( this, false );
        oldChild->attachOrDetachProperty_TransformHeight( this, false );

        // Avoid cycles while propagating attachment (see lwbugswb #2422313)
        m_child.set( this, nullptr );
        this->attachOrDetachProperty_Attachment( oldChild, false );
        m_child.set( this, oldChild );  // probably unnecessary

        this->attachOrDetachProperty_RtxUniversalTraversal( oldChild, false );
        this->attachOrDetachProperty_TraceCaller( oldChild, false );
        if( getVisitProgram() )
        {
            getVisitProgram()->attachOrDetachProperty_DirectCaller( oldChild, false );
        }
    }

    // Update the child link
    m_child.set( this, child );

    GraphNode* newChild = m_child.get();
    if( newChild )
    {
        // Add new properties
        if( getVisitProgram() )
        {
            getVisitProgram()->attachOrDetachProperty_DirectCaller( newChild, true );
        }
        this->attachOrDetachProperty_TraceCaller( newChild, true );
        this->attachOrDetachProperty_RtxUniversalTraversal( newChild, true );
        this->attachOrDetachProperty_Attachment( newChild, true );
        newChild->attachOrDetachProperty_TransformHeight( this, true );
        newChild->attachOrDetachProperty_AccelerationHeight( this, true );
        newChild->attachOrDetachProperty_HasMotionAabbs( this, true );
        newChild->attachOrDetachProperty_InstanceTransform( this, &m_child );
        newChild->attachOrDetachProperty_SBTIndex( this, &m_child );
        this->attachOrDetachProperty_RequiresTraversable( newChild, true );
        this->attachOrDetachProperty_TransformRequiresTraversable( newChild, true );
        newChild->attachTraversableHandle( this, &m_child );
        newChild->attachOrDetachProperty_VisibilityMaskInstanceFlags( this, &m_child );
    }

    // Needed because transforms only have a single child and we store
    // it directly in the object record (as opposed to e.g. group children
    // which are stored in a buffer).
    subscribeForValidation();
    writeRecord();
}

GraphNode* Transform::getChild() const
{
    return m_child.get();
}

void Transform::setMatrix( const float* matrix, bool transpose )
{
    const bool oldHasMotionAabbs = hasMotionAabbs();
    const bool oldHasMotionKeys  = hasMotionKeys();

    // Clear existing motion keys
    m_keyType = RT_MOTIONKEYTYPE_NONE;
    m_keys.clear();

    const Matrix<4, 4>* source = reinterpret_cast<const Matrix<4, 4>*>( matrix );
    const Matrix4x4 oldTransform = m_transform;
    m_transform                  = transpose ? source->transpose() : *source;
    writeRecord();

    if( m_context->useRtxDataModel() && oldTransform != m_transform )
    {
        sendPropertyDidChange_InstanceTransform( getInstanceTransform() );
        updateTraversable();
    }

    // Push HasMotionAabbs property up the graph.  This is similar to receivePropertyDidChange_HasMotionAabbs
    // except that we are updating our self motion, not receiving it from a child subgraph.
    const bool newHasMotionAabbs = hasMotionAabbs();
    if( oldHasMotionAabbs != newHasMotionAabbs )
    {
        // Add new value or remove old one.  Since this is a boolean, we only have to do one of these.
        sendPropertyDidChange_HasMotionAabbs( newHasMotionAabbs );
    }

    // Update global property: does the graph have motion transforms
    if( isAttached() )
    {
        const bool newHasMotionKeys = hasMotionKeys();
        if( oldHasMotionKeys != newHasMotionKeys )
            m_context->getBindingManager()->receivePropertyDidChange_HasMotionTransforms( newHasMotionKeys );
    }
}

void Transform::setIlwerseMatrix( const float* matrix, bool transpose )
{
    // Assumption: if setIlwerseMatrix is called, setMatrix was called before!
    // I.e., the transform is already 'static', all motion blur related properties are cleared
    RT_ASSERT( m_keys.empty() );
    const Matrix<4, 4>* source = reinterpret_cast<const Matrix<4, 4>*>( matrix );
    m_ilwerse_transform = transpose ? source->transpose() : *source;
    writeRecord();
}

void Transform::getMatrix( float* matrix, bool transpose ) const
{
    Matrix<4, 4>* dest = reinterpret_cast<Matrix<4, 4>*>( matrix );
    *dest = transpose ? m_transform.transpose() : m_transform;
}

void Transform::getIlwerseMatrix( float* matrix, bool transpose ) const
{
    Matrix<4, 4>* dest = reinterpret_cast<Matrix<4, 4>*>( matrix );
    *dest = transpose ? m_ilwerse_transform.transpose() : m_ilwerse_transform;
}

void Transform::setMotionRange( float timeBegin, float timeEnd )
{
    m_timeBegin = timeBegin;
    m_timeEnd   = timeEnd;
}

void Transform::getMotionRange( float& timeBegin, float& timeEnd ) const
{
    timeBegin = m_timeBegin;
    timeEnd   = m_timeEnd;
}

void Transform::setMotionBorderMode( RTmotionbordermode borderBegin, RTmotionbordermode borderEnd )
{
    m_beginBorderMode = borderBegin;
    m_endBorderMode   = borderEnd;
}

void Transform::getMotionBorderMode( RTmotionbordermode& borderBegin, RTmotionbordermode& borderEnd )
{
    borderBegin = m_beginBorderMode;
    borderEnd   = m_endBorderMode;
}

void Transform::setKeys( int keyCount, RTmotionkeytype keyType, const float* keys )
{
    RT_ASSERT( keyCount >= 0 );
    RT_ASSERT( keys != nullptr || keyCount == 0 );
    RT_ASSERT( keyType != RT_MOTIONKEYTYPE_NONE || keyCount == 0 );

    const bool wasMotionTransform = m_keys.size() > 1U;

    // Clear existing static transform
    m_transform = Matrix<4, 4>::identity();

    // NOTE: Don't insert any early outs between old values and their use below.
    const bool            oldHasMotionAabbs = hasMotionAabbs();
    const bool            oldHasMotionKeys  = hasMotionKeys();
    const RTmotionkeytype oldKeyType        = getKeyType();
    const unsigned int    oldKeyCount       = getKeyCount();

    m_keyType = keyType;
    m_keys.resize( keyCount * getKeySize( m_keyType ) );

    if( keyCount == 0 )
    {
        m_keyType = RT_MOTIONKEYTYPE_NONE;
        if( m_motionData )
        {
            m_motionData->detachFromParents();
            m_motionData.reset();
        }

        if( m_context->useRtxDataModel() && wasMotionTransform )
        {
            sendPropertyDidChange_InstanceTransform( getInstanceTransform() );
        }
    }
    else if( keyCount == 1 )
    {
        // Colwert the single key into a static transform
        m_keyType = RT_MOTIONKEYTYPE_NONE;
        m_keys.clear();
        if( keyType == RT_MOTIONKEYTYPE_MATRIX_FLOAT12 )
        {
            std::copy( keys, keys + 12, &m_transform[0] );
            m_transform[12] = 0.0f;
            m_transform[13] = 0.0f;
            m_transform[14] = 0.0f;
            m_transform[15] = 1.0f;
        }
        else  // RT_MOTIONKEYTYPE_SRT_FLOAT16
        {
            // nullify the identity
            m_transform[0]  = 0.0f;
            m_transform[5]  = 0.0f;
            m_transform[10] = 0.0f;

            // scale
            Matrix4x4 scale = getScaleMatrixFromSrtKey( keys );

            // rotation
            // normalize user input
            Quaternion  quat( keys[9], keys[10], keys[11], keys[12] );
            const float ilw_length = 1.f / sqrt( quat.m_q.x * quat.m_q.x + quat.m_q.y * quat.m_q.y
                                                 + quat.m_q.z * quat.m_q.z + quat.m_q.w * quat.m_q.w );
            quat.m_q.x *= ilw_length;
            quat.m_q.y *= ilw_length;
            quat.m_q.z *= ilw_length;
            quat.m_q.w *= ilw_length;
            Matrix4x4 rotate;
            quat.toMatrix( rotate.getData() );

            // compute R*S
            for( int i = 0; i < 3; ++i )
                for( int j = 0; j < 4; ++j )
                    for( int k = 0; k < 3; ++k )
                        m_transform[i * 4 + j] += rotate[i * 4 + k] * scale[k * 4 + j];

            // translation
            m_transform[0 + 3] += keys[13] + keys[3];
            m_transform[4 + 3] += keys[14] + keys[6];
            m_transform[8 + 3] += keys[15] + keys[8];
        }
        m_ilwerse_transform = m_transform.ilwerse();

        if( m_context->useRtxDataModel() && wasMotionTransform )
        {
            sendPropertyDidChange_InstanceTransform( getInstanceTransform() );
        }
    }
    else
    {
        std::copy( keys, keys + keyCount * getKeySize( m_keyType ), m_keys.begin() );
        if( m_keyType == RT_MOTIONKEYTYPE_SRT_FLOAT16 )
        {
            // normalize user input now so we do not need to do it during traversal all the time
            for( int keyI = 0; keyI < keyCount; ++keyI )
            {
                float*      key = &m_keys[keyI * 16];
                const float ilw_length =
                    1.f / sqrt( key[9] * key[9] + key[10] * key[10] + key[11] * key[11] + key[12] * key[12] );
                key[9] *= ilw_length;
                key[10] *= ilw_length;
                key[11] *= ilw_length;
                key[12] *= ilw_length;
            }
        }

        if( m_context->useRtxDataModel() && !wasMotionTransform )
        {
            sendPropertyDidChange_InstanceTransform( m_transform );
        }
    }

    writeRecord();

    // Push HasMotionAabbs property up the graph.  This is similar to receivePropertyDidChange_HasMotionAabbs
    // except that we are updating our self motion, not receiving it from a child subgraph.
    const bool newHasMotionAabbs = hasMotionAabbs();
    if( oldHasMotionAabbs != newHasMotionAabbs )
    {
        // Add new value or remove old one.  Since this is a boolean, we only have to do one of these.
        sendPropertyDidChange_HasMotionAabbs( newHasMotionAabbs );
    }

    const bool newHasMotionKeys = hasMotionKeys();
    if( oldHasMotionKeys != newHasMotionKeys )
    {
        // Motion keys induce requiresTraversable on children
        if( m_child )
            m_child->receivePropertyDidChange_RequiresTraversable( nullptr, newHasMotionKeys );
        this->receivePropertyDidChange_RequiresTraversable( nullptr, newHasMotionKeys );
    }
    updateTraversable();

    // Update global property: does the graph have motion transforms
    if( isAttached() )
    {
        if( oldHasMotionKeys != newHasMotionKeys )
            m_context->getBindingManager()->receivePropertyDidChange_HasMotionTransforms( newHasMotionKeys );
    }
}

unsigned int Transform::getKeyCount() const
{
    if( m_keyType == RT_MOTIONKEYTYPE_NONE )
        return 0u;
    return static_cast<unsigned>( m_keys.size() / getKeySize( m_keyType ) );
}

RTmotionkeytype Transform::getKeyType() const
{
    return m_keyType;
}

void Transform::getKeys( float* keys ) const
{
    RT_ASSERT( keys );
    algorithm::copy( m_keys, keys );
}

// clang-format off
MotionAabb Transform::computeMotionAabb( const MotionAabb& child_maabb ) const
{
    if( !child_maabb.isValid() )
        return child_maabb;

    MotionAabb maabb( child_maabb );
    if( m_keys.empty() )
    {
        // static transform
        for( size_t i = 0; i < maabb.keyCount(); ++i )
            maabb.aabbs()[i] = transform( maabb.aabbs()[i], m_transform.getData() );
    }
    else
    {
        // Motion case
        const unsigned int keycount = getKeyCount();
        RT_ASSERT( keycount >= 2 );
        RT_ASSERT( m_keyType != RT_MOTIONKEYTYPE_NONE );

        const bool keys_align_perfectly = maabb.isStatic() || maabb.keysAlign( m_timeBegin, m_timeEnd, keycount );
        if( !keys_align_perfectly )
        {
            // while the keys may not perfectly align, they may still create a regular distribution in the end
            // do a simple / partial test for this case here
            const bool output_keys_regularly_distributed = maabb.keysAreRegularlyDistributed() && m_timeBegin == maabb.timeFirstKey() && m_timeEnd == maabb.timeLastKey() &&
                ((std::max( keycount, maabb.keyCount())-1 ) % (std::min( keycount, maabb.keyCount())-1 ) == 0);
            MotionAabb outMaabb = output_keys_regularly_distributed ? MotionAabb( m_timeBegin, m_timeEnd ) : MotionAabb();
            // if either the transform or the child has the vanish-border-mode set, nothing is visible in the out-of-borders case
            // hence, we pass this information along
            if( maabb.borderModeBegin() == RT_MOTIONBORDERMODE_VANISH || m_beginBorderMode == RT_MOTIONBORDERMODE_VANISH ) outMaabb.setBorderModeBegin( RT_MOTIONBORDERMODE_VANISH );
            if( maabb.borderModeEnd() == RT_MOTIONBORDERMODE_VANISH || m_endBorderMode == RT_MOTIONBORDERMODE_VANISH ) outMaabb.setBorderModeEnd( RT_MOTIONBORDERMODE_VANISH );
            // if(!output_keys_regularly_distributed) we will do epsilon comparisons with the output times of the keys to check for regular distribution
            handleIrregularKeys( maabb, keycount, !output_keys_regularly_distributed, outMaabb );
            return outMaabb;
        }
        else
        {
            const bool child_is_static = maabb.isStatic();
            if( child_is_static )
                maabb.resizeWithRegularDistribution( m_timeBegin, m_timeEnd, keycount, maabb.aabbs()[0] );

            // overwrite in case we have vanish set here
            // if vanish was already set, this will have no effect
            if( m_beginBorderMode == RT_MOTIONBORDERMODE_VANISH ) maabb.setBorderModeBegin( RT_MOTIONBORDERMODE_VANISH );
            if( m_endBorderMode == RT_MOTIONBORDERMODE_VANISH ) maabb.setBorderModeEnd( RT_MOTIONBORDERMODE_VANISH );

            if( m_keyType == RT_MOTIONKEYTYPE_MATRIX_FLOAT12 )
            {
                if( child_is_static )
                {
                    for( unsigned int i = 0; i < maabb.keyCount(); ++i )
                        maabb.aabb( i ) = transform( maabb.aabb( i ), &m_keys[ i * 12 ] );
                }
                else // input aabbs not static && key_count > 1
                {
                    RT_ASSERT( maabb.keyCount() > 1 );
                    // we cannot do this in place, need to copy input data
                    MotionAabb inputAabb = maabb;
                    // all but aabb 0 will be overwritten below, so ilwalidate aabb 0 here
                    maabb.aabb( 0 ).ilwalidate();
                    for( unsigned int i = 1; i < maabb.keyCount(); ++i )
                    {
                        // interval [i-1,i] with motion input (2 linearly interpolated aabbs), 2 transform keys (linearly interpolated)
                        // this operation is nonlinear, i.e., we cannot just apply the first transform on the first aabb
                        // and the second transform on the second aabb to get an interpolatable result that bounds M(t)*aabb(t)
                        // this function returns valid bounds at i-1, i for the interval
                        // these bounds may be larger than previously computed bounds at i-1, hence, we need to update them!
                        Aabb maabb0;
                        transform( inputAabb.aabb( i - 1 ), inputAabb.aabb( i ), &m_keys[(i - 1) * 12], &m_keys[i * 12], maabb0, maabb.aabb( i ) );
                        // update previous output key to include the required outAabb0 for this interval
                        maabb.aabb( i - 1 ).include( maabb0 );
                    }
                }
            }
            else
            {
                // aabbs[i] will contain the SRT aabb which is constant over interval from time segment (i-1, i).

                Aabb outAabb0;
                if( child_is_static )
                {
                    // we compute the new aabbs in place, replacing the previous aabbs.
                    // except for aabb at index 0, i.e., we need to ilwalidate the first aabb
                    maabb.aabb( 0 ).ilwalidate();
                    for( unsigned int i = 1; i < keycount; ++i )
                    {
                        transformSRT( maabb.aabb( i ), &m_keys[(i - 1) * 16], &m_keys[i * 16], outAabb0, maabb.aabb( i ) );
                        // update previous output key to include the required outAabb0 for this interval
                        maabb.aabb( i - 1 ).include( outAabb0 );
                    }
                }
                else
                {
                    // we cannot do this in place, need to copy input data
                    MotionAabb inputAabb = maabb;
                    // aabb 0 will be modified below and needs to be ilwalidated here
                    maabb.aabb( 0 ).ilwalidate();
                    for( unsigned int i = 1; i < keycount; ++i )
                    {
                        transformSRT( inputAabb.aabb( i - 1 ), inputAabb.aabb( i ), &m_keys[(i - 1) * 16], &m_keys[i * 16], 0.0f, 1.0f, outAabb0, maabb.aabb( i ) );
                        // update previous output key to include the required outAabb0 for this interval
                        maabb.aabb( i - 1 ).include( outAabb0 );
                    }
                }
            }
        }
    }

    return maabb;
}
// clang-format on

void Transform::validate() const
{
    // Attachment isn't always required to perform validation.  rtBufferValidate doesn't require
    // attachment, but ValidationManager does.  Do the attachment check in ValidationManager.

    // parent class
    GraphNode::validate();

    if( !hasChild() )
        throw ValidationError( RT_EXCEPTION_INFO, "No child set for transform node" );

    if( !m_unresolvedRemaining.empty() )
    {
        const VariableReference* varRef =
            m_context->getProgramManager()->getVariableReferenceById( m_unresolvedRemaining.front() );
        const std::string& varname = varRef->getInputName();
        throw VariableNotFound( RT_EXCEPTION_INFO, this, "Unresolved reference to variable " + varname + " from "
                                                             + varRef->getParent()->getInputFunctionName() );
    }
}


//------------------------------------------------------------------------
// Internal API
//------------------------------------------------------------------------


ObjectClass Transform::getChildType() const
{
    return getChild()->getClass();
}

bool Transform::hasChild() const
{
    return m_child;
}

unsigned int Transform::getKeySize( RTmotionkeytype keyType )
{
    switch( keyType )
    {
        case RT_MOTIONKEYTYPE_NONE:
            return 0;
        case RT_MOTIONKEYTYPE_MATRIX_FLOAT12:
            return 12;
        case RT_MOTIONKEYTYPE_SRT_FLOAT16:
            return 16;
        default:
            throw prodlib::IlwalidValue( RT_EXCEPTION_INFO, "Unsupported RTmotionkeytype found on Transform" );
    }
}

//------------------------------------------------------------------------
// LinkedPtr relationship management
//------------------------------------------------------------------------

void Transform::detachLinkedChild( const LinkedPtr_Link* link )
{
    if( link == &m_child )
        setChild( nullptr );
    else
        detachLinkedProgram( link );
}


//------------------------------------------------------------------------
// Object record management
//------------------------------------------------------------------------

size_t Transform::getRecordBaseSize() const
{
    if( !m_context->useRtxDataModel() )
        return sizeof( cort::TransformRecord );
    else if( requiresDirectTraversable() )
        return sizeof( cort::GraphNodeRecord );

    return 0;
}

void Transform::updateMotionData() const
{
    if( m_keys.empty() )
    {
        // we may have had a motion data buffer, remove it now as it is not needed anymore (and must not be used!)
        m_motionData.reset();
        return;
    }

    if( !m_motionData )
    {
        m_motionData.reset( createBuffer<float>( m_context, 1, RT_BUFFER_INPUT ) );
        m_motionData->markAsBindlessForInternalUse();
    }

    MappedBuffer<float> mdbuf( m_motionData.get(), MAP_WRITE_DISCARD,
                               static_cast<int>( MotionDataOffset::KEYS ) + m_keys.size() );
    float* mdptr                                                   = mdbuf.ptr();
    mdptr[static_cast<int>( MotionDataOffset::KEY_TYPE )]          = int_as_float( m_keyType );
    mdptr[static_cast<int>( MotionDataOffset::BEGIN_BORDER_MODE )] = int_as_float( m_beginBorderMode );
    mdptr[static_cast<int>( MotionDataOffset::END_BORDER_MODE )]   = int_as_float( m_endBorderMode );
    mdptr[static_cast<int>( MotionDataOffset::TIME_BEGIN )]        = m_timeBegin;
    mdptr[static_cast<int>( MotionDataOffset::TIME_END )]          = m_timeEnd;
    mdptr[static_cast<int>( MotionDataOffset::NUM_KEYS )] = int_as_float( m_keys.size() / getKeySize( m_keyType ) );

    algorithm::copy( m_keys, mdptr + static_cast<int>( MotionDataOffset::KEYS ) );
}

void Transform::writeRecord() const
{
    if( !recordIsAllocated() )
        return;

    if( !m_context->useRtxDataModel() )
    {
        updateMotionData();

        cort::TransformRecord* t = getObjectRecord<cort::TransformRecord>();
        RT_ASSERT( t != nullptr );
        getMatrix( (float*)t->matrix.matrix, false );
        getIlwerseMatrix( (float*)t->ilwerse_matrix.matrix, false );
        t->child      = getSafeOffset( m_child.get() );
        t->motionData = m_motionData ? m_motionData->getId() : RT_BUFFER_ID_NULL;
    }
    GraphNode::writeRecord();
}


//------------------------------------------------------------------------
// Unresolved reference property
//------------------------------------------------------------------------

void Transform::sendPropertyDidChange_UnresolvedReference( VariableReferenceID refid, bool added ) const
{
    // Transform has no parent, but we need to keep track of the unresolved references
    // for validation.  Note, unlike GlobalScope which has to take into consideration
    // default values, we should not have any default values.
    bool changed = m_unresolvedRemaining.addOrRemoveProperty( refid, added );
    if( changed && added )
    {
        // const cast required because subscribe is non-const
        Transform* nonconst_this = const_cast<Transform*>( this );
        nonconst_this->subscribeForValidation();
    }
}


//------------------------------------------------------------------------
// Attachment
//------------------------------------------------------------------------

void Transform::sendPropertyDidChange_Attachment( bool added ) const
{
    GraphNode::sendPropertyDidChange_Attachment( added );

    if( m_child )
        m_child->receivePropertyDidChange_Attachment( added );
}

void Transform::attachmentDidChange( bool new_A )
{
    if( requiresDirectTraversable() )
        reallocateRecord();

    if( hasMotionKeys() )
        m_context->getBindingManager()->receivePropertyDidChange_HasMotionTransforms( new_A );
}


//------------------------------------------------------------------------
// RtxUniversalTraversal
//------------------------------------------------------------------------

void Transform::sendPropertyDidChange_RtxUniversalTraversal() const
{
    if( m_child )
        m_child->receivePropertyDidChange_RtxUniversalTraversal();
}

bool Transform::rtxTraversableNeedUniversalTraversal() const
{
    // rtx requires universal traversal when traversing a graph starting at a transform.
    return true;
}

//------------------------------------------------------------------------
// Geometry flags and visibility mask properties
//------------------------------------------------------------------------

bool Transform::getVisibilityMaskInstanceFlags( RTvisibilitymask& mask, RTinstanceflags& flags ) const
{
    if( !m_child )
        return false;
    return m_child->getVisibilityMaskInstanceFlags( mask, flags );
}

void Transform::attachOrDetachProperty_VisibilityMaskInstanceFlags( GraphNode* parent, LinkedPtr_Link* child )
{
    RTvisibilitymask mask;
    RTinstanceflags  flags;
    if( m_child && m_child->getVisibilityMaskInstanceFlags( mask, flags ) )
    {
        // NOTE: child, when properly downcast, is the same as this.
        parent->receivePropertyDidChange_VisibilityMaskInstanceFlags( child, mask, flags );
    }
}

//------------------------------------------------------------------------
// Direct Caller
//------------------------------------------------------------------------
void Transform::attachOrDetachProperty_DirectCaller_toChildren( Program* program, bool added ) const
{
    if( m_child )
        program->attachOrDetachProperty_DirectCaller( m_child.get(), added );
}


//------------------------------------------------------------------------
// Instance Transform property
//------------------------------------------------------------------------
void Transform::receiveProperty_InstanceTransform( LinkedPtr_Link* /*child*/, Matrix4x4 childTransform )
{
    RT_ASSERT( m_context->useRtxDataModel() );
    sendPropertyDidChange_InstanceTransform( m_transform * childTransform );
}

void Transform::instanceTransformDidChange( LinkedPtr_Link* parentLink, Matrix4x4 childTransform )
{
    // If we are a motion keyed transform and our child changed it's instance transform, it
    // doesn't propagate up through us because from our parent's perspective, our instance
    // transform is always identity.
    if( !hasMotionKeys() )
    {
        sendPropertyDidChange_InstanceTransform( m_transform * childTransform );
    }
}

Matrix4x4 Transform::getInstanceTransform() const
{
    RT_ASSERT_MSG( m_context->useRtxDataModel(), "Instance Transform only makes sense in RTX exelwtion model" );
    if( hasMotionKeys() )
    {
        return Matrix4x4::identity();
    }

    Matrix4x4 transform = m_transform;
    if( m_child )
    {
        GraphNode* child = m_child.get();
        transform *= child->getInstanceTransform();
    }
    return transform;
}

unsigned int Transform::getSBTIndex() const
{
    return m_child ? m_child->getSBTIndex() : GraphNode::getSBTIndex();
}


//------------------------------------------------------------------------
// SBTIndex
//------------------------------------------------------------------------

void Transform::childSBTIndexDidChange( LinkedPtr_Link* parentLink, unsigned int oldSBTIndex, unsigned int newSBTIndex )
{
    if( !m_context->useRtxDataModel() )
        return;

    // with RTX allow the SBT index to propagate up to the first group.
    notifyParents_SBTIndexDidChange( oldSBTIndex, newSBTIndex );
}

//------------------------------------------------------------------------
// Traversable support
//------------------------------------------------------------------------
GraphNode::TravSource Transform::getTraversableSource() const
{
    return m_keyType != RT_MOTIONKEYTYPE_NONE ? TravSource::OTHER_DIRECT : TravSource::TRANSFORM_DIRECT;
}

void Transform::receivePropertyDidChange_RequiresTraversable( GraphNode* fromParent, bool attached )
{
    bool changed = m_requiresTraversable.addOrRemoveProperty( attached );
    if( changed )
    {
        if( attached )
            reallocateRecord();

        updateTraversable();
        receivePropertyDidChange_TransformRequiresTraversable( fromParent, attached );
    }
}

void Transform::receivePropertyDidChange_TransformRequiresTraversable( GraphNode* transform, bool attached )
{
    const bool changed = m_transformRequiresTraversable.addOrRemoveProperty( attached );
    if( changed && m_child )
        m_child->receivePropertyDidChange_TransformRequiresTraversable( transform, attached );
}

RtcTraversableHandle Transform::getTraversableHandle( unsigned int allDeviceIndex ) const
{
    if( !m_context->useRtxDataModel() )
        return 0;

    if( !m_traversables[allDeviceIndex] )
        return 0;

    const LWDADevice* lwdaDevice = deviceCast<const LWDADevice>( m_context->getDeviceManager()->allDevices()[allDeviceIndex] );
    if( !lwdaDevice )
        return 0;

    return getTraversableHandleForDevice( lwdaDevice );
}

bool Transform::requiresDirectTraversable() const
{
    return !m_requiresTraversable.empty();
}

void Transform::updateTraversable()
{
    if( !m_context->useRtxDataModel() )
        return;

    if( requiresDirectTraversable() )
    {
        // if the primary device has a traverable, all active devices have one
        const unsigned int primaryAllDeviceListIndex = m_context->getDeviceManager()->primaryDevice()->allDeviceListIndex();
        if( m_traversables[primaryAllDeviceListIndex] )
            resizeTraversables();
        else
            allocateTraversables();

        for( Device* dev : m_context->getDeviceManager()->activeDevices() )
        {
            writeTraversable( dev->allDeviceListIndex() );
        }
    }
    else
    {
        for( auto& traversable : m_traversables )
            traversable.reset();
    }
}

static size_t getTraversableSize( RTmotionkeytype keyType, size_t numKeys )
{
    RT_ASSERT_MSG( keyType == RT_MOTIONKEYTYPE_NONE || numKeys >= 2, "Must have at least 2 motion keys." );
    if( keyType == RT_MOTIONKEYTYPE_MATRIX_FLOAT12 )
        return sizeof( RtcMatrixMotionTransform ) + sizeof( float[12] ) * ( numKeys - 2 );
    if( keyType == RT_MOTIONKEYTYPE_SRT_FLOAT16 )
        return sizeof( RtcSRTMotionTransform ) + sizeof( float[16] ) * ( numKeys - 2 );
    // RT_MOTIONKEYTYPE_NONE
    return sizeof( RtcTravStaticTransform );
}

BufferDimensions Transform::getTraversableDimensions() const
{
    return BufferDimensions( RT_FORMAT_USER, getTraversableSize( m_keyType, getKeyCount() ), 1, 1, 1, 1 );
}

void Transform::allocateTraversables()
{
    for( Device* device : m_context->getDeviceManager()->allDevices() )
    {
        if( !device->isActive() )
            continue;

        const unsigned int allDeviceIndex = device->allDeviceListIndex();
        if( !m_traversables[allDeviceIndex] )
        {
            DeviceSet set( device );
            m_traversables[allDeviceIndex] =
                m_context->getMemoryManager()->allocateMBuffer( getTraversableDimensions(), MBufferPolicy::gpuLocal, set, this );
        }
    }
}

void Transform::resizeTraversables()
{
    const BufferDimensions desiredSize = getTraversableDimensions();
    for( Device* device : m_context->getDeviceManager()->allDevices() )
    {
        if( !device->isActive() )
            continue;

        const unsigned int allDeviceIndex = device->allDeviceListIndex();
        if( m_traversables[allDeviceIndex]->getDimensions() != desiredSize )
            m_context->getMemoryManager()->changeSize( m_traversables[allDeviceIndex], desiredSize );
    }
}

void Transform::childTraversableHandleDidChange( LinkedPtr_Link* child, TravSource source, unsigned int allDeviceIndex, RtcTraversableHandle childHandle )
{
    if( !m_context->useRtxDataModel() )
        return;

    if( hasMotionKeys() )
    {
        if( &m_child == child && requiresDirectTraversable()
            && ( source == TravSource::OTHER_DIRECT || source == TravSource::GEOMGROUP_DIRECT || source == TravSource::TRANSFORM_DIRECT ) )
        {
            allocateTraversables();
            writeTraversable( allDeviceIndex, childHandle );
        }

        // In the case of motion blur, the traversable will not get forwarded.
    }
    else
    {
        if( requiresDirectTraversable() && ( source == TravSource::OTHER_DIRECT || source == TravSource::GEOMGROUP_DIRECT ) )
        {
            allocateTraversables();
            writeTraversable( allDeviceIndex, childHandle );
        }

        forwardDidChange_TraversableHandle( source, allDeviceIndex, childHandle );
    }
}

void Transform::attachOrDetachProperty_RequiresTraversable( GraphNode* oldChild, bool added )
{
    // A motion transform requires its direct child to have a traversable because it doesn't flatten
    // a chain of child static transforms; however m_requiresTraversable can be non-empty when a
    // variable is attached to a transform and in that case, the direct child of the transform is
    // not required to have a traversable because the static transform's traversable will collapse
    // a chain of transforms.
    if( hasMotionKeys() )
    {
        oldChild->receivePropertyDidChange_RequiresTraversable( this, added );
    }
}

void Transform::attachOrDetachProperty_TransformRequiresTraversable( GraphNode* oldChild, bool added )
{
    if( !m_transformRequiresTraversable.empty() )
    {
        oldChild->receivePropertyDidChange_TransformRequiresTraversable( this, added );
    }
}

// When sending child traversable notifications, we don't want to send them up to attached
// variables, which should only hold *our* traversable handle.  That will happen through
// sendDidChange_TraversableHandle when our traversable handle is allocated.
void Transform::forwardDidChange_TraversableHandle( TravSource source, unsigned int allDeviceIndex, RtcTraversableHandle travHandle )
{
    for( auto parentLink : m_linkedPointers )
    {
        if( Selector* selector = getLinkToGraphNodeFrom<Selector>( parentLink ) )
        {
            selector->childTraversableHandleDidChange( parentLink, source, allDeviceIndex, travHandle );
        }
        else if( Transform* transform = getLinkToGraphNodeFrom<Transform>( parentLink ) )
        {
            transform->childTraversableHandleDidChange( parentLink, source, allDeviceIndex, travHandle );
        }
        else if( AbstractGroup* abstractGroup = getLinkFrom<AbstractGroup, LexicalScope>( parentLink ) )
        {
            RT_ASSERT( abstractGroup->getClass() == RT_OBJECT_GROUP );
            managedObjectCast<Group>( abstractGroup )->childTraversableHandleDidChange( parentLink, source, allDeviceIndex, travHandle );
        }
    }
}

GraphNode::TraversableDataForTest Transform::getTraversableDataForTest( unsigned int allDeviceIndex ) const
{
    RT_ASSERT( m_traversables[allDeviceIndex] );
    // TODO: handle multiple devices and use MBuffer corresponding to allDeviceIndex
    TraversableDataForTest data{};
    data.m_size         = getTraversableSize( m_keyType, getKeyCount() );
    const void* hostPtr = m_context->getMemoryManager()->mapToHost( m_traversables[allDeviceIndex], MAP_READ );
    switch( m_keyType )
    {
        case RT_MOTIONKEYTYPE_NONE:
            data.m_type            = RTC_TRAVERSABLE_TYPE_STATIC_TRANSFORM;
            data.m_staticTransform = static_cast<const RtcTravStaticTransform*>( hostPtr );
            break;
        case RT_MOTIONKEYTYPE_MATRIX_FLOAT12:
            data.m_type                  = RTC_TRAVERSABLE_TYPE_MATRIX_MOTION_TRANSFORM;
            data.m_matrixMotionTransform = static_cast<const RtcMatrixMotionTransform*>( hostPtr );
            break;
        case RT_MOTIONKEYTYPE_SRT_FLOAT16:
            data.m_type               = RTC_TRAVERSABLE_TYPE_SRT_MOTION_TRANSFORM;
            data.m_srtMotionTransform = static_cast<const RtcSRTMotionTransform*>( hostPtr );
            break;
    }
    data.m_traversableId = m_traversableId ? *m_traversableId : -1;

    return data;
}

// The GPU address of our traversable changed
void Transform::eventMBufferMAccessDidChange( const MBuffer* buffer, const Device* device, const MAccess& oldMBA, const MAccess& newMBA )
{
    const LWDADevice* lwdaDevice = deviceCast<const LWDADevice>( device );
    if( !lwdaDevice )
        return;

    const unsigned int         allDeviceIndex = device->allDeviceListIndex();
    const RtcTraversableHandle travHandle     = getTraversableHandleFromMAccess( lwdaDevice, newMBA );
    sendDidChange_TraversableHandle( getTraversableSource(), allDeviceIndex, travHandle );
}

RtcTraversableHandle Transform::getTraversableHandleForDevice( const LWDADevice* lwdaDevice ) const
{
    return getTraversableHandleFromMAccess( lwdaDevice, m_traversables[lwdaDevice->allDeviceListIndex()]->getAccess( lwdaDevice ) );
}

RtcTraversableHandle Transform::getTraversableHandleFromMAccess( const LWDADevice* lwdaDevice, const MAccess& access ) const
{
    RtcTraversableHandle travHandle = 0;
    if( access.getKind() == MAccess::LINEAR )
    {
        RtcGpuVA devPtr = reinterpret_cast<RtcGpuVA>( access.getLinearPtr() );

        // Only universal traversal directly uses the traversable handle of a transform during traversal
        RtcTraversableType travType = RTC_TRAVERSABLE_TYPE_STATIC_TRANSFORM;
        if( m_keyType == RT_MOTIONKEYTYPE_MATRIX_FLOAT12 )
            travType = RTC_TRAVERSABLE_TYPE_MATRIX_MOTION_TRANSFORM;
        else if( m_keyType == RT_MOTIONKEYTYPE_SRT_FLOAT16 )
            travType = RTC_TRAVERSABLE_TYPE_SRT_MOTION_TRANSFORM;
        m_context->getRTCore()->colwertPointerToTraversableHandle( lwdaDevice->rtcContext(), devPtr, travType,
                                                                   RTC_ACCEL_TYPE_NOACCEL, &travHandle );
    }
    return travHandle;
}

void Transform::writeTraversable( unsigned int allDeviceIndex )
{
    writeTraversable( allDeviceIndex, m_child ? m_child->getTraversableHandle( allDeviceIndex ) : 0U );
}

static void copyMatrix4x3( RtcF32 dest[12], const Matrix4x4& src )
{
    const float* srcData = src.getData();
    std::copy( &srcData[0], &srcData[12], &dest[0] );
};

void Transform::writeTraversable( unsigned int allDeviceIndex, RtcTraversableHandle childHandle )
{
    MemoryManager* mm      = m_context->getMemoryManager();
    char*          address = mm->mapToHost( m_traversables[allDeviceIndex], MAP_WRITE_DISCARD );
    switch( m_keyType )
    {
        case RT_MOTIONKEYTYPE_NONE:
        {
            RtcTravStaticTransform* data = reinterpret_cast<RtcTravStaticTransform*>( address );
            data->child                  = childHandle;
            const Matrix4x4 transform    = getInstanceTransform();
            copyMatrix4x3( data->transform, transform );
            copyMatrix4x3( data->ilwTransform, transform.ilwerse() );
        }
        break;

        case RT_MOTIONKEYTYPE_MATRIX_FLOAT12:
        {
            RtcMatrixMotionTransform* data = reinterpret_cast<RtcMatrixMotionTransform*>( address );
            data->child                    = childHandle;
            data->numKeys                  = getKeyCount();

            RTmotionbordermode beginMode, endMode;
            getMotionBorderMode( beginMode, endMode );
            data->flags = 0;
            if( beginMode == RT_MOTIONBORDERMODE_VANISH )
                data->flags |= RTC_MOTION_FLAG_START_VANISH;
            if( endMode == RT_MOTIONBORDERMODE_VANISH )
                data->flags |= RTC_MOTION_FLAG_END_VANISH;

            data->timeBegin = m_timeBegin;
            data->timeEnd   = m_timeEnd;
            algorithm::copy( m_keys, &data->transform[0][0] );
        }
        break;

        case RT_MOTIONKEYTYPE_SRT_FLOAT16:
        {
            RtcSRTMotionTransform* data = reinterpret_cast<RtcSRTMotionTransform*>( address );
            data->child                 = childHandle;
            data->numKeys               = getKeyCount();

            RTmotionbordermode beginMode, endMode;
            getMotionBorderMode( beginMode, endMode );
            data->flags = 0;
            if( beginMode == RT_MOTIONBORDERMODE_VANISH )
                data->flags |= RTC_MOTION_FLAG_START_VANISH;
            if( endMode == RT_MOTIONBORDERMODE_VANISH )
                data->flags |= RTC_MOTION_FLAG_END_VANISH;

            data->timeBegin = m_timeBegin;
            data->timeEnd   = m_timeEnd;
            algorithm::copy( m_keys, &data->quaternion[0][0] );

            // bake rotation pivot point into translation and scale matrix
            for( unsigned int i = 0; i < data->numKeys; ++i )
            {
                data->quaternion[i][13] += data->quaternion[i][3];
                data->quaternion[i][14] += data->quaternion[i][6];
                data->quaternion[i][15] += data->quaternion[i][8];
                data->quaternion[i][3] = -data->quaternion[i][3];
                data->quaternion[i][6] = -data->quaternion[i][6];
                data->quaternion[i][8] = -data->quaternion[i][8];
            }
        }
        break;
    }
    mm->unmapFromHost( m_traversables[allDeviceIndex] );
}

//------------------------------------------------------------------------
// Trace Caller
//------------------------------------------------------------------------
void Transform::sendPropertyDidChange_TraceCaller( CanonicalProgramID cpid, bool added ) const
{
    if( m_child )
        m_child->receivePropertyDidChange_TraceCaller( cpid, added );
}


//------------------------------------------------------------------------
// Helper functions
//------------------------------------------------------------------------

GraphNode* Transform::getNonTransformChild( Matrix4x4* joinedMatrix ) const
{
    Matrix4x4  result = m_transform;
    GraphNode* child  = getChild();
    while( child->getClass() == RT_OBJECT_TRANSFORM )
    {
        const Transform* trans = static_cast<const Transform*>( child );
        result *= trans->m_transform;
        child = trans->getChild();
    }
    if( joinedMatrix )
        *joinedMatrix = result;
    return child;
}

// Affine transformation of a point.
static inline float3 transform( const float3& v, const float m[16] )
{
    float3 ret;
    ret.x = v.x * m[0] + v.y * m[1] + v.z * m[2] + m[3];
    ret.y = v.x * m[4] + v.y * m[5] + v.z * m[6] + m[7];
    ret.z = v.x * m[8] + v.y * m[9] + v.z * m[10] + m[11];
    return ret;
}

static void adjust_difference_aabb( Aabb& difference_aabb, float t, float val, unsigned int dim, const Aabb& transf_aabb0, const Aabb& transf_aabb1 )
{
    float interp_min =
        *( &transf_aabb0.m_min.x + dim ) + ( *( &transf_aabb1.m_min.x + dim ) - *( &transf_aabb0.m_min.x + dim ) ) * t;
    float interp_max =
        *( &transf_aabb0.m_max.x + dim ) + ( *( &transf_aabb1.m_max.x + dim ) - *( &transf_aabb0.m_max.x + dim ) ) * t;
    float diff = val - interp_min;
    if( diff < *( &difference_aabb.m_min.x + dim ) )
        *( &difference_aabb.m_min.x + dim ) = diff;
    diff                                    = val - interp_max;
    if( diff > *( &difference_aabb.m_max.x + dim ) )
        *( &difference_aabb.m_max.x + dim ) = diff;
}

// Enlarge aabbs by differences given in difference_aabbs.

static void add_difference( Aabb& aabb0, Aabb& aabb1, const Aabb& difference_aabb )
{
    for( unsigned int dim = 0; dim < 3; ++dim )
    {
        if( *( &difference_aabb.m_max.x + dim ) > 0.f )
        {
            *( &aabb0.m_max.x + dim ) += *( &difference_aabb.m_max.x + dim );
            *( &aabb1.m_max.x + dim ) += *( &difference_aabb.m_max.x + dim );
        }
        if( *( &difference_aabb.m_min.x + dim ) < 0.f )
        {
            *( &aabb0.m_min.x + dim ) += *( &difference_aabb.m_min.x + dim );
            *( &aabb1.m_min.x + dim ) += *( &difference_aabb.m_min.x + dim );
        }
    }
}

// Given an aabb, extract its 8 corner points.

static inline void extractCornersFromAabb( float3* v, const Aabb& aabb )
{
    v[0] = make_float3( aabb[0].x, aabb[0].y, aabb[0].z );
    v[1] = make_float3( aabb[0].x, aabb[0].y, aabb[1].z );
    v[2] = make_float3( aabb[0].x, aabb[1].y, aabb[0].z );
    v[3] = make_float3( aabb[0].x, aabb[1].y, aabb[1].z );
    v[4] = make_float3( aabb[1].x, aabb[0].y, aabb[0].z );
    v[5] = make_float3( aabb[1].x, aabb[0].y, aabb[1].z );
    v[6] = make_float3( aabb[1].x, aabb[1].y, aabb[0].z );
    v[7] = make_float3( aabb[1].x, aabb[1].y, aabb[1].z );
}

// Given two aabbs at time 0 and time 1 and its linear interpolation for t between 0 and 1,
// apply the transformation given by the keys to it. (transformation on moving input)
// It's not sufficient to apply key0 on aabb0 and key1 on aabb1, in addition these aabbs need to be enlarged
// to guarantee that motion for all t between 0 and 1 is covered by linearly interpolating the aabbs.

static void transform( const Aabb& aabb0, const Aabb& aabb1, const float* key0, const float* key1, Aabb& outaabb0, Aabb& outaabb1 )
{
    // v0 vertices of aabb0, v1 vertices from aabb1
    float3 v0[8], v1[8];
    extractCornersFromAabb( v0, aabb0 );
    extractCornersFromAabb( v1, aabb1 );

    float3 d_v[8] = {v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2], v1[3] - v0[3],
                     v1[4] - v0[4], v1[5] - v0[5], v1[6] - v0[6], v1[7] - v0[7]};

    float d_k[12] = {key1[0] - key0[0], key1[1] - key0[1], key1[2] - key0[2],   key1[3] - key0[3],
                     key1[4] - key0[4], key1[5] - key0[5], key1[6] - key0[6],   key1[7] - key0[7],
                     key1[8] - key0[8], key1[9] - key0[9], key1[10] - key0[10], key1[11] - key0[11]};

    // transformation of the motion aabb at time 0 and time 1
    Aabb transf_aabb0 = transform( aabb0, key0 );
    Aabb transf_aabb1 = transform( aabb1, key1 );

    // difference_aabb will contain the the necessary enlargement of transf_aabb0 and transf_aabb1,
    // such that the linear interpolation between these two will contain all (non-linear) motion between 0 and 1
    Aabb difference_aabb;

    // For all 8 corners check whether its quadratic motion path exceeds the interpolation, separately for x, y, z.
    // This is done by computing the extremum of the difference of the motion path
    // and the interpolation of the end points of the motion path. The extremum is checked whether its inside
    // the interpolated transformed aabbs, otherwise the transformed aabbs at time 0 and 1 are extended
    // such that the interpolation contain the intermediate motion. This gives tighter aabbs than just using the union of
    // the input aabbs as replacement for the input aabbs at 0 and 1.

    // point i
    for( unsigned int i = 0; i < 8; ++i )
    {
        // dimension X, Y, or Z
        for( unsigned int dim = 0; dim < 3; ++dim )
        {
            const unsigned int off = dim * 4;

            float denom = d_v[i].x * d_k[off + 0] + d_v[i].y * d_k[off + 1] + d_v[i].z * d_k[off + 2];
            if( denom != 0.f )
            {
                float num = v0[i].x * d_k[off + 0] + d_v[i].x * key0[off + 0] + v0[i].y * d_k[off + 1]
                            + d_v[i].y * key0[off + 1] + v0[i].z * d_k[off + 2] + d_v[i].z * key0[off + 2];

                float t = -0.5f * ( num - ( *( &transf_aabb1.m_min.x + dim ) - *( &transf_aabb0.m_min.x + dim ) ) ) / denom;
                if( t > 0.f && t < 1.f )
                {
                    float val = ( v0[i].x + d_v[i].x * t ) * ( key0[off + 0] + d_k[off + 0] * t )
                                + ( v0[i].y + d_v[i].y * t ) * ( key0[off + 1] + d_k[off + 1] * t )
                                + ( v0[i].z + d_v[i].z * t ) * ( key0[off + 2] + d_k[off + 2] * t ) + key0[off + 3]
                                + d_k[off + 3] * t;
                    adjust_difference_aabb( difference_aabb, t, val, dim, transf_aabb0, transf_aabb1 );
                }

                t = -0.5f * ( num - ( *( &transf_aabb1.m_max.x + dim ) - *( &transf_aabb0.m_max.x + dim ) ) ) / denom;
                if( t > 0.f && t < 1.f )
                {
                    float val = ( v0[i].x + d_v[i].x * t ) * ( key0[off + 0] + d_k[off + 0] * t )
                                + ( v0[i].y + d_v[i].y * t ) * ( key0[off + 1] + d_k[off + 1] * t )
                                + ( v0[i].z + d_v[i].z * t ) * ( key0[off + 2] + d_k[off + 2] * t ) + key0[off + 3]
                                + d_k[off + 3] * t;
                    adjust_difference_aabb( difference_aabb, t, val, dim, transf_aabb0, transf_aabb1 );
                }
            }
        }
    }

    // enlarge transf_aabb0 and transf_aabb1 to have its interpolation contain motion for all t between 0 and 1

    add_difference( transf_aabb0, transf_aabb1, difference_aabb );

    outaabb0 = transf_aabb0;
    outaabb1 = transf_aabb1;
}

// Affine transformation of a bounding box,
// using matrix transform.
static inline Aabb transform( const Aabb& aabb, const float m[16] )
{
    float3 p[8];
    extractCornersFromAabb( p, aabb );

    Aabb ret;
    for( float3 i : p )
        ret.include( transform( i, m ) );
    return ret;
}

static optix::Matrix4x4 lerp( const optix::Matrix4x4& m0, const optix::Matrix4x4& m1, float t )
{
    optix::Matrix4x4 m;
    float*           result = m.getData();
    const float*     a      = m0.getData();
    const float*     b      = m1.getData();

    for( int i    = 0; i < 16; ++i )
        result[i] = optix::lerp( a[i], b[i], t );

    return m;
}

static float3 eval_sr( const optix::Matrix4x4& scale0, const optix::Matrix4x4& scale1, float4 quat0, float4 quat1, float t, float3 p )
{
    // lerp scale/shear matrix
    const optix::Matrix4x4 scale = lerp( scale0, scale1, t );

    // nlerp rotation as quaternion
    Quaternion q0( quat0 );
    Quaternion q1( quat1 );
    Quaternion rotate = optix::nlerp( q0, q1, t );

    return make_float3( rotate * ( scale * make_float4( p, 1.0f ) ) );
}

static float3 eval_srt( const Matrix4x4& scale, float4 quat, float3 trans, float3 p )
{
    // scale/shear matrix "scale"
    // quaternion "rotate" built from quad
    const optix::Quaternion rotate( quat.x, quat.y, quat.z, quat.w );
    // translate "trans"
    return trans + make_float3( rotate * ( scale * make_float4( p, 1.0f ) ) );
}

static inline Matrix4x4 getScaleMatrixFromSrtKey( const float key[] )
{
    // clang-format off
    // scale and shear entries from key
    const float scale_data[16] = { key[0], key[1], key[2], -key[3], 0.f, key[4], key[5], -key[6],
                                    0.f, 0.f, key[7], -key[8], 0.f, 0.f, 0.f, 1.f };
    Matrix4x4 scale;
    for (int i = 0; i < 16; ++i)
      scale.getData()[i] = scale_data[i];
    return scale;
    // clang-format on
}

static void includeSRCornerPath( Aabb&             ret,
                                 float3            p,
                                 const Matrix4x4&  scale0,
                                 const Matrix4x4&  scale1,
                                 const float4      quat0,
                                 const float4      quat1,
                                 float             t0,
                                 float             t1,
                                 SrtDerivativeTerm c0[],
                                 SrtDerivativeTerm c1[],
                                 SrtDerivativeTerm c2[],
                                 SrtDerivativeTerm c3[],
                                 SrtDerivativeTerm c4[] )
{
    // Specific input point p (e.g., one corner of an aabb) to be transformed

    // Evaluate coefficients for each axis with given input point.
    // 3: x,y,z
    // 5: coefficients of polynomial c(t)
    float coeffs[3][5];
    for( int k = 0; k < 3; ++k )
    {
        coeffs[k][0] = c0[k].eval( p );
        coeffs[k][1] = c1[k].eval( p );
        coeffs[k][2] = c2[k].eval( p );
        coeffs[k][3] = c3[k].eval( p );
        coeffs[k][4] = c4[k].eval( p );
    }

    // root finding
    float3 boxmin = make_float3( std::numeric_limits<float>::max() );
    float3 boxmax = make_float3( -std::numeric_limits<float>::max() );
    float* bmin   = &boxmin.x;
    float* bmax   = &boxmax.x;
    for( int axis = 0; axis < 3; ++axis )
    {
        float     roots[4];
        const int numroots = findRealRoots4( coeffs[axis], roots );

        // Evaluate at roots that are inside segment time interval
        for( int i = 0; i < numroots; ++i )
        {
            if( roots[i] > t0 && roots[i] < t1 )
            {
                // evaluate RS*p at this root and update min and max for this axis
                float3 pt  = eval_sr( scale0, scale1, quat0, quat1, roots[i], p );
                bmin[axis] = std::min( bmin[axis], *( &pt.x + axis ) );
                bmax[axis] = std::max( bmax[axis], *( &pt.x + axis ) );
            }
        }

        // include end points of interval (t==t0 and t==t1)
        {
            float3 pt  = eval_sr( scale0, scale1, quat0, quat1, t0, p );
            bmin[axis] = std::min( bmin[axis], *( &pt.x + axis ) );
            bmax[axis] = std::max( bmax[axis], *( &pt.x + axis ) );
        }
        {
            float3 pt  = eval_sr( scale0, scale1, quat0, quat1, t1, p );
            bmin[axis] = std::min( bmin[axis], *( &pt.x + axis ) );
            bmax[axis] = std::max( bmax[axis], *( &pt.x + axis ) );
        }
    }
    // Pad bounds to account for error in root finding.  Otherwise error check below shows
    // that some points near the root are outside the bounds.
    boxmin = boxmin - 0.00001f * ( boxmax - boxmin );
    boxmax = boxmax + 0.00001f * ( boxmax - boxmin );

    ret.include( boxmin );
    ret.include( boxmax );
}

// static input aabb
// NO translation
// interpolated srt: key0 -> key1
// time interval with t0,t1 in [0;1] and srt0 is at t=0 and srt1 is at t=1, i.e., t0,t1 are times within the timespan of key0, key1
// result: aabb of interpolated srt applied on input aabb within interval [t0, t1]
static Aabb transformSR( const Aabb& aabb, const float* key0, const float* key1, float t0, float t1 )
{
    // 8 corner points of axis-aligned bounding box
    float3 v[8];
    extractCornersFromAabb( v, aabb );

    // Made up test data using pivot point for rotation:
    // R*(P*S), where P is pivot

    Matrix4x4 scale0 = getScaleMatrixFromSrtKey( key0 );
    Matrix4x4 scale1 = getScaleMatrixFromSrtKey( key1 );

    const float4 quat0 = make_float4( key0[9], key0[10], key0[11], key0[12] );
    const float4 quat1 = make_float4( key1[9], key1[10], key1[11], key1[12] );

    SrtDerivativeTerm c0[3], c1[3], c2[3], c3[3], c4[3];
    makeSrtDerivativeTerms( scale0.getData(), scale1.getData(), quat0, quat1, c0, c1, c2, c3, c4 );

    Aabb ret;

    for( float3 i : v )
        includeSRCornerPath( ret, i, scale0, scale1, quat0, quat1, t0, t1, c0, c1, c2, c3, c4 );

    return ret;
}

// transformSR transforms moving input aabbs, the interpolation between aabbb0 and aabb1,
// with an SR transform specified by key0 and key1. T is handled separately outside this function.
//
// interpolated srt: key0 at time 0 -> key1 at time 1
// interpolated input aabb: aabb0 is at time t0, aabb1 is at time t1, with t0, t1 between 0 and 1
// result: aabb of interpolated srt applied on input aabb within interval [t0, t1]
static Aabb transformSR( const Aabb& aabb0, const Aabb& aabb1, const float* key0, const float* key1, float t0, float t1 )
{
    // TODO: make use of t0, t1

    if( aabb0.contains( aabb1 ) )
        return transformSR( aabb0, key0, key1, t0, t1 );
    else if( aabb1.contains( aabb0 ) )
        return transformSR( aabb1, key0, key1, t0, t1 );

    // corner points of aabb0 and aabb1
    float3 v0[8], v1[8];
    extractCornersFromAabb( v0, aabb0 );
    extractCornersFromAabb( v1, aabb1 );

    float3 diff_v[8] = {v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2], v1[3] - v0[3],
                        v1[4] - v0[4], v1[5] - v0[5], v1[6] - v0[6], v1[7] - v0[7]};

    Matrix4x4 scale0 = getScaleMatrixFromSrtKey( key0 );
    Matrix4x4 scale1 = getScaleMatrixFromSrtKey( key1 );

    const float4 quat0 = make_float4( key0[9], key0[10], key0[11], key0[12] );
    const float4 quat1 = make_float4( key1[9], key1[10], key1[11], key1[12] );

    Aabb ret;

    for( int i = 0; i < 8; ++i )
    {
        Matrix4x4 scale1_m = Matrix4x4::translate( diff_v[i] ) * scale1;

        SrtDerivativeTerm c0[3], c1[3], c2[3], c3[3], c4[3];
        makeSrtDerivativeTerms( scale0.getData(), scale1_m.getData(), quat0, quat1, c0, c1, c2, c3, c4 );

        includeSRCornerPath( ret, v0[i], scale0, scale1_m, quat0, quat1, t0, t1, c0, c1, c2, c3, c4 );
    }

    return ret;
}

static inline void applyTranslationfromSRT( const float* key0, const float* key1, float t0, float t1, Aabb& inoutAabb0, Aabb& inoutAabb1 )
{
    const float3 trans0 = make_float3( key0[13] + key0[3], key0[14] + key0[6], key0[15] + key0[8] );
    const float3 trans1 = make_float3( key1[13] + key1[3], key1[14] + key1[6], key1[15] + key1[8] );
    inoutAabb0.m_min += lerp( trans0, trans1, t0 );
    inoutAabb0.m_max += lerp( trans0, trans1, t0 );
    inoutAabb1.m_min += lerp( trans0, trans1, t1 );
    inoutAabb1.m_max += lerp( trans0, trans1, t1 );
}

static inline void transformSRT( const Aabb& aabb, const float* key0, const float* key1, Aabb& outAabb0, Aabb& outAabb1 )
{
    transformSRT( aabb, key0, key1, 0.0f, 1.0f, outAabb0, outAabb1 );
}

static inline void transformSRT( const Aabb& aabb, const float* key0, const float* key1, float t0, float t1, Aabb& outAabb0, Aabb& outAabb1 )
{
    // factored translation
    outAabb0 = outAabb1 = transformSR( aabb, key0, key1, t0, t1 );
    applyTranslationfromSRT( key0, key1, t0, t1, outAabb0, outAabb1 );
}

static inline void transformSRT( const Aabb& aabb0, const Aabb& aabb1, const float* key0, const float* key1, float t0, float t1, Aabb& outAabb0, Aabb& outAabb1 )
{
    // factored translation
    outAabb0 = outAabb1 = transformSR( aabb0, aabb1, key0, key1, t0, t1 );
    applyTranslationfromSRT( key0, key1, t0, t1, outAabb0, outAabb1 );
}

static inline Aabb transformSRT( const Aabb& aabb, const float* key )
{
    Matrix4x4    scale = getScaleMatrixFromSrtKey( key );
    const float3 trans = make_float3( key[13] + key[3], key[14] + key[6], key[15] + key[8] );
    const float4 quat  = make_float4( key[9], key[10], key[11], key[12] );

    // 8 corner points of bounding box
    float3 v[8];
    extractCornersFromAabb( v, aabb );

    Aabb ret;
    for( float3 i : v )
        ret.include( eval_srt( scale, quat, trans, i ) );

    return ret;
}

void Transform::handleIrregularKeys( const MotionAabb& inputMaabb, const unsigned int keycount, bool addIrregularTimes, MotionAabb& maabb ) const
{
    // scale epsilon based on time range of input / transform
    const float keys_align_epsilon =
        0.00001f * std::max( inputMaabb.timeLastKey() - inputMaabb.timeFirstKey(), m_timeEnd - m_timeBegin );

    // "auto detect" of the output keys are regularly distributed
    // will be set once we have the first interval (i.e., on the second key)
    // all further intervals will be checked against this interval
    // auto detection is not needed if it was previously "decided" (detected) that output is regular
    bool  maabbHasRegularDistribution = true;
    float detectedIntervalTime        = 0.0f;

    maabb.aabbs().reserve( inputMaabb.keyCount() + keycount );
    if( addIrregularTimes )
        maabb.keyTimes().reserve( inputMaabb.keyCount() + keycount );

    // General idea:
    // We "march" in time from key to key, always considering the next (time-wise) key whether this is an input or a transform key.
    // As a result, we generally handle intervals and return the aabb for the beginning of the interval and the aabb at the end of the interval such that
    //  the aabbs (outAabb0, outAabb1) can be linearly interpolated and fully contain, i.e., lerp(outAabb0, outAabb1, t) >= transform(key0, key1, t) * lerp(inAabb0, inAabb1, t)
    const float  transformIntervalTime = ( m_timeEnd - m_timeBegin ) / ( keycount - 1 );
    unsigned int nextTransformKey      = 0;
    unsigned int nextInputKey          = 0;
    float        prevTransformKeyTime  = m_timeBegin;
    float        nextTransformKeyTime  = m_timeBegin;
    float        nextInputKeyTime      = inputMaabb.keyTime( nextInputKey );
    float        prevKeyTime;
    bool         nextKeyIsTranformKey, prevKeyWasTransformKey, prevKeyWasInputKey;

    // The following functions are there to avoid code duplication. We end up with a single marching (looping the keys) code while handling the similar, but yet different
    //  transform types (matrix vs. SRT).
    // The std::functions can be seen as different implementations for all the special cases during the marching.
    //
    // Matrix of possibilities
    // next transform == next key to process is a transform key, i.e., interval goes from previous key to transform key
    // next input == next key to process is an input key, i.e., interval goes from previous key to input key
    // next merge == next transform key and input key align, i.e., interval goes from previous key to transform/input key
    std::function<void( void )>                 nextTransformPreInput, nextTransformPastInput, nextTransformInterpolate;
    std::function<void( void )>                 nextInputPreTransform, nextInputPastTransform, nextInputInterpolate;
    std::function<void( void )>                 nextMergeKeys;
    std::function<const float*( unsigned int )> getKey;

    // helper functions that need to be declared here, otherwise they are out of scope, i.e., empty
    std::function<void( const Aabb&, const float* )> handleIntervalMatrixTransform;
    std::function<void( const Aabb& )> handleIntervalSRTTransform;
    std::function<void( unsigned int, unsigned int, float, float* )> lerpMatrixTransformKeys;

    if( m_keyType == RT_MOTIONKEYTYPE_MATRIX_FLOAT12 )
    {
        getKey                  = [&]( unsigned int keyIndex ) -> const float* { return &m_keys[keyIndex * 12]; };
        lerpMatrixTransformKeys = [&]( unsigned int index_key0, unsigned int index_key1, float t, float* interpolatedTransform ) {
            const float* key0         = getKey( index_key0 );
            const float* key1         = getKey( index_key1 );
            interpolatedTransform[0]  = lerp( key0[0], key1[0], t );
            interpolatedTransform[1]  = lerp( key0[1], key1[1], t );
            interpolatedTransform[2]  = lerp( key0[2], key1[2], t );
            interpolatedTransform[3]  = lerp( key0[3], key1[3], t );
            interpolatedTransform[4]  = lerp( key0[4], key1[4], t );
            interpolatedTransform[5]  = lerp( key0[5], key1[5], t );
            interpolatedTransform[6]  = lerp( key0[6], key1[6], t );
            interpolatedTransform[7]  = lerp( key0[7], key1[7], t );
            interpolatedTransform[8]  = lerp( key0[8], key1[8], t );
            interpolatedTransform[9]  = lerp( key0[9], key1[9], t );
            interpolatedTransform[10] = lerp( key0[10], key1[10], t );
            interpolatedTransform[11] = lerp( key0[11], key1[11], t );
        };

        // if the input OR the transform is static for an interval, we can simply transform at the output key, interpolation from the previous/to the next aabb will be correct
        // otherwise, we need to compute bounds for the interval and update the previous aabb

        // static input
        nextTransformPreInput = [&]() {
            RT_ASSERT( nextInputKey == 0 );
            maabb.aabbs().push_back( transform( inputMaabb.aabb( nextInputKey ), getKey( nextTransformKey ) ) );
        };
        // static input
        nextTransformPastInput = [&]() {
            RT_ASSERT( nextInputKey == inputMaabb.keyCount() );
            maabb.aabbs().push_back( transform( inputMaabb.aabb( nextInputKey - 1 ), getKey( nextTransformKey ) ) );
        };

        // static transform
        nextInputPreTransform = [&]() {
            RT_ASSERT( nextTransformKey == 0 );
            maabb.aabbs().push_back( transform( inputMaabb.aabb( nextInputKey ), getKey( nextTransformKey ) ) );
        };
        // static transform
        nextInputPastTransform = [&]() {
            RT_ASSERT( nextTransformKey == keycount );
            maabb.aabbs().push_back( transform( inputMaabb.aabb( nextInputKey ), getKey( nextTransformKey - 1 ) ) );
        };

        // helper function  for the cases below
        // inAabb0, key0 depend on whether the previous key was a transform or an input key
        // inAabb1, key1 are provided as parameters
        handleIntervalMatrixTransform = [&]( const Aabb& inAabb1, const float* key1 ) {
            // if we are processing the first SRT key, we do not need to consider an interval, just transform the key
            // the same is true if we are outputting the first key, i.e., there is no interval to consider
            // note that due to vanish border mode handling, we may process the first SRT key with nextTransformKey != 0
            if( nextTransformKey == 0 || nextInputKey == 0 || maabb.aabbs().empty() )
                maabb.aabbs().push_back( transform( inAabb1, key1 ) );
            else
            {
                const float* key0;
                float        interpolatedTransform[12];
                if( prevKeyWasTransformKey )
                {
                    key0 = getKey( nextTransformKey - 1 );
                }
                else
                {
                    const float t = ( prevKeyTime - prevTransformKeyTime ) / transformIntervalTime;
                    lerpMatrixTransformKeys( nextTransformKey - 1, nextTransformKey, t, interpolatedTransform );
                    key0 = interpolatedTransform;
                }
                const Aabb* pinAabb0;
                Aabb        inAabb0;
                if( prevKeyWasInputKey )
                {
                    pinAabb0 = &inputMaabb.aabb( nextInputKey - 1 );
                }
                else
                {
                    inAabb0  = inputMaabb.interpolateKeys( nextInputKey - 1, nextInputKey, prevKeyTime );
                    pinAabb0 = &inAabb0;
                }
                Aabb outAabb0, outAabb1;
                transform( *pinAabb0, inAabb1, key0, key1, outAabb0, outAabb1 );
                maabb.aabbs().back().include( outAabb0 );
                maabb.aabbs().push_back( outAabb1 );
            }
        };
        nextMergeKeys = [&]() {
            // merge case... next transform/input keys align
            // next key is input key, i.e., pass it as inAabb1
            // next key is transform key, i.e., pass it as key1
            handleIntervalMatrixTransform( inputMaabb.aabb( nextInputKey ), getKey( nextTransformKey ) );
        };
        nextTransformInterpolate = [&]() {
            RT_ASSERT( nextInputKey > 0 || nextInputKey < inputMaabb.keyCount() );
            // interpolate input "keys" (aabbs) for t=nextTransformKeyTime
            const Aabb inAabb1 = inputMaabb.interpolateKeys( nextInputKey - 1, nextInputKey, nextTransformKeyTime );
            // next key is transform key, i.e., pass it as key1
            handleIntervalMatrixTransform( inAabb1, getKey( nextTransformKey ) );
        };
        nextInputInterpolate = [&]() {
            RT_ASSERT( nextTransformKey > 0 || nextTransformKey < keycount );
            // next key is input key, i.e., pass it as inAabb1
            const Aabb& inAabb1 = inputMaabb.aabb( nextInputKey );
            // interpolate key1
            float       key1[12];
            const float t = ( nextInputKeyTime - prevTransformKeyTime ) / transformIntervalTime;
            lerpMatrixTransformKeys( nextTransformKey - 1, nextTransformKey, t, key1 );
            handleIntervalMatrixTransform( inAabb1, key1 );
        };
    }
    else
    {
        // SRT (m_keyType == RT_MOTIONKEYTYPE_MATRIX_FLOAT16)
        getKey = [&]( unsigned int keyIndex ) -> const float* { return &m_keys[keyIndex * 16]; };

        // static input
        nextTransformPreInput = [&]() {
            RT_ASSERT( nextInputKey == 0 );
            // if we are processing the first SRT key, we do not need to consider an interval, just transform the key
            // the same is true if we are outputting the first key, i.e., there is no interval to consider
            // note that due to vanish border mode handling, we may process the first SRT key with nextTransformKey != 0
            if( nextTransformKey == 0 || maabb.aabbs().empty() )
                // static transform
                maabb.aabbs().push_back( transformSRT( inputMaabb.aabb( nextInputKey ), getKey( nextTransformKey ) ) );
            else
            {
                Aabb outAabb0, outAabb1;
                transformSRT( inputMaabb.aabb( nextInputKey ), getKey( nextTransformKey - 1 ),
                              getKey( nextTransformKey ), outAabb0, outAabb1 );
                maabb.aabbs().back().include( outAabb0 );
                maabb.aabbs().push_back( outAabb1 );
            }
        };
        // static input
        nextTransformPastInput = [&]() {
            RT_ASSERT( nextInputKey == inputMaabb.keyCount() );
            // if we are processing the first SRT key, we do not need to consider an interval, just transform the key
            // the same is true if we are outputting the first key, i.e., there is no interval to consider
            // note that due to vanish border mode handling, we may process the first SRT key with nextTransformKey != 0
            // HOWEVER, this is just a safe guard here as we this function assumes we have process all input keys, i.e.,
            //  we must have created output aabbs when processing them!
            if( nextTransformKey == 0 || maabb.aabbs().empty() )
                // static transform
                maabb.aabbs().push_back( transformSRT( inputMaabb.aabb( nextInputKey - 1 ), getKey( nextTransformKey ) ) );
            else
            {
                // prev key may have been an input key, hence, we have an interval from last input key to next srt key
                // if(prevKeyWasTransformKey) srt_interval_t0 = 0.0f;
                const float srt_interval_t0 = ( prevKeyTime - prevTransformKeyTime ) / transformIntervalTime;
                Aabb        outAabb0, outAabb1;
                transformSRT( inputMaabb.aabb( nextInputKey - 1 ), getKey( nextTransformKey - 1 ),
                              getKey( nextTransformKey ), srt_interval_t0, 1.0f, outAabb0, outAabb1 );
                maabb.aabbs().back().include( outAabb0 );
                maabb.aabbs().push_back( outAabb1 );
            }
        };

        // static transform
        nextInputPreTransform = [&]() {
            RT_ASSERT( nextTransformKey == 0 );
            maabb.aabbs().push_back( transformSRT( inputMaabb.aabb( nextInputKey ), getKey( nextTransformKey ) ) );
        };
        // static transform
        nextInputPastTransform = [&]() {
            RT_ASSERT( nextTransformKey == keycount );
            maabb.aabbs().push_back( transformSRT( inputMaabb.aabb( nextInputKey ), getKey( nextTransformKey - 1 ) ) );
        };

        // helper function  for the cases below
        // inAabb0 is computed if required
        // inAabb1 is provided as parameter and is assumed to be extrapolated to nextTransformKeyTime
        // key0, key1 are always the previous/next transform keys as transformSrtAabb() always requires these inputs
        handleIntervalSRTTransform = [&]( const Aabb& inAabb1 ) {
            // if we are processing the first SRT key, we do not need to consider an interval, just transform the key
            // the same is true if we are outputting the first key, i.e., there is no interval to consider
            // note that due to vanish border mode handling, we may process the first SRT key with nextTransformKey != 0
            if( nextTransformKey == 0 || maabb.aabbs().empty() )
                // static transform
                // key1 == get_key(nextTransformKey)
                maabb.aabbs().push_back( transformSRT( inAabb1, getKey( nextTransformKey ) ) );
            else if( nextInputKey == 0 )
            {
                // static input
                Aabb outAabb0, outAabb1;
                transformSRT( inAabb1, getKey( nextTransformKey - 1 ), getKey( nextTransformKey ), outAabb0, outAabb1 );
                maabb.aabbs().back().include( outAabb0 );
                maabb.aabbs().push_back( outAabb1 );
            }
            else
            {
                // time interval within srt0, srt1
                float srt_interval_t0, srt_interval_t1;
                if( prevKeyWasTransformKey )
                {
                    srt_interval_t0 = 0.0f;
                }
                else
                {
                    srt_interval_t0 = ( prevKeyTime - prevTransformKeyTime ) / transformIntervalTime;
                }
                if( nextKeyIsTranformKey )
                    srt_interval_t1 = 1.0f;
                else
                    srt_interval_t1 = ( nextInputKeyTime - prevTransformKeyTime ) / transformIntervalTime;
                const Aabb* pinAabb0;
                Aabb        inAabb0;
                if( prevKeyWasInputKey )
                {
                    pinAabb0 = &inputMaabb.aabb( nextInputKey - 1 );
                }
                else
                {
                    inAabb0  = inputMaabb.interpolateKeys( nextInputKey - 1, nextInputKey, prevKeyTime );
                    pinAabb0 = &inAabb0;
                }

                Aabb outAabb0, outAabb1;
                transformSRT( *pinAabb0, inAabb1, getKey( nextTransformKey - 1 ), getKey( nextTransformKey ),
                              srt_interval_t0, srt_interval_t1, outAabb0, outAabb1 );
                maabb.aabbs().back().include( outAabb0 );
                maabb.aabbs().push_back( outAabb1 );
            }
        };
        nextMergeKeys = [&]() {
            // merge case... next transform/input keys align
            // next key is input key, i.e., pass it as inAabb1
            handleIntervalSRTTransform( inputMaabb.aabb( nextInputKey ) );
        };
        nextTransformInterpolate = [&]() {
            RT_ASSERT( nextInputKey > 0 || nextInputKey < inputMaabb.keyCount() );
            // interpolate input "keys" (aabbs) for t=nextTransformKeyTime
            const Aabb inAabb1 = inputMaabb.interpolateKeys( nextInputKey - 1, nextInputKey, nextTransformKeyTime );
            handleIntervalSRTTransform( inAabb1 );
        };
        // next key is input key, therefore the behavior is the same as for merging!
        nextInputInterpolate = nextMergeKeys;
    }

    // two more helper functions to prevent code duplication
    std::function<void( void )> advanceTransformKey = [&]() {
        nextTransformKey++;
        prevKeyTime = prevTransformKeyTime = nextTransformKeyTime;
        nextTransformKeyTime               = nextTransformKey < keycount ?
                                   optix::lerp( m_timeBegin, m_timeEnd, float( nextTransformKey ) / ( keycount - 1u ) ) :
                                   std::numeric_limits<float>::max();
    };
    std::function<void( void )> advanceInputKey = [&]() {
        nextInputKey++;
        prevKeyTime      = nextInputKeyTime;
        nextInputKeyTime = nextInputKey < inputMaabb.keyCount() ? inputMaabb.keyTime( nextInputKey ) :
                                                                  std::numeric_limits<float>::max();
    };

    if( inputMaabb.borderModeBegin() == RT_MOTIONBORDERMODE_VANISH )
    {
        while( ( nextTransformKeyTime + keys_align_epsilon ) < nextInputKeyTime )
        {
            advanceTransformKey();
            prevKeyWasTransformKey = true;
            prevKeyWasInputKey     = false;
        }
    }
    if( m_beginBorderMode == RT_MOTIONBORDERMODE_VANISH )
    {
        while( ( nextInputKeyTime + keys_align_epsilon ) < nextTransformKeyTime )
        {
            advanceInputKey();
            prevKeyWasTransformKey = false;
            prevKeyWasInputKey     = true;
        }
    }

    // the actual marching (zipping) over the keys, calling the functions from above at the according times
    // make sure that the functions are set
    RT_ASSERT( nextTransformPreInput && nextTransformPastInput && nextTransformInterpolate && nextInputPreTransform
               && nextInputPastTransform && nextInputInterpolate && nextMergeKeys
               && ( handleIntervalSRTTransform || ( handleIntervalMatrixTransform && lerpMatrixTransformKeys ) ) );

    for( ; nextInputKey < inputMaabb.keyCount() || nextTransformKey < keycount; )
    {
        RT_ASSERT( nextTransformKeyTime != std::numeric_limits<float>::max()
                   || nextInputKeyTime != std::numeric_limits<float>::max() );
        nextKeyIsTranformKey = nextTransformKeyTime < nextInputKeyTime;

        if( addIrregularTimes )
        {
            const float nextKeyTime = nextKeyIsTranformKey ? nextTransformKeyTime : nextInputKeyTime;
            if( maabb.keyTimes().size() == 1 )
            {
                detectedIntervalTime = nextKeyTime - maabb.keyTimes()[0];
            }
            else if( maabb.keyTimes().size() > 1 )
            {
                maabbHasRegularDistribution &= std::abs( detectedIntervalTime - ( nextKeyTime - prevKeyTime ) ) <= keys_align_epsilon;
            }
            maabb.keyTimes().push_back( nextKeyTime );
        }

        if( nextKeyIsTranformKey )
        {
            RT_ASSERT( nextTransformKey <= keycount - 1 );
            RT_ASSERT( nextInputKey <= inputMaabb.keyCount() );
            const bool mergeKey = nextInputKeyTime - nextTransformKeyTime <= keys_align_epsilon;
            if( mergeKey )
            {
                // the input key coming after the next TransformKey is less than epsilon time away... let's merge them!
                nextMergeKeys();
                advanceInputKey();
            }
            // first input, nothing to interpolate
            else if( nextInputKey == 0 )
            {
                nextTransformPreInput();
            }
            // only one left... use that one!
            else if( nextInputKey == inputMaabb.keyCount() )
            {
                nextTransformPastInput();
            }
            else
            {
                nextTransformInterpolate();
            }
            advanceTransformKey();
            prevKeyWasTransformKey = true;
            prevKeyWasInputKey     = mergeKey;
            // we are done if we vanish is set afterwards!
            if( nextTransformKey == keycount && m_endBorderMode == RT_MOTIONBORDERMODE_VANISH )
                break;
        }
        else
        {
            RT_ASSERT( nextInputKey <= inputMaabb.keyCount() - 1 );
            RT_ASSERT( nextTransformKey <= keycount );
            const bool mergeKey = nextTransformKeyTime - nextInputKeyTime < keys_align_epsilon;
            if( mergeKey )
            {
                // the transform key coming after the next input key is less than epsilon time away... let's merge them!
                nextMergeKeys();
                advanceTransformKey();
            }
            // first input, nothing to interpolate
            else if( nextTransformKey == 0 )
            {
                nextInputPreTransform();
            }
            else if( nextTransformKey == keycount )
            {
                nextInputPastTransform();
            }
            else
            {
                nextInputInterpolate();
            }
            advanceInputKey();
            prevKeyWasInputKey     = true;
            prevKeyWasTransformKey = mergeKey;
            // we are done if we vanish is set afterwards!
            if( nextInputKey == inputMaabb.keyCount() && inputMaabb.borderModeEnd() == RT_MOTIONBORDERMODE_VANISH )
                break;
        }
    }

    if( maabbHasRegularDistribution && maabb.keyTimes().size() > 2 )
    {
        maabb.keyTimes()[1] = maabb.keyTimes().back();
        maabb.keyTimes().resize( 2 );
    }

    // if we are left with one key only, something went wrong..
    // can only happen, if VANISH clamp mode is used on transform as well as input and they have a single aligning key
    if( maabb.aabbs().size() <= 1 )
        maabb.setIlwalid();
}
