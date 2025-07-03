#pragma once

#include <Context/UpdateManager.h>
#include <Device/LWDADevice.h>
#include <ExelwtionStrategy/Common/ConstantMemAllocations.h>
#include <ExelwtionStrategy/Common/ProgramPlan.h>

#include <array>
#include <map>
#include <string>

namespace optix {

class Plan;

class ConstantMemoryPlan : public UpdateEventListenerNop
{
  public:
    typedef std::vector<std::pair<LWDADevice*, ProgramPlan*>> PerDeviceProgramPlan;

    ConstantMemoryPlan( Plan* plan, Context* context, const PerDeviceProgramPlan& perDeviceProgramPlan, size_t constBytesToReserve, bool deduplicateConstants );
    ~ConstantMemoryPlan() override;

    bool isCompatibleWith( const ConstantMemoryPlan& otherPlan ) const;
    const ConstantMemAllocations& getAllocationInfo();
    std::string                   summaryString() const;

  private:
    void createPlan( const PerDeviceProgramPlan& perDeviceProgramPlan, size_t constBytesToReserve, bool deduplcateConstants );

    bool tableSizeChanged( size_t tableSize, size_t newTableSize );

    void eventTableManagerObjectRecordResized( size_t oldSize, size_t newSize ) override;
    void eventTableManagerBufferHeaderTableResized( size_t oldSize, size_t newSize ) override;
    void eventTableManagerProgramHeaderTableResized( size_t oldSize, size_t newSize ) override;
    void eventTableManagerTextureHeaderTableResized( size_t oldSize, size_t newSize ) override;
    void eventTableManagerTraversableHeaderTableResized( size_t oldSize, size_t newSize ) override;

    void eventContextSetPreferFastRecompiles( bool oldValue, bool newValue ) override;

    Plan*    m_plan    = nullptr;
    Context* m_context = nullptr;

    ConstantMemAllocations m_constMemAllocs;

    const int CONST_MEMORY_ALIGNMENT = 16;
};
}
