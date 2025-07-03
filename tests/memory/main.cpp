

#include "newDependence.h"
#include "memoryCfgMgrIf.hpp"


int main()
{
    CWinServicesIf* winServices = new CWinServicesIf;
    CMemoryCfgMgrIf* mcm = CMemoryCfgMgrIf::createInstance( winServices );
    MCMInit mcmInit = {};
    mcm->initialize(mcmInit);
    mcm->buildSegmentDescriptors();
    return 0;
}


