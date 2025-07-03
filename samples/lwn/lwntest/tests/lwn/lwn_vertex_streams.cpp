/*
 * Copyright (c) 2016 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "lwntest_cpp.h"
#include "lwn_utils.h"

using namespace lwn;

class LWLWertexStreamsTest
{
public:
    LWNTEST_CppMethods();
};

lwString LWLWertexStreamsTest::getDescription() const
{
    lwStringBuf sb;
    sb << "Test that there's enough resources allocated even for the maximum\n"
          "supported number of vertex bindings and that nothing breaks.\n"
          "Output should be all green.";
    return sb.str();
}

int LWLWertexStreamsTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(52, 9);
}

void LWLWertexStreamsTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &cmd = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    cmd.ClearColor(0, 1.0, 0.0, 0.0, 1.0);

    int numStreams = 0;
    device->GetInteger(DeviceInfo::VERTEX_ATTRIBUTES, &numStreams);

    // Create and bind vertex stream states, as much as possible.
    VertexStreamState* vtxs = new VertexStreamState[numStreams];
    for (int i = 0; i < numStreams; i++) {
        vtxs[i].SetDefaults();
    }
    cmd.BindVertexStreamState(numStreams, vtxs);

    cmd.ClearColor(0, 0.0, 1.0, 0.0, 1.0);
    cmd.submit();
    queue->Finish();

    delete[] vtxs;
}

OGTEST_CppTest(LWLWertexStreamsTest, lwn_vertex_streams, );
