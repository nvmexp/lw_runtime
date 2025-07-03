#pragma once

#include "lwtensor.h"
#include "lwtensor/internal/operatorsPLC3.h"
#include "lwtensor/internal/heuristicDnn.h"

DeviceProp createLwstomDeviceProp(const DeviceProp* originalDeviceProp, int numSMs)
{
    DeviceProp lwstomDeviceProp;

    lwstomDeviceProp.multiProcessorCount = numSMs;

    lwstomDeviceProp.totalGlobalMem = originalDeviceProp->totalGlobalMem;
    lwstomDeviceProp.sharedMemPerBlock = originalDeviceProp->sharedMemPerBlock;
    lwstomDeviceProp.regsPerBlock = originalDeviceProp->regsPerBlock;
    lwstomDeviceProp.maxThreadsPerBlock = originalDeviceProp->maxThreadsPerBlock;
    for (int i = 0; i < 3; i++)
    {
        lwstomDeviceProp.maxThreadsDim[i] = originalDeviceProp->maxThreadsDim[i];
        lwstomDeviceProp.maxGridSize[i] = originalDeviceProp->maxGridSize[i];
    }
    lwstomDeviceProp.clockRate = originalDeviceProp->clockRate;
    lwstomDeviceProp.totalConstMem = originalDeviceProp->totalConstMem;
    lwstomDeviceProp.major = originalDeviceProp->major;
    lwstomDeviceProp.minor = originalDeviceProp->minor;
    lwstomDeviceProp.memoryClockRate = originalDeviceProp->memoryClockRate;
    lwstomDeviceProp.memoryBusWidth = originalDeviceProp->memoryBusWidth;
    lwstomDeviceProp.l2CacheSize = originalDeviceProp->l2CacheSize;
    lwstomDeviceProp.maxThreadsPerMultiProcessor = originalDeviceProp->maxThreadsPerMultiProcessor;
    lwstomDeviceProp.sharedMemPerMultiprocessor = originalDeviceProp->sharedMemPerMultiprocessor;
    lwstomDeviceProp.regsPerMultiprocessor = originalDeviceProp->regsPerMultiprocessor;
    lwstomDeviceProp.singleToDoublePrecisionPerfRatio = originalDeviceProp->singleToDoublePrecisionPerfRatio;

    return lwstomDeviceProp;
}

template <typename Weights>
void runNumberOfFeaturesTest(int ground_truth)
{
    EXPECT_EQ(HeuristicDnn<Weights>().numberOfFeatures(), ground_truth);
}

template <typename Weights>
void runComputeFeaturesTest(int numSMs, std::vector<float> ground_truth,
                            lwdaDataType_t typeA, lwdaDataType_t typeB,
                            lwdaDataType_t typeC, lwtensorComputeType_t typeCompute)
{
    lwtensorHandle_t handle;
    ASSERT_EQ(lwtensorInit(&handle), LWTENSOR_STATUS_SUCCESS);
    auto ctx = reinterpret_cast<const Context*>(&handle);
    const DeviceProp* deviceProp = ctx->getDeviceProp();
    ContractionDescriptorInternal params;


    /* key: -modeCi,a,b -alignmentReqA128 -alignmentA128 -modeAa,b,p,q -alignmentReqB128 -alignmentB128 -       modeBq,i,p -extenta=276,b=8,i=160,p=376,q=4, -alignmentReqC128 -alignmentC128 -opAB3 -opABC3 -           opReduce3 -opA1 -opB1 -opC1 -Pad -Pbd -Pcd -Pcompd -alpha1.200000 -beta0.000000 -gamma0.000000 -         Rcontraction -algo+0 | lwTENSOR:1401.609 GFLOPS | lwTENSOR-max:1408.505 GFLOPS | lwTENSOR-avg:1400.      025 GFLOPS| lwBLAS:4054.740 GFLOPS | bandwidth:45.145 GB/s | memcpy:694.667 GB/s | info:A(1,1,16)0,      1,2,(160:4)(4:1)(376:640)B(1,1,16)3,1,2,(2208:1)(4:830208)(376:2208)C(1,1,16)0,3,(160:1)(2208:           160)16,0kernel:tb:64,64,8;w:32,32,8;is:1,1,1;a:1,1,1;s:0,0;t:1,1;op:1,1;nk:8;cc:70,70,72;ar:0;fm:0;      oc:1;tp:d,d,d,d,d;reg:238;lmem:0;ac:2;wa:1028;ls:132;lg:684;la:54;d(0,0,0)t(0,0,0)sw(1) */


    // info:A(1,1,16)0,1,2,(160:4)(4:1)(376:640)B(1,1,16)3,1,2,(2208:1)(4:830208)(376:2208)C(1,1,16)0,3,(160:1)(2208:160)16,0
    // kernel:tb:64,64,8;w:32,16,8;is:1,1,1;a:1,1,1;s:0,0;t:1,1;op:1,1;nk:8;cc:70,70,72;ar:0;fm:0;oc:1;tp:d,d,d,d,d;reg:238;lmem:0;ac:2;wa:1028;ls:132;lg:684;la:54;d(0,0,0)t(0,0,0)sw(1)
    //
    // 0 1 2 3 4 5
    // a b p q i

    ModeList modeA({0, 1, 2, 3});
    ModeList modeB({3, 4, 2});
    ModeList modeC({4, 0, 1});
    ModeList modeM({0, 1});
    ModeList modeN({4});
    ModeList modeK({2, 3});
    ModeList modeL({});

    ExtentMap extent;
    extent.insert(std::make_pair(0, 276));
    extent.insert(std::make_pair(1, 8));
    extent.insert(std::make_pair(2, 376));
    extent.insert(std::make_pair(3, 4));
    extent.insert(std::make_pair(4, 160));

    StrideMap strideA;
    strideA.insert(std::make_pair(0, 1));
    strideA.insert(std::make_pair(1, 276));
    strideA.insert(std::make_pair(2, 276 * 8));
    strideA.insert(std::make_pair(3, 276 * 8 * 376));

    StrideMap strideB;
    strideB.insert(std::make_pair(3, 1));
    strideB.insert(std::make_pair(4, 4));
    strideB.insert(std::make_pair(2, 4 * 160));

    StrideMap strideC;
    strideC.insert(std::make_pair(4, 1));
    strideC.insert(std::make_pair(0, 160));
    strideC.insert(std::make_pair(1, 160 * 276));

    bool swapAB = false;

    params.initContractionDescriptorInternal(
            typeA, typeB, typeC, typeCompute,
            modeB, modeA, modeN, modeM, modeK, modeL,
            extent,
            LWTENSOR_OP_IDENTITY, strideB, 16,
            LWTENSOR_OP_IDENTITY, strideA, 16,
            LWTENSOR_OP_IDENTITY, strideC, 16,
            false, false, false, false, swapAB
            );

    CandidateInfoLwtlass candidateInfo;
    candidateInfo.threadblockM = 64;
    candidateInfo.threadblockN = 64;
    candidateInfo.threadblockK = 8;
    candidateInfo.warpM = 32;
    candidateInfo.warpN = 32;
    candidateInfo.warpK = 8;
    candidateInfo.elementsPerAccessA = 1;
    candidateInfo.elementsPerAccessB = 1;
    candidateInfo.elementsPerAccessC = 1;
    candidateInfo.numThreads = 128; //
    candidateInfo.maxCTAsPerSM = 2;
    candidateInfo.localMemoryUsage = 0;
    candidateInfo.waitSchedule = 1028;
    candidateInfo.avgLDS = 132;
    candidateInfo.avgLDG = 684;
    candidateInfo.avgAntidep = 54;

    HeuristicDnn<Weights> heuristic;

    ASSERT_EQ(ground_truth.size(), heuristic.numberOfFeatures());

    std::vector<float> features(heuristic.numberOfFeatures(), 0.f);

    DeviceProp lwstonDeviceProp = createLwstomDeviceProp(deviceProp, numSMs);
    heuristic.computeFeatures(params, candidateInfo, &lwstonDeviceProp, features.data());
    for (int i = 0; i < heuristic.numberOfFeatures(); ++i)
    {
        EXPECT_NEAR(ground_truth[i], features[i], 1e-3) << i;
    }
}

template <typename Weights>
void runEvaluateTest(std::vector<float> ground_truth)
{
    HeuristicDnn<Weights> heuristic;

    const int BATCH_SIZE = 10;

    std::vector<float> x = {
        0.5488135, 0.71518937, 0.60276338, 0.54488318, 0.4236548, 0.64589411, 0.43758721, 0.891773, 0.96366276, 0.38344152, 0.79172504, 0.52889492, 0.56804456, 0.92559664, 0.07103606, 0.0871293, 0.0202184, 0.83261985, 0.77815675, 0.87001215, 0.97861834,
    0.79915856, 0.46147936, 0.78052918, 0.11827443, 0.63992102, 0.14335329, 0.94466892, 0.52184832, 0.41466194, 0.26455561, 0.77423369, 0.45615033, 0.56843395, 0.0187898,  0.6176355,  0.61209572, 0.616934,   0.94374808, 0.6818203,  0.3595079,  0.43703195,
    0.6976312,  0.06022547, 0.66676672, 0.67063787, 0.21038256, 0.1289263, 0.31542835, 0.36371077, 0.57019677, 0.43860151, 0.98837384, 0.10204481, 0.20887676, 0.16130952, 0.65310833, 0.2532916,  0.46631077, 0.24442559, 0.15896958, 0.11037514, 0.65632959,
    0.13818295, 0.19658236, 0.36872517, 0.82099323, 0.09710128, 0.83794491, 0.09609841, 0.97645947, 0.4686512,  0.97676109, 0.60484552, 0.73926358, 0.03918779, 0.28280696, 0.12019656, 0.2961402,  0.11872772, 0.31798318, 0.41426299, 0.0641475,  0.69247212,
    0.56660145, 0.26538949, 0.52324805, 0.09394051, 0.5759465,  0.9292962, 0.31856895, 0.66741038, 0.13179786, 0.7163272,  0.28940609, 0.18319136, 0.58651293, 0.02010755, 0.82894003, 0.00469548, 0.67781654, 0.27000797, 0.73519402, 0.96218855, 0.24875314,
    0.57615733, 0.59204193, 0.57225191, 0.22308163, 0.95274901, 0.44712538, 0.84640867, 0.69947928, 0.29743695, 0.81379782, 0.39650574, 0.8811032, 0.58127287, 0.88173536, 0.69253159, 0.72525428, 0.50132438, 0.95608363, 0.6439902,  0.42385505, 0.60639321,
    0.0191932,  0.30157482, 0.66017354, 0.29007761, 0.61801543, 0.4287687, 0.13547406, 0.29828233, 0.56996491, 0.59087276, 0.57432525, 0.65320082, 0.65210327, 0.43141844, 0.8965466,  0.36756187, 0.43586493, 0.89192336, 0.80619399, 0.70388858, 0.10022689,
    0.91948261, 0.7142413,  0.99884701, 0.1494483,  0.86812606, 0.16249293, 0.61555956, 0.12381998, 0.84800823, 0.80731896, 0.56910074, 0.4071833, 0.069167,   0.69742877, 0.45354268, 0.7220556,  0.86638233, 0.97552151, 0.85580334, 0.01171408, 0.35997806,
    0.72999056, 0.17162968, 0.52103661, 0.05433799, 0.19999652, 0.01852179, 0.7936977,  0.22392469, 0.34535168, 0.92808129, 0.7044144,  0.03183893, 0.16469416, 0.6214784,  0.57722859, 0.23789282, 0.934214,   0.61396596, 0.5356328,  0.58990998, 0.73012203,
    0.311945,   0.39822106, 0.20984375, 0.18619301, 0.94437239, 0.7395508, 0.49045881, 0.22741463, 0.25435648, 0.05802916, 0.43441663, 0.31179588, 0.69634349, 0.37775184, 0.17960368, 0.02467873, 0.06724963, 0.67939277, 0.45369684, 0.53657921, 0.89667129,
    };

    ASSERT_EQ(x.size(), BATCH_SIZE * heuristic.numberOfFeatures());

    ASSERT_EQ(ground_truth.size(), BATCH_SIZE);
    float p[BATCH_SIZE];

    heuristic.evaluate(BATCH_SIZE, x.data(), p);

    for (int i = 0; i < BATCH_SIZE; ++i)
    {
        EXPECT_NEAR(ground_truth[i], lwSigmoid(p[i]), 1e-3);
    }

}
