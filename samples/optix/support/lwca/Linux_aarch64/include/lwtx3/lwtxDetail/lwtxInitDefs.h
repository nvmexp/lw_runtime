#ifndef LWTX_IMPL_GUARD
#error Never include this file directly -- it is automatically included by lwToolsExt.h (except when LWTX_NO_IMPL is defined).
#endif

LWTX_LINKONCE_DEFINE_FUNCTION void LWTX_API LWTX_VERSIONED_IDENTIFIER(lwtxMarkEx_impl_init)(const lwtxEventAttributes_t* eventAttrib){
    LWTX_VERSIONED_IDENTIFIER(lwtxInitOnce)();
    lwtxMarkEx(eventAttrib);
}

LWTX_LINKONCE_DEFINE_FUNCTION void LWTX_API LWTX_VERSIONED_IDENTIFIER(lwtxMarkA_impl_init)(const char* message){
    LWTX_VERSIONED_IDENTIFIER(lwtxInitOnce)();
    lwtxMarkA(message);
}

LWTX_LINKONCE_DEFINE_FUNCTION void LWTX_API LWTX_VERSIONED_IDENTIFIER(lwtxMarkW_impl_init)(const wchar_t* message){
    LWTX_VERSIONED_IDENTIFIER(lwtxInitOnce)();
    lwtxMarkW(message);
}

LWTX_LINKONCE_DEFINE_FUNCTION lwtxRangeId_t LWTX_API LWTX_VERSIONED_IDENTIFIER(lwtxRangeStartEx_impl_init)(const lwtxEventAttributes_t* eventAttrib){
    LWTX_VERSIONED_IDENTIFIER(lwtxInitOnce)();
    return lwtxRangeStartEx(eventAttrib);
}

LWTX_LINKONCE_DEFINE_FUNCTION lwtxRangeId_t LWTX_API LWTX_VERSIONED_IDENTIFIER(lwtxRangeStartA_impl_init)(const char* message){
    LWTX_VERSIONED_IDENTIFIER(lwtxInitOnce)();
    return lwtxRangeStartA(message);
}

LWTX_LINKONCE_DEFINE_FUNCTION lwtxRangeId_t LWTX_API LWTX_VERSIONED_IDENTIFIER(lwtxRangeStartW_impl_init)(const wchar_t* message){
    LWTX_VERSIONED_IDENTIFIER(lwtxInitOnce)();
    return lwtxRangeStartW(message);
}

LWTX_LINKONCE_DEFINE_FUNCTION void LWTX_API LWTX_VERSIONED_IDENTIFIER(lwtxRangeEnd_impl_init)(lwtxRangeId_t id){
    LWTX_VERSIONED_IDENTIFIER(lwtxInitOnce)();
    lwtxRangeEnd(id);
}

LWTX_LINKONCE_DEFINE_FUNCTION int LWTX_API LWTX_VERSIONED_IDENTIFIER(lwtxRangePushEx_impl_init)(const lwtxEventAttributes_t* eventAttrib){
    LWTX_VERSIONED_IDENTIFIER(lwtxInitOnce)();
    return lwtxRangePushEx(eventAttrib);
}

LWTX_LINKONCE_DEFINE_FUNCTION int LWTX_API LWTX_VERSIONED_IDENTIFIER(lwtxRangePushA_impl_init)(const char* message){
    LWTX_VERSIONED_IDENTIFIER(lwtxInitOnce)();
    return lwtxRangePushA(message);
}

LWTX_LINKONCE_DEFINE_FUNCTION int LWTX_API LWTX_VERSIONED_IDENTIFIER(lwtxRangePushW_impl_init)(const wchar_t* message){
    LWTX_VERSIONED_IDENTIFIER(lwtxInitOnce)();
    return lwtxRangePushW(message);
}

LWTX_LINKONCE_DEFINE_FUNCTION int LWTX_API LWTX_VERSIONED_IDENTIFIER(lwtxRangePop_impl_init)(void){
    LWTX_VERSIONED_IDENTIFIER(lwtxInitOnce)();
    return lwtxRangePop();
}

LWTX_LINKONCE_DEFINE_FUNCTION void LWTX_API LWTX_VERSIONED_IDENTIFIER(lwtxNameCategoryA_impl_init)(uint32_t category, const char* name){
    LWTX_VERSIONED_IDENTIFIER(lwtxInitOnce)();
    lwtxNameCategoryA(category, name);
}

LWTX_LINKONCE_DEFINE_FUNCTION void LWTX_API LWTX_VERSIONED_IDENTIFIER(lwtxNameCategoryW_impl_init)(uint32_t category, const wchar_t* name){
    LWTX_VERSIONED_IDENTIFIER(lwtxInitOnce)();
    lwtxNameCategoryW(category, name);
}

LWTX_LINKONCE_DEFINE_FUNCTION void LWTX_API LWTX_VERSIONED_IDENTIFIER(lwtxNameOsThreadA_impl_init)(uint32_t threadId, const char* name){
    LWTX_VERSIONED_IDENTIFIER(lwtxInitOnce)();
    lwtxNameOsThreadA(threadId, name);
}

LWTX_LINKONCE_DEFINE_FUNCTION void LWTX_API LWTX_VERSIONED_IDENTIFIER(lwtxNameOsThreadW_impl_init)(uint32_t threadId, const wchar_t* name){
    LWTX_VERSIONED_IDENTIFIER(lwtxInitOnce)();
    lwtxNameOsThreadW(threadId, name);
}

LWTX_LINKONCE_DEFINE_FUNCTION void LWTX_API LWTX_VERSIONED_IDENTIFIER(lwtxDomainMarkEx_impl_init)(lwtxDomainHandle_t domain, const lwtxEventAttributes_t* eventAttrib){
    LWTX_VERSIONED_IDENTIFIER(lwtxInitOnce)();
    lwtxDomainMarkEx(domain, eventAttrib);
}

LWTX_LINKONCE_DEFINE_FUNCTION lwtxRangeId_t LWTX_API LWTX_VERSIONED_IDENTIFIER(lwtxDomainRangeStartEx_impl_init)(lwtxDomainHandle_t domain, const lwtxEventAttributes_t* eventAttrib){
    LWTX_VERSIONED_IDENTIFIER(lwtxInitOnce)();
    return lwtxDomainRangeStartEx(domain, eventAttrib);
}

LWTX_LINKONCE_DEFINE_FUNCTION void LWTX_API LWTX_VERSIONED_IDENTIFIER(lwtxDomainRangeEnd_impl_init)(lwtxDomainHandle_t domain, lwtxRangeId_t id){
    LWTX_VERSIONED_IDENTIFIER(lwtxInitOnce)();
    lwtxDomainRangeEnd(domain, id);
}

LWTX_LINKONCE_DEFINE_FUNCTION int LWTX_API LWTX_VERSIONED_IDENTIFIER(lwtxDomainRangePushEx_impl_init)(lwtxDomainHandle_t domain, const lwtxEventAttributes_t* eventAttrib){
    LWTX_VERSIONED_IDENTIFIER(lwtxInitOnce)();
    return lwtxDomainRangePushEx(domain, eventAttrib);
}

LWTX_LINKONCE_DEFINE_FUNCTION int LWTX_API LWTX_VERSIONED_IDENTIFIER(lwtxDomainRangePop_impl_init)(lwtxDomainHandle_t domain){
    LWTX_VERSIONED_IDENTIFIER(lwtxInitOnce)();
    return lwtxDomainRangePop(domain);
}

LWTX_LINKONCE_DEFINE_FUNCTION lwtxResourceHandle_t LWTX_API LWTX_VERSIONED_IDENTIFIER(lwtxDomainResourceCreate_impl_init)(lwtxDomainHandle_t domain, lwtxResourceAttributes_t* attribs){
    LWTX_VERSIONED_IDENTIFIER(lwtxInitOnce)();
    return lwtxDomainResourceCreate(domain, attribs);
}

LWTX_LINKONCE_DEFINE_FUNCTION void LWTX_API LWTX_VERSIONED_IDENTIFIER(lwtxDomainResourceDestroy_impl_init)(lwtxResourceHandle_t resource){
    LWTX_VERSIONED_IDENTIFIER(lwtxInitOnce)();
    lwtxDomainResourceDestroy(resource);
}

LWTX_LINKONCE_DEFINE_FUNCTION void LWTX_API LWTX_VERSIONED_IDENTIFIER(lwtxDomainNameCategoryA_impl_init)(lwtxDomainHandle_t domain, uint32_t category, const char* name){
    LWTX_VERSIONED_IDENTIFIER(lwtxInitOnce)();
    lwtxDomainNameCategoryA(domain, category, name);
}

LWTX_LINKONCE_DEFINE_FUNCTION void LWTX_API LWTX_VERSIONED_IDENTIFIER(lwtxDomainNameCategoryW_impl_init)(lwtxDomainHandle_t domain, uint32_t category, const wchar_t* name){
    LWTX_VERSIONED_IDENTIFIER(lwtxInitOnce)();
    lwtxDomainNameCategoryW(domain, category, name);
}

LWTX_LINKONCE_DEFINE_FUNCTION lwtxStringHandle_t LWTX_API LWTX_VERSIONED_IDENTIFIER(lwtxDomainRegisterStringA_impl_init)(lwtxDomainHandle_t domain, const char* string){
    LWTX_VERSIONED_IDENTIFIER(lwtxInitOnce)();
    return lwtxDomainRegisterStringA(domain, string);
}

LWTX_LINKONCE_DEFINE_FUNCTION lwtxStringHandle_t LWTX_API LWTX_VERSIONED_IDENTIFIER(lwtxDomainRegisterStringW_impl_init)(lwtxDomainHandle_t domain, const wchar_t* string){
    LWTX_VERSIONED_IDENTIFIER(lwtxInitOnce)();
    return lwtxDomainRegisterStringW(domain, string);
}

LWTX_LINKONCE_DEFINE_FUNCTION lwtxDomainHandle_t LWTX_API LWTX_VERSIONED_IDENTIFIER(lwtxDomainCreateA_impl_init)(const char* message){
    LWTX_VERSIONED_IDENTIFIER(lwtxInitOnce)();
    return lwtxDomainCreateA(message);
}

LWTX_LINKONCE_DEFINE_FUNCTION lwtxDomainHandle_t LWTX_API LWTX_VERSIONED_IDENTIFIER(lwtxDomainCreateW_impl_init)(const wchar_t* message){
    LWTX_VERSIONED_IDENTIFIER(lwtxInitOnce)();
    return lwtxDomainCreateW(message);
}

LWTX_LINKONCE_DEFINE_FUNCTION void LWTX_API LWTX_VERSIONED_IDENTIFIER(lwtxDomainDestroy_impl_init)(lwtxDomainHandle_t domain){
    LWTX_VERSIONED_IDENTIFIER(lwtxInitOnce)();
    lwtxDomainDestroy(domain);
}

LWTX_LINKONCE_DEFINE_FUNCTION void LWTX_API LWTX_VERSIONED_IDENTIFIER(lwtxInitialize_impl_init)(const void* reserved){
    LWTX_VERSIONED_IDENTIFIER(lwtxInitOnce)();
    lwtxInitialize(reserved);
}

LWTX_LINKONCE_DEFINE_FUNCTION void LWTX_API LWTX_VERSIONED_IDENTIFIER(lwtxNameLwDeviceA_impl_init)(lwtx_LWdevice device, const char* name){
    lwtxNameLwDeviceA_fakeimpl_fntype local;
    LWTX_VERSIONED_IDENTIFIER(lwtxInitOnce)();
    local = LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameLwDeviceA_impl_fnptr;
    if (local)
        local(device, name);
}

LWTX_LINKONCE_DEFINE_FUNCTION void LWTX_API LWTX_VERSIONED_IDENTIFIER(lwtxNameLwDeviceW_impl_init)(lwtx_LWdevice device, const wchar_t* name){
    lwtxNameLwDeviceW_fakeimpl_fntype local;
    LWTX_VERSIONED_IDENTIFIER(lwtxInitOnce)();
    local = LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameLwDeviceW_impl_fnptr;
    if (local)
        local(device, name);
}

LWTX_LINKONCE_DEFINE_FUNCTION void LWTX_API LWTX_VERSIONED_IDENTIFIER(lwtxNameLwContextA_impl_init)(lwtx_LWcontext context, const char* name){
    lwtxNameLwContextA_fakeimpl_fntype local;
    LWTX_VERSIONED_IDENTIFIER(lwtxInitOnce)();
    local = LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameLwContextA_impl_fnptr;
    if (local)
        local(context, name);
}

LWTX_LINKONCE_DEFINE_FUNCTION void LWTX_API LWTX_VERSIONED_IDENTIFIER(lwtxNameLwContextW_impl_init)(lwtx_LWcontext context, const wchar_t* name){
    lwtxNameLwContextW_fakeimpl_fntype local;
    LWTX_VERSIONED_IDENTIFIER(lwtxInitOnce)();
    local = LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameLwContextW_impl_fnptr;
    if (local)
        local(context, name);
}

LWTX_LINKONCE_DEFINE_FUNCTION void LWTX_API LWTX_VERSIONED_IDENTIFIER(lwtxNameLwStreamA_impl_init)(lwtx_LWstream stream, const char* name){
    lwtxNameLwStreamA_fakeimpl_fntype local;
    LWTX_VERSIONED_IDENTIFIER(lwtxInitOnce)();
    local = LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameLwStreamA_impl_fnptr;
    if (local)
        local(stream, name);
}

LWTX_LINKONCE_DEFINE_FUNCTION void LWTX_API LWTX_VERSIONED_IDENTIFIER(lwtxNameLwStreamW_impl_init)(lwtx_LWstream stream, const wchar_t* name){
    lwtxNameLwStreamW_fakeimpl_fntype local;
    LWTX_VERSIONED_IDENTIFIER(lwtxInitOnce)();
    local = LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameLwStreamW_impl_fnptr;
    if (local)
        local(stream, name);
}

LWTX_LINKONCE_DEFINE_FUNCTION void LWTX_API LWTX_VERSIONED_IDENTIFIER(lwtxNameLwEventA_impl_init)(lwtx_LWevent event, const char* name){
    lwtxNameLwEventA_fakeimpl_fntype local;
    LWTX_VERSIONED_IDENTIFIER(lwtxInitOnce)();
    local = LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameLwEventA_impl_fnptr;
    if (local)
        local(event, name);
}

LWTX_LINKONCE_DEFINE_FUNCTION void LWTX_API LWTX_VERSIONED_IDENTIFIER(lwtxNameLwEventW_impl_init)(lwtx_LWevent event, const wchar_t* name){
    lwtxNameLwEventW_fakeimpl_fntype local;
    LWTX_VERSIONED_IDENTIFIER(lwtxInitOnce)();
    local = LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameLwEventW_impl_fnptr;
    if (local)
        local(event, name);
}

LWTX_LINKONCE_DEFINE_FUNCTION void LWTX_API LWTX_VERSIONED_IDENTIFIER(lwtxNameLwdaDeviceA_impl_init)(int device, const char* name){
    lwtxNameLwdaDeviceA_impl_fntype local;
    LWTX_VERSIONED_IDENTIFIER(lwtxInitOnce)();
    local = LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameLwdaDeviceA_impl_fnptr;
    if (local)
        local(device, name);
}

LWTX_LINKONCE_DEFINE_FUNCTION void LWTX_API LWTX_VERSIONED_IDENTIFIER(lwtxNameLwdaDeviceW_impl_init)(int device, const wchar_t* name){
    lwtxNameLwdaDeviceW_impl_fntype local;
    LWTX_VERSIONED_IDENTIFIER(lwtxInitOnce)();
    local = LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameLwdaDeviceW_impl_fnptr;
    if (local)
        local(device, name);
}

LWTX_LINKONCE_DEFINE_FUNCTION void LWTX_API LWTX_VERSIONED_IDENTIFIER(lwtxNameLwdaStreamA_impl_init)(lwtx_lwdaStream_t stream, const char* name){
    lwtxNameLwdaStreamA_fakeimpl_fntype local;
    LWTX_VERSIONED_IDENTIFIER(lwtxInitOnce)();
    local = LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameLwdaStreamA_impl_fnptr;
    if (local)
        local(stream, name);
}

LWTX_LINKONCE_DEFINE_FUNCTION void LWTX_API LWTX_VERSIONED_IDENTIFIER(lwtxNameLwdaStreamW_impl_init)(lwtx_lwdaStream_t stream, const wchar_t* name){
    lwtxNameLwdaStreamW_fakeimpl_fntype local;
    LWTX_VERSIONED_IDENTIFIER(lwtxInitOnce)();
    local = LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameLwdaStreamW_impl_fnptr;
    if (local)
        local(stream, name);
}

LWTX_LINKONCE_DEFINE_FUNCTION void LWTX_API LWTX_VERSIONED_IDENTIFIER(lwtxNameLwdaEventA_impl_init)(lwtx_lwdaEvent_t event, const char* name){
    lwtxNameLwdaEventA_fakeimpl_fntype local;
    LWTX_VERSIONED_IDENTIFIER(lwtxInitOnce)();
    local = LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameLwdaEventA_impl_fnptr;
    if (local)
        local(event, name);
}

LWTX_LINKONCE_DEFINE_FUNCTION void LWTX_API LWTX_VERSIONED_IDENTIFIER(lwtxNameLwdaEventW_impl_init)(lwtx_lwdaEvent_t event, const wchar_t* name){
    lwtxNameLwdaEventW_fakeimpl_fntype local;
    LWTX_VERSIONED_IDENTIFIER(lwtxInitOnce)();
    local = LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameLwdaEventW_impl_fnptr;
    if (local)
        local(event, name);
}

LWTX_LINKONCE_DEFINE_FUNCTION void LWTX_API LWTX_VERSIONED_IDENTIFIER(lwtxNameClDeviceA_impl_init)(lwtx_cl_device_id device, const char* name){
    lwtxNameClDeviceA_fakeimpl_fntype local;
    LWTX_VERSIONED_IDENTIFIER(lwtxInitOnce)();
    local = LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameClDeviceA_impl_fnptr;
    if (local)
        local(device, name);
}

LWTX_LINKONCE_DEFINE_FUNCTION void LWTX_API LWTX_VERSIONED_IDENTIFIER(lwtxNameClDeviceW_impl_init)(lwtx_cl_device_id device, const wchar_t* name){
    lwtxNameClDeviceW_fakeimpl_fntype local;
    LWTX_VERSIONED_IDENTIFIER(lwtxInitOnce)();
    local = LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameClDeviceW_impl_fnptr;
    if (local)
        local(device, name);
}

LWTX_LINKONCE_DEFINE_FUNCTION void LWTX_API LWTX_VERSIONED_IDENTIFIER(lwtxNameClContextA_impl_init)(lwtx_cl_context context, const char* name){
    lwtxNameClContextA_fakeimpl_fntype local;
    LWTX_VERSIONED_IDENTIFIER(lwtxInitOnce)();
    local = LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameClContextA_impl_fnptr;
    if (local)
        local(context, name);
}

LWTX_LINKONCE_DEFINE_FUNCTION void LWTX_API LWTX_VERSIONED_IDENTIFIER(lwtxNameClContextW_impl_init)(lwtx_cl_context context, const wchar_t* name){
    lwtxNameClContextW_fakeimpl_fntype local;
    LWTX_VERSIONED_IDENTIFIER(lwtxInitOnce)();
    local = LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameClContextW_impl_fnptr;
    if (local)
        local(context, name);
}

LWTX_LINKONCE_DEFINE_FUNCTION void LWTX_API LWTX_VERSIONED_IDENTIFIER(lwtxNameClCommandQueueA_impl_init)(lwtx_cl_command_queue command_queue, const char* name){
    lwtxNameClCommandQueueA_fakeimpl_fntype local;
    LWTX_VERSIONED_IDENTIFIER(lwtxInitOnce)();
    local = LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameClCommandQueueA_impl_fnptr;
    if (local)
        local(command_queue, name);
}

LWTX_LINKONCE_DEFINE_FUNCTION void LWTX_API LWTX_VERSIONED_IDENTIFIER(lwtxNameClCommandQueueW_impl_init)(lwtx_cl_command_queue command_queue, const wchar_t* name){
    lwtxNameClCommandQueueW_fakeimpl_fntype local;
    LWTX_VERSIONED_IDENTIFIER(lwtxInitOnce)();
    local = LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameClCommandQueueW_impl_fnptr;
    if (local)
        local(command_queue, name);
}

LWTX_LINKONCE_DEFINE_FUNCTION void LWTX_API LWTX_VERSIONED_IDENTIFIER(lwtxNameClMemObjectA_impl_init)(lwtx_cl_mem memobj, const char* name){
    lwtxNameClMemObjectA_fakeimpl_fntype local;
    LWTX_VERSIONED_IDENTIFIER(lwtxInitOnce)();
    local = LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameClMemObjectA_impl_fnptr;
    if (local)
        local(memobj, name);
}

LWTX_LINKONCE_DEFINE_FUNCTION void LWTX_API LWTX_VERSIONED_IDENTIFIER(lwtxNameClMemObjectW_impl_init)(lwtx_cl_mem memobj, const wchar_t* name){
    lwtxNameClMemObjectW_fakeimpl_fntype local;
    LWTX_VERSIONED_IDENTIFIER(lwtxInitOnce)();
    local = LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameClMemObjectW_impl_fnptr;
    if (local)
        local(memobj, name);
}

LWTX_LINKONCE_DEFINE_FUNCTION void LWTX_API LWTX_VERSIONED_IDENTIFIER(lwtxNameClSamplerA_impl_init)(lwtx_cl_sampler sampler, const char* name){
    lwtxNameClSamplerA_fakeimpl_fntype local;
    LWTX_VERSIONED_IDENTIFIER(lwtxInitOnce)();
    local = LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameClSamplerA_impl_fnptr;
    if (local)
        local(sampler, name);
}

LWTX_LINKONCE_DEFINE_FUNCTION void LWTX_API LWTX_VERSIONED_IDENTIFIER(lwtxNameClSamplerW_impl_init)(lwtx_cl_sampler sampler, const wchar_t* name){
    lwtxNameClSamplerW_fakeimpl_fntype local;
    LWTX_VERSIONED_IDENTIFIER(lwtxInitOnce)();
    local = LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameClSamplerW_impl_fnptr;
    if (local)
        local(sampler, name);
}

LWTX_LINKONCE_DEFINE_FUNCTION void LWTX_API LWTX_VERSIONED_IDENTIFIER(lwtxNameClProgramA_impl_init)(lwtx_cl_program program, const char* name){
    lwtxNameClProgramA_fakeimpl_fntype local;
    LWTX_VERSIONED_IDENTIFIER(lwtxInitOnce)();
    local = LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameClProgramA_impl_fnptr;
    if (local)
        local(program, name);
}

LWTX_LINKONCE_DEFINE_FUNCTION void LWTX_API LWTX_VERSIONED_IDENTIFIER(lwtxNameClProgramW_impl_init)(lwtx_cl_program program, const wchar_t* name){
    lwtxNameClProgramW_fakeimpl_fntype local;
    LWTX_VERSIONED_IDENTIFIER(lwtxInitOnce)();
    local = LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameClProgramW_impl_fnptr;
    if (local)
        local(program, name);
}

LWTX_LINKONCE_DEFINE_FUNCTION void LWTX_API LWTX_VERSIONED_IDENTIFIER(lwtxNameClEventA_impl_init)(lwtx_cl_event evnt, const char* name){
    lwtxNameClEventA_fakeimpl_fntype local;
    LWTX_VERSIONED_IDENTIFIER(lwtxInitOnce)();
    local = LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameClEventA_impl_fnptr;
    if (local)
        local(evnt, name);
}

LWTX_LINKONCE_DEFINE_FUNCTION void LWTX_API LWTX_VERSIONED_IDENTIFIER(lwtxNameClEventW_impl_init)(lwtx_cl_event evnt, const wchar_t* name){
    lwtxNameClEventW_fakeimpl_fntype local;
    LWTX_VERSIONED_IDENTIFIER(lwtxInitOnce)();
    local = LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameClEventW_impl_fnptr;
    if (local)
        local(evnt, name);
}

LWTX_LINKONCE_DEFINE_FUNCTION lwtxSynlwser_t LWTX_API LWTX_VERSIONED_IDENTIFIER(lwtxDomainSynlwserCreate_impl_init)(lwtxDomainHandle_t domain, const lwtxSynlwserAttributes_t* attribs){
    lwtxDomainSynlwserCreate_impl_fntype local;
    LWTX_VERSIONED_IDENTIFIER(lwtxInitOnce)();
    local = LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainSynlwserCreate_impl_fnptr;
    if (local) {
        return local(domain, attribs);
    }
    return (lwtxSynlwser_t)0;
}

LWTX_LINKONCE_DEFINE_FUNCTION void LWTX_API LWTX_VERSIONED_IDENTIFIER(lwtxDomainSynlwserDestroy_impl_init)(lwtxSynlwser_t handle){
    lwtxDomainSynlwserDestroy_impl_fntype local;
    LWTX_VERSIONED_IDENTIFIER(lwtxInitOnce)();
    local = LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainSynlwserDestroy_impl_fnptr;
    if (local)
        local(handle);
}

LWTX_LINKONCE_DEFINE_FUNCTION void LWTX_API LWTX_VERSIONED_IDENTIFIER(lwtxDomainSynlwserAcquireStart_impl_init)(lwtxSynlwser_t handle){
    lwtxDomainSynlwserAcquireStart_impl_fntype local;
    LWTX_VERSIONED_IDENTIFIER(lwtxInitOnce)();
    local = LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainSynlwserAcquireStart_impl_fnptr;
    if (local)
        local(handle);
}

LWTX_LINKONCE_DEFINE_FUNCTION void LWTX_API LWTX_VERSIONED_IDENTIFIER(lwtxDomainSynlwserAcquireFailed_impl_init)(lwtxSynlwser_t handle){
    lwtxDomainSynlwserAcquireFailed_impl_fntype local;
    LWTX_VERSIONED_IDENTIFIER(lwtxInitOnce)();
    local = LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainSynlwserAcquireFailed_impl_fnptr;
    if (local)
        local(handle);
}

LWTX_LINKONCE_DEFINE_FUNCTION void LWTX_API LWTX_VERSIONED_IDENTIFIER(lwtxDomainSynlwserAcquireSuccess_impl_init)(lwtxSynlwser_t handle){
    lwtxDomainSynlwserAcquireSuccess_impl_fntype local;
    LWTX_VERSIONED_IDENTIFIER(lwtxInitOnce)();
    local = LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainSynlwserAcquireSuccess_impl_fnptr;
    if (local)
        local(handle);
}

LWTX_LINKONCE_DEFINE_FUNCTION void LWTX_API LWTX_VERSIONED_IDENTIFIER(lwtxDomainSynlwserReleasing_impl_init)(lwtxSynlwser_t handle){
    lwtxDomainSynlwserReleasing_impl_fntype local;
    LWTX_VERSIONED_IDENTIFIER(lwtxInitOnce)();
    local = LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainSynlwserReleasing_impl_fnptr;
    if (local)
        local(handle);
}

LWTX_LINKONCE_FWDDECL_FUNCTION void LWTX_VERSIONED_IDENTIFIER(lwtxSetInitFunctionsToNoops)(int forceAllToNoops);
LWTX_LINKONCE_DEFINE_FUNCTION void LWTX_VERSIONED_IDENTIFIER(lwtxSetInitFunctionsToNoops)(int forceAllToNoops)
{
    if (LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxMarkEx_impl_fnptr == LWTX_VERSIONED_IDENTIFIER(lwtxMarkEx_impl_init) || forceAllToNoops)
        LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxMarkEx_impl_fnptr = NULL;
    if (LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxMarkA_impl_fnptr == LWTX_VERSIONED_IDENTIFIER(lwtxMarkA_impl_init) || forceAllToNoops)
        LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxMarkA_impl_fnptr = NULL;
    if (LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxMarkW_impl_fnptr == LWTX_VERSIONED_IDENTIFIER(lwtxMarkW_impl_init) || forceAllToNoops)
        LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxMarkW_impl_fnptr = NULL;
    if (LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxRangeStartEx_impl_fnptr == LWTX_VERSIONED_IDENTIFIER(lwtxRangeStartEx_impl_init) || forceAllToNoops)
        LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxRangeStartEx_impl_fnptr = NULL;
    if (LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxRangeStartA_impl_fnptr == LWTX_VERSIONED_IDENTIFIER(lwtxRangeStartA_impl_init) || forceAllToNoops)
        LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxRangeStartA_impl_fnptr = NULL;
    if (LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxRangeStartW_impl_fnptr == LWTX_VERSIONED_IDENTIFIER(lwtxRangeStartW_impl_init) || forceAllToNoops)
        LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxRangeStartW_impl_fnptr = NULL;
    if (LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxRangeEnd_impl_fnptr == LWTX_VERSIONED_IDENTIFIER(lwtxRangeEnd_impl_init) || forceAllToNoops)
        LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxRangeEnd_impl_fnptr = NULL;
    if (LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxRangePushEx_impl_fnptr == LWTX_VERSIONED_IDENTIFIER(lwtxRangePushEx_impl_init) || forceAllToNoops)
        LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxRangePushEx_impl_fnptr = NULL;
    if (LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxRangePushA_impl_fnptr == LWTX_VERSIONED_IDENTIFIER(lwtxRangePushA_impl_init) || forceAllToNoops)
        LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxRangePushA_impl_fnptr = NULL;
    if (LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxRangePushW_impl_fnptr == LWTX_VERSIONED_IDENTIFIER(lwtxRangePushW_impl_init) || forceAllToNoops)
        LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxRangePushW_impl_fnptr = NULL;
    if (LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxRangePop_impl_fnptr == LWTX_VERSIONED_IDENTIFIER(lwtxRangePop_impl_init) || forceAllToNoops)
        LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxRangePop_impl_fnptr = NULL;
    if (LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameCategoryA_impl_fnptr == LWTX_VERSIONED_IDENTIFIER(lwtxNameCategoryA_impl_init) || forceAllToNoops)
        LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameCategoryA_impl_fnptr = NULL;
    if (LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameCategoryW_impl_fnptr == LWTX_VERSIONED_IDENTIFIER(lwtxNameCategoryW_impl_init) || forceAllToNoops)
        LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameCategoryW_impl_fnptr = NULL;
    if (LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameOsThreadA_impl_fnptr == LWTX_VERSIONED_IDENTIFIER(lwtxNameOsThreadA_impl_init) || forceAllToNoops)
        LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameOsThreadA_impl_fnptr = NULL;
    if (LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameOsThreadW_impl_fnptr == LWTX_VERSIONED_IDENTIFIER(lwtxNameOsThreadW_impl_init) || forceAllToNoops)
        LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameOsThreadW_impl_fnptr = NULL;

    if (LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameLwDeviceA_impl_fnptr == LWTX_VERSIONED_IDENTIFIER(lwtxNameLwDeviceA_impl_init) || forceAllToNoops)
        LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameLwDeviceA_impl_fnptr = NULL;
    if (LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameLwDeviceW_impl_fnptr == LWTX_VERSIONED_IDENTIFIER(lwtxNameLwDeviceW_impl_init) || forceAllToNoops)
        LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameLwDeviceW_impl_fnptr = NULL;
    if (LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameLwContextA_impl_fnptr == LWTX_VERSIONED_IDENTIFIER(lwtxNameLwContextA_impl_init) || forceAllToNoops)
        LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameLwContextA_impl_fnptr = NULL;
    if (LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameLwContextW_impl_fnptr == LWTX_VERSIONED_IDENTIFIER(lwtxNameLwContextW_impl_init) || forceAllToNoops)
        LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameLwContextW_impl_fnptr = NULL;
    if (LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameLwStreamA_impl_fnptr == LWTX_VERSIONED_IDENTIFIER(lwtxNameLwStreamA_impl_init) || forceAllToNoops)
        LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameLwStreamA_impl_fnptr = NULL;
    if (LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameLwStreamW_impl_fnptr == LWTX_VERSIONED_IDENTIFIER(lwtxNameLwStreamW_impl_init) || forceAllToNoops)
        LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameLwStreamW_impl_fnptr = NULL;
    if (LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameLwEventA_impl_fnptr == LWTX_VERSIONED_IDENTIFIER(lwtxNameLwEventA_impl_init) || forceAllToNoops)
        LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameLwEventA_impl_fnptr = NULL;
    if (LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameLwEventW_impl_fnptr == LWTX_VERSIONED_IDENTIFIER(lwtxNameLwEventW_impl_init) || forceAllToNoops)
        LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameLwEventW_impl_fnptr = NULL;

    if (LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameClDeviceA_impl_fnptr == LWTX_VERSIONED_IDENTIFIER(lwtxNameClDeviceA_impl_init) || forceAllToNoops)
        LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameClDeviceA_impl_fnptr = NULL;
    if (LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameClDeviceW_impl_fnptr == LWTX_VERSIONED_IDENTIFIER(lwtxNameClDeviceW_impl_init) || forceAllToNoops)
        LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameClDeviceW_impl_fnptr = NULL;
    if (LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameClContextA_impl_fnptr == LWTX_VERSIONED_IDENTIFIER(lwtxNameClContextA_impl_init) || forceAllToNoops)
        LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameClContextA_impl_fnptr = NULL;
    if (LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameClContextW_impl_fnptr == LWTX_VERSIONED_IDENTIFIER(lwtxNameClContextW_impl_init) || forceAllToNoops)
        LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameClContextW_impl_fnptr = NULL;
    if (LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameClCommandQueueA_impl_fnptr == LWTX_VERSIONED_IDENTIFIER(lwtxNameClCommandQueueA_impl_init) || forceAllToNoops)
        LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameClCommandQueueA_impl_fnptr = NULL;
    if (LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameClCommandQueueW_impl_fnptr == LWTX_VERSIONED_IDENTIFIER(lwtxNameClCommandQueueW_impl_init) || forceAllToNoops)
        LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameClCommandQueueW_impl_fnptr = NULL;
    if (LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameClMemObjectA_impl_fnptr == LWTX_VERSIONED_IDENTIFIER(lwtxNameClMemObjectA_impl_init) || forceAllToNoops)
        LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameClMemObjectA_impl_fnptr = NULL;
    if (LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameClMemObjectW_impl_fnptr == LWTX_VERSIONED_IDENTIFIER(lwtxNameClMemObjectW_impl_init) || forceAllToNoops)
        LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameClMemObjectW_impl_fnptr = NULL;
    if (LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameClSamplerA_impl_fnptr == LWTX_VERSIONED_IDENTIFIER(lwtxNameClSamplerA_impl_init) || forceAllToNoops)
        LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameClSamplerA_impl_fnptr = NULL;
    if (LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameClSamplerW_impl_fnptr == LWTX_VERSIONED_IDENTIFIER(lwtxNameClSamplerW_impl_init) || forceAllToNoops)
        LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameClSamplerW_impl_fnptr = NULL;
    if (LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameClProgramA_impl_fnptr == LWTX_VERSIONED_IDENTIFIER(lwtxNameClProgramA_impl_init) || forceAllToNoops)
        LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameClProgramA_impl_fnptr = NULL;
    if (LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameClProgramW_impl_fnptr == LWTX_VERSIONED_IDENTIFIER(lwtxNameClProgramW_impl_init) || forceAllToNoops)
        LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameClProgramW_impl_fnptr = NULL;
    if (LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameClEventA_impl_fnptr == LWTX_VERSIONED_IDENTIFIER(lwtxNameClEventA_impl_init) || forceAllToNoops)
        LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameClEventA_impl_fnptr = NULL;
    if (LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameClEventW_impl_fnptr == LWTX_VERSIONED_IDENTIFIER(lwtxNameClEventW_impl_init) || forceAllToNoops)
        LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameClEventW_impl_fnptr = NULL;

    if (LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameLwdaDeviceA_impl_fnptr == LWTX_VERSIONED_IDENTIFIER(lwtxNameLwdaDeviceA_impl_init) || forceAllToNoops)
        LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameLwdaDeviceA_impl_fnptr = NULL;
    if (LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameLwdaDeviceW_impl_fnptr == LWTX_VERSIONED_IDENTIFIER(lwtxNameLwdaDeviceW_impl_init) || forceAllToNoops)
        LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameLwdaDeviceW_impl_fnptr = NULL;
    if (LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameLwdaStreamA_impl_fnptr == LWTX_VERSIONED_IDENTIFIER(lwtxNameLwdaStreamA_impl_init) || forceAllToNoops)
        LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameLwdaStreamA_impl_fnptr = NULL;
    if (LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameLwdaStreamW_impl_fnptr == LWTX_VERSIONED_IDENTIFIER(lwtxNameLwdaStreamW_impl_init) || forceAllToNoops)
        LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameLwdaStreamW_impl_fnptr = NULL;
    if (LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameLwdaEventA_impl_fnptr == LWTX_VERSIONED_IDENTIFIER(lwtxNameLwdaEventA_impl_init) || forceAllToNoops)
        LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameLwdaEventA_impl_fnptr = NULL;
    if (LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameLwdaEventW_impl_fnptr == LWTX_VERSIONED_IDENTIFIER(lwtxNameLwdaEventW_impl_init) || forceAllToNoops)
        LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameLwdaEventW_impl_fnptr = NULL;

    if (LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainMarkEx_impl_fnptr == LWTX_VERSIONED_IDENTIFIER(lwtxDomainMarkEx_impl_init) || forceAllToNoops)
        LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainMarkEx_impl_fnptr = NULL;
    if (LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainRangeStartEx_impl_fnptr == LWTX_VERSIONED_IDENTIFIER(lwtxDomainRangeStartEx_impl_init) || forceAllToNoops)
        LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainRangeStartEx_impl_fnptr = NULL;
    if (LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainRangeEnd_impl_fnptr == LWTX_VERSIONED_IDENTIFIER(lwtxDomainRangeEnd_impl_init) || forceAllToNoops)
        LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainRangeEnd_impl_fnptr = NULL;
    if (LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainRangePushEx_impl_fnptr == LWTX_VERSIONED_IDENTIFIER(lwtxDomainRangePushEx_impl_init) || forceAllToNoops)
        LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainRangePushEx_impl_fnptr = NULL;
    if (LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainRangePop_impl_fnptr == LWTX_VERSIONED_IDENTIFIER(lwtxDomainRangePop_impl_init) || forceAllToNoops)
        LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainRangePop_impl_fnptr = NULL;
    if (LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainResourceCreate_impl_fnptr == LWTX_VERSIONED_IDENTIFIER(lwtxDomainResourceCreate_impl_init) || forceAllToNoops)
        LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainResourceCreate_impl_fnptr = NULL;
    if (LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainResourceDestroy_impl_fnptr == LWTX_VERSIONED_IDENTIFIER(lwtxDomainResourceDestroy_impl_init) || forceAllToNoops)
        LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainResourceDestroy_impl_fnptr = NULL;
    if (LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainNameCategoryA_impl_fnptr == LWTX_VERSIONED_IDENTIFIER(lwtxDomainNameCategoryA_impl_init) || forceAllToNoops)
        LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainNameCategoryA_impl_fnptr = NULL;
    if (LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainNameCategoryW_impl_fnptr == LWTX_VERSIONED_IDENTIFIER(lwtxDomainNameCategoryW_impl_init) || forceAllToNoops)
        LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainNameCategoryW_impl_fnptr = NULL;
    if (LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainRegisterStringA_impl_fnptr == LWTX_VERSIONED_IDENTIFIER(lwtxDomainRegisterStringA_impl_init) || forceAllToNoops)
        LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainRegisterStringA_impl_fnptr = NULL;
    if (LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainRegisterStringW_impl_fnptr == LWTX_VERSIONED_IDENTIFIER(lwtxDomainRegisterStringW_impl_init) || forceAllToNoops)
        LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainRegisterStringW_impl_fnptr = NULL;
    if (LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainCreateA_impl_fnptr == LWTX_VERSIONED_IDENTIFIER(lwtxDomainCreateA_impl_init) || forceAllToNoops)
        LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainCreateA_impl_fnptr = NULL;
    if (LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainCreateW_impl_fnptr == LWTX_VERSIONED_IDENTIFIER(lwtxDomainCreateW_impl_init) || forceAllToNoops)
        LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainCreateW_impl_fnptr = NULL;
    if (LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainDestroy_impl_fnptr == LWTX_VERSIONED_IDENTIFIER(lwtxDomainDestroy_impl_init) || forceAllToNoops)
        LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainDestroy_impl_fnptr = NULL;
    if (LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxInitialize_impl_fnptr == LWTX_VERSIONED_IDENTIFIER(lwtxInitialize_impl_init) || forceAllToNoops)
        LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxInitialize_impl_fnptr = NULL;

    if (LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainSynlwserCreate_impl_fnptr == LWTX_VERSIONED_IDENTIFIER(lwtxDomainSynlwserCreate_impl_init) || forceAllToNoops)
        LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainSynlwserCreate_impl_fnptr = NULL;
    if (LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainSynlwserDestroy_impl_fnptr == LWTX_VERSIONED_IDENTIFIER(lwtxDomainSynlwserDestroy_impl_init) || forceAllToNoops)
        LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainSynlwserDestroy_impl_fnptr = NULL;
    if (LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainSynlwserAcquireStart_impl_fnptr == LWTX_VERSIONED_IDENTIFIER(lwtxDomainSynlwserAcquireStart_impl_init) || forceAllToNoops)
        LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainSynlwserAcquireStart_impl_fnptr = NULL;
    if (LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainSynlwserAcquireFailed_impl_fnptr == LWTX_VERSIONED_IDENTIFIER(lwtxDomainSynlwserAcquireFailed_impl_init) || forceAllToNoops)
        LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainSynlwserAcquireFailed_impl_fnptr = NULL;
    if (LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainSynlwserAcquireSuccess_impl_fnptr == LWTX_VERSIONED_IDENTIFIER(lwtxDomainSynlwserAcquireSuccess_impl_init) || forceAllToNoops)
        LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainSynlwserAcquireSuccess_impl_fnptr = NULL;
    if (LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainSynlwserReleasing_impl_fnptr == LWTX_VERSIONED_IDENTIFIER(lwtxDomainSynlwserReleasing_impl_init) || forceAllToNoops)
        LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainSynlwserReleasing_impl_fnptr = NULL;
}
