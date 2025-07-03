LWTX_DECLSPEC void LWTX_API lwtxMarkEx(const lwtxEventAttributes_t* eventAttrib)
{
#ifndef LWTX_DISABLE
    lwtxMarkEx_impl_fntype local = LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxMarkEx_impl_fnptr;
    if(local!=0)
        (*local)(eventAttrib);
#endif /*LWTX_DISABLE*/
}

LWTX_DECLSPEC void LWTX_API lwtxMarkA(const char* message)
{
#ifndef LWTX_DISABLE
    lwtxMarkA_impl_fntype local = LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxMarkA_impl_fnptr;
    if(local!=0)
        (*local)(message);
#endif /*LWTX_DISABLE*/
}

LWTX_DECLSPEC void LWTX_API lwtxMarkW(const wchar_t* message)
{
#ifndef LWTX_DISABLE
    lwtxMarkW_impl_fntype local = LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxMarkW_impl_fnptr;
    if(local!=0)
        (*local)(message);
#endif /*LWTX_DISABLE*/
}

LWTX_DECLSPEC lwtxRangeId_t LWTX_API lwtxRangeStartEx(const lwtxEventAttributes_t* eventAttrib)
{
#ifndef LWTX_DISABLE
    lwtxRangeStartEx_impl_fntype local = LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxRangeStartEx_impl_fnptr;
    if(local!=0)
        return (*local)(eventAttrib);
    else
#endif  /*LWTX_DISABLE*/
        return (lwtxRangeId_t)0;
}

LWTX_DECLSPEC lwtxRangeId_t LWTX_API lwtxRangeStartA(const char* message)
{
#ifndef LWTX_DISABLE
    lwtxRangeStartA_impl_fntype local = LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxRangeStartA_impl_fnptr;
    if(local!=0)
        return (*local)(message);
    else
#endif  /*LWTX_DISABLE*/
        return (lwtxRangeId_t)0;
}

LWTX_DECLSPEC lwtxRangeId_t LWTX_API lwtxRangeStartW(const wchar_t* message)
{
#ifndef LWTX_DISABLE
    lwtxRangeStartW_impl_fntype local = LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxRangeStartW_impl_fnptr;
    if(local!=0)
        return (*local)(message);
    else
#endif  /*LWTX_DISABLE*/
        return (lwtxRangeId_t)0;
}

LWTX_DECLSPEC void LWTX_API lwtxRangeEnd(lwtxRangeId_t id)
{
#ifndef LWTX_DISABLE
    lwtxRangeEnd_impl_fntype local = LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxRangeEnd_impl_fnptr;
    if(local!=0)
        (*local)(id);
#endif /*LWTX_DISABLE*/
}

LWTX_DECLSPEC int LWTX_API lwtxRangePushEx(const lwtxEventAttributes_t* eventAttrib)
{
#ifndef LWTX_DISABLE
    lwtxRangePushEx_impl_fntype local = LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxRangePushEx_impl_fnptr;
    if(local!=0)
        return (*local)(eventAttrib);
    else
#endif  /*LWTX_DISABLE*/
        return (int)LWTX_NO_PUSH_POP_TRACKING;
}

LWTX_DECLSPEC int LWTX_API lwtxRangePushA(const char* message)
{
#ifndef LWTX_DISABLE
    lwtxRangePushA_impl_fntype local = LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxRangePushA_impl_fnptr;
    if(local!=0)
        return (*local)(message);
    else
#endif  /*LWTX_DISABLE*/
        return (int)LWTX_NO_PUSH_POP_TRACKING;
}

LWTX_DECLSPEC int LWTX_API lwtxRangePushW(const wchar_t* message)
{
#ifndef LWTX_DISABLE
    lwtxRangePushW_impl_fntype local = LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxRangePushW_impl_fnptr;
    if(local!=0)
        return (*local)(message);
    else
#endif  /*LWTX_DISABLE*/
        return (int)LWTX_NO_PUSH_POP_TRACKING;
}

LWTX_DECLSPEC int LWTX_API lwtxRangePop(void)
{
#ifndef LWTX_DISABLE
    lwtxRangePop_impl_fntype local = LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxRangePop_impl_fnptr;
    if(local!=0)
        return (*local)();
    else
#endif  /*LWTX_DISABLE*/
        return (int)LWTX_NO_PUSH_POP_TRACKING;
}

LWTX_DECLSPEC void LWTX_API lwtxNameCategoryA(uint32_t category, const char* name)
{
#ifndef LWTX_DISABLE
    lwtxNameCategoryA_impl_fntype local = LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameCategoryA_impl_fnptr;
    if(local!=0)
        (*local)(category, name);
#endif /*LWTX_DISABLE*/
}

LWTX_DECLSPEC void LWTX_API lwtxNameCategoryW(uint32_t category, const wchar_t* name)
{
#ifndef LWTX_DISABLE
    lwtxNameCategoryW_impl_fntype local = LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameCategoryW_impl_fnptr;
    if(local!=0)
        (*local)(category, name);
#endif /*LWTX_DISABLE*/
}

LWTX_DECLSPEC void LWTX_API lwtxNameOsThreadA(uint32_t threadId, const char* name)
{
#ifndef LWTX_DISABLE
    lwtxNameOsThreadA_impl_fntype local = LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameOsThreadA_impl_fnptr;
    if(local!=0)
        (*local)(threadId, name);
#endif /*LWTX_DISABLE*/
}

LWTX_DECLSPEC void LWTX_API lwtxNameOsThreadW(uint32_t threadId, const wchar_t* name)
{
#ifndef LWTX_DISABLE
    lwtxNameOsThreadW_impl_fntype local = LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxNameOsThreadW_impl_fnptr;
    if(local!=0)
        (*local)(threadId, name);
#endif /*LWTX_DISABLE*/
}

LWTX_DECLSPEC void LWTX_API lwtxDomainMarkEx(lwtxDomainHandle_t domain, const lwtxEventAttributes_t* eventAttrib)
{
#ifndef LWTX_DISABLE
    lwtxDomainMarkEx_impl_fntype local = LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainMarkEx_impl_fnptr;
    if(local!=0)
        (*local)(domain, eventAttrib);
#endif /*LWTX_DISABLE*/
}

LWTX_DECLSPEC lwtxRangeId_t LWTX_API lwtxDomainRangeStartEx(lwtxDomainHandle_t domain, const lwtxEventAttributes_t* eventAttrib)
{
#ifndef LWTX_DISABLE
    lwtxDomainRangeStartEx_impl_fntype local = LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainRangeStartEx_impl_fnptr;
    if(local!=0)
        return (*local)(domain, eventAttrib);
    else
#endif  /*LWTX_DISABLE*/
        return (lwtxRangeId_t)0;
}

LWTX_DECLSPEC void LWTX_API lwtxDomainRangeEnd(lwtxDomainHandle_t domain, lwtxRangeId_t id)
{
#ifndef LWTX_DISABLE
    lwtxDomainRangeEnd_impl_fntype local = LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainRangeEnd_impl_fnptr;
    if(local!=0)
        (*local)(domain, id);
#endif /*LWTX_DISABLE*/
}

LWTX_DECLSPEC int LWTX_API lwtxDomainRangePushEx(lwtxDomainHandle_t domain, const lwtxEventAttributes_t* eventAttrib)
{
#ifndef LWTX_DISABLE
    lwtxDomainRangePushEx_impl_fntype local = LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainRangePushEx_impl_fnptr;
    if(local!=0)
        return (*local)(domain, eventAttrib);
    else
#endif  /*LWTX_DISABLE*/
        return (int)LWTX_NO_PUSH_POP_TRACKING;
}

LWTX_DECLSPEC int LWTX_API lwtxDomainRangePop(lwtxDomainHandle_t domain)
{
#ifndef LWTX_DISABLE
    lwtxDomainRangePop_impl_fntype local = LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainRangePop_impl_fnptr;
    if(local!=0)
        return (*local)(domain);
    else
#endif  /*LWTX_DISABLE*/
        return (int)LWTX_NO_PUSH_POP_TRACKING;
}

LWTX_DECLSPEC lwtxResourceHandle_t LWTX_API lwtxDomainResourceCreate(lwtxDomainHandle_t domain, lwtxResourceAttributes_t* attribs)
{
#ifndef LWTX_DISABLE
    lwtxDomainResourceCreate_impl_fntype local = LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainResourceCreate_impl_fnptr;
    if(local!=0)
        return (*local)(domain, attribs);
    else
#endif  /*LWTX_DISABLE*/
        return (lwtxResourceHandle_t)0;
}

LWTX_DECLSPEC void LWTX_API lwtxDomainResourceDestroy(lwtxResourceHandle_t resource)
{
#ifndef LWTX_DISABLE
    lwtxDomainResourceDestroy_impl_fntype local = LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainResourceDestroy_impl_fnptr;
    if(local!=0)
        (*local)(resource);
#endif /*LWTX_DISABLE*/
}

LWTX_DECLSPEC void LWTX_API lwtxDomainNameCategoryA(lwtxDomainHandle_t domain, uint32_t category, const char* name)
{
#ifndef LWTX_DISABLE
    lwtxDomainNameCategoryA_impl_fntype local = LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainNameCategoryA_impl_fnptr;
    if(local!=0)
        (*local)(domain, category, name);
#endif /*LWTX_DISABLE*/
}

LWTX_DECLSPEC void LWTX_API lwtxDomainNameCategoryW(lwtxDomainHandle_t domain, uint32_t category, const wchar_t* name)
{
#ifndef LWTX_DISABLE
    lwtxDomainNameCategoryW_impl_fntype local = LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainNameCategoryW_impl_fnptr;
    if(local!=0)
        (*local)(domain, category, name);
#endif /*LWTX_DISABLE*/
}

LWTX_DECLSPEC lwtxStringHandle_t LWTX_API lwtxDomainRegisterStringA(lwtxDomainHandle_t domain, const char* string)
{
#ifndef LWTX_DISABLE
    lwtxDomainRegisterStringA_impl_fntype local = LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainRegisterStringA_impl_fnptr;
    if(local!=0)
        return (*local)(domain, string);
    else
#endif  /*LWTX_DISABLE*/
        return (lwtxStringHandle_t)0;
}

LWTX_DECLSPEC lwtxStringHandle_t LWTX_API lwtxDomainRegisterStringW(lwtxDomainHandle_t domain, const wchar_t* string)
{
#ifndef LWTX_DISABLE
    lwtxDomainRegisterStringW_impl_fntype local = LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainRegisterStringW_impl_fnptr;
    if(local!=0)
        return (*local)(domain, string);
    else
#endif  /*LWTX_DISABLE*/
        return (lwtxStringHandle_t)0;
}

LWTX_DECLSPEC lwtxDomainHandle_t LWTX_API lwtxDomainCreateA(const char* message)
{
#ifndef LWTX_DISABLE
    lwtxDomainCreateA_impl_fntype local = LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainCreateA_impl_fnptr;
    if(local!=0)
        return (*local)(message);
    else
#endif  /*LWTX_DISABLE*/
        return (lwtxDomainHandle_t)0;
}

LWTX_DECLSPEC lwtxDomainHandle_t LWTX_API lwtxDomainCreateW(const wchar_t* message)
{
#ifndef LWTX_DISABLE
    lwtxDomainCreateW_impl_fntype local = LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainCreateW_impl_fnptr;
    if(local!=0)
        return (*local)(message);
    else
#endif  /*LWTX_DISABLE*/
        return (lwtxDomainHandle_t)0;
}

LWTX_DECLSPEC void LWTX_API lwtxDomainDestroy(lwtxDomainHandle_t domain)
{
#ifndef LWTX_DISABLE
    lwtxDomainDestroy_impl_fntype local = LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxDomainDestroy_impl_fnptr;
    if(local!=0)
        (*local)(domain);
#endif /*LWTX_DISABLE*/
}

LWTX_DECLSPEC void LWTX_API lwtxInitialize(const void* reserved)
{
#ifndef LWTX_DISABLE
    lwtxInitialize_impl_fntype local = LWTX_VERSIONED_IDENTIFIER(lwtxGlobals).lwtxInitialize_impl_fnptr;
    if(local!=0)
        (*local)(reserved);
#endif /*LWTX_DISABLE*/
}
