
//
// Copyright (c) 2018, LWPU CORPORATION.  All rights reserved.
//
// LWPU CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from LWPU CORPORATION is strictly prohibited.
//
///////////////////////////////////////////////////////////////////////////////
//                                                                           
//        THIS FILE IS GENERATED FROM EVENT SCHEMA v0.7, DO NOT MODIFY IT
//        Please use https://sms.gfe.lwpu.com/ to update the schema
//        and generate the code.
//
///////////////////////////////////////////////////////////////////////////////
#pragma once

#include <windows.h>
#include <string>

#include <cstdint>



namespace LwTelemetry
{
  namespace Ansel
  {
    
    enum class StyleTransferStatusEnum : uint8_t
    {
      FORWARD_SUCCESS,
      FORWARD_FAILED,
      FORWARD_FAILED_NOT_ENOUGH_VRAM,
      FORWARD_HDR_SUCCESS,
      FORWARD_HDR_FAILED,
      FORWARD_HDR_FAILED_NOT_ENOUGH_VRAM,
      HDR_COLWERT_FAILED,
      HDR_STORAGE_COLWERT_FAILED,
      INSTALLATION_FAILED,
      DOWNLOADING_FAILED,
      EXCEPTION_OCLWRED,
      OPERATION_FAILED,
      OPERATION_TIMEOUT,
      STARTUP_FAILURE,
      INSTALLATION_SUCCESS,
      DOWNLOAD_STARTUP_FAILED,
      COMPUTE_CAPABILITY_TO_OLD,
      LIBRESTYLE_NOT_FOUND,
      MODEL_NOT_FOUND,
      NOT_ENOUGH_VRAM,
      INITIALIZATION_FAILED,
      LOADING_STYLE_FAILED,
      DECLINED_INSTALLATION,
      ACCEPTED_INSTALLATION
    };
    
    enum class KindSliderEnum : uint8_t
    {
      REGULAR,
      MONO_360,
      HIGHRES,
      STEREO,
      STEREO_360,
      NONE,
      REGULAR_UI
    };
    
    enum class SpecialEffectsModeEnum : uint8_t
    {
      NONE,
      YAML
    };
    
    typedef const std::string& StringType;
    
    typedef float FloatType;
    
    typedef double DoubleType;
    
    typedef int32_t IntType;
    
    typedef uint32_t UintType;
    
    typedef int64_t Int64Type;
    
    typedef int8_t BoolType;
    
    enum class UserConstantTypeEnum : uint8_t
    {
      BOOL,
      INT,
      UINT,
      FLOAT
    };
    
    enum class UserConstantSliderStateEnum : uint8_t
    {
      NOT_CREATED,
      CREATED_VISIBLE_ENABLED
    };
    
    typedef const std::string& PackedAllUserConstants;
    
    typedef uint32_t DxgiFormat;
    
    enum class AnselCaptureState : uint8_t
    {
      CAPTURE_STATE_NOT_STARTED,
      CAPTURE_STATE_STARTED,
      CAPTURE_STATE_ABORT,
      CAPTURE_STATE_REGULAR,
      CAPTURE_STATE_REGULARSTEREO,
      CAPTURE_STATE_HIGHRES,
      CAPTURE_STATE_360,
      CAPTURE_STATE_360STEREO
    };
    
    enum class ErrorType : uint8_t
    {
      HANDLE_FAILURE_FATAL_ERROR,
      EFFECT_COMPILATION_ERROR,
      NON_FATAL_ERROR
    };
    
    enum class ColorRangeType : uint8_t
    {
      RGB,
      EXR
    };
    
    enum class UIModeType : uint8_t
    {
      STANDALONE_UI,
      IPC_UI
    };
    
    enum class GamepadMappingType : uint8_t
    {
      UNKNOWN,
      SHIELD,
      XBOX360,
      XBOXONE,
      DUALSHOCK4
    };
    
    HRESULT Init();
    HRESULT DeInit();
    HRESULT HasFunctionalConsent(const std::string& userId);
    HRESULT HasTechnicalConsent(const std::string& userId);
    HRESULT HasBehavioralConsent(const std::string& userId);
    HRESULT SetDeviceConsent(uint32_t consentFlags);
    HRESULT SetDeviceConsent(const std::string& clientId, uint32_t consentFlags);
    HRESULT SetUserConsent(const std::string& userId, uint32_t consentFlags);
    
    HRESULT Send_StyleTransferDownloadStarted_Event(
      StringType schemaDefinedArg_url,
      StringType schemaDefinedArg_version,
      UintType schemaDefinedArg_computeCapMajor,
      UintType schemaDefinedArg_computeCapMinor,
      
      const std::string& clientVer,
      const std::string& userId,
      const uint64_t* timestampFileTimeFormatOptional = nullptr
    );
  
    HRESULT Send_StyleTransferDownloadFinished_Event(
      UintType schemaDefinedArg_secondsSpent,
      IntType schemaDefinedArg_status,
      
      const std::string& clientVer,
      const std::string& userId,
      const uint64_t* timestampFileTimeFormatOptional = nullptr
    );
  
    HRESULT Send_StyleTransferStatus_Event(
      StyleTransferStatusEnum schemaDefinedArg_status,
      StringType schemaDefinedArg_comment,
      
      const std::string& clientVer,
      const std::string& userId,
      const uint64_t* timestampFileTimeFormatOptional = nullptr
    );
  
    HRESULT Send_CaptureStarted_Event(
      StringType schemaDefinedArg_appExeName,
      StringType schemaDefinedArg_drsProfileName,
      StringType schemaDefinedArg_drsAppName,
      UintType schemaDefinedArg_screenResolutionX,
      UintType schemaDefinedArg_screenResolutionY,
      DxgiFormat schemaDefinedArg_colorBufferFormat,
      DxgiFormat schemaDefinedArg_depthBufferFormat,
      KindSliderEnum schemaDefinedArg_kindOfShot,
      UintType schemaDefinedArg_highresMultiplier,
      FloatType schemaDefinedArg_quality360Fov,
      FloatType schemaDefinedArg_fov,
      FloatType schemaDefinedArg_roll,
      FloatType schemaDefinedArg_lwrrentCameraPosX,
      FloatType schemaDefinedArg_lwrrentCameraPosY,
      FloatType schemaDefinedArg_lwrrentCameraPosZ,
      FloatType schemaDefinedArg_lwrrentCameraRotX,
      FloatType schemaDefinedArg_lwrrentCameraRotY,
      FloatType schemaDefinedArg_lwrrentCameraRotZ,
      FloatType schemaDefinedArg_lwrrentCameraRotW,
      FloatType schemaDefinedArg_originalCameraPosX,
      FloatType schemaDefinedArg_originalCameraPosY,
      FloatType schemaDefinedArg_originalCameraPosZ,
      FloatType schemaDefinedArg_originalCameraRotX,
      FloatType schemaDefinedArg_originalCameraRotY,
      FloatType schemaDefinedArg_originalCameraRotZ,
      FloatType schemaDefinedArg_originalCameraRotW,
      SpecialEffectsModeEnum schemaDefinedArg_specialEffectsMode,
      StringType schemaDefinedArg_effectName,
      UserConstantSliderStateEnum schemaDefinedArg_userConstant0_sliderState,
      StringType schemaDefinedArg_userConstant0_name,
      UserConstantTypeEnum schemaDefinedArg_userConstant0_type,
      StringType schemaDefinedArg_userConstant0_value,
      UserConstantSliderStateEnum schemaDefinedArg_userConstant1_sliderState,
      StringType schemaDefinedArg_userConstant1_name,
      UserConstantTypeEnum schemaDefinedArg_userConstant1_type,
      StringType schemaDefinedArg_userConstant1_value,
      UserConstantSliderStateEnum schemaDefinedArg_userConstant2_sliderState,
      StringType schemaDefinedArg_userConstant2_name,
      UserConstantTypeEnum schemaDefinedArg_userConstant2_type,
      StringType schemaDefinedArg_userConstant2_value,
      UserConstantSliderStateEnum schemaDefinedArg_userConstant3_sliderState,
      StringType schemaDefinedArg_userConstant3_name,
      UserConstantTypeEnum schemaDefinedArg_userConstant3_type,
      StringType schemaDefinedArg_userConstant3_value,
      UserConstantSliderStateEnum schemaDefinedArg_userConstant4_sliderState,
      StringType schemaDefinedArg_userConstant4_name,
      UserConstantTypeEnum schemaDefinedArg_userConstant4_type,
      StringType schemaDefinedArg_userConstant4_value,
      PackedAllUserConstants schemaDefinedArg_allUserConstants,
      UIModeType schemaDefinedArg_uiMode,
      ColorRangeType schemaDefinedArg_colorRange,
      Int64Type schemaDefinedArg_quality360resolution,
      GamepadMappingType schemaDefinedArg_gamepadMapping,
      UintType schemaDefinedArg_gamepadProductId,
      UintType schemaDefinedArg_gamepadVendorId,
      UintType schemaDefinedArg_gamepadVersionNumber,
      BoolType schemaDefinedArg_gamepadUsedForCameraOperationDuringThisSession,
      BoolType schemaDefinedArg_gamepadUsedForUIInteractionDuringThisSession,
      DoubleType schemaDefinedArg_timeElapsedSinceSessionStarted,
      BoolType schemaDefinedArg_highresEnhancementEnabled,
      UintType schemaDefinedArg_anselSDKMajor,
      UintType schemaDefinedArg_anselSDKMinor,
      UintType schemaDefinedArg_anselSDKCommit,
      
      const std::string& clientVer,
      const std::string& userId,
      const uint64_t* timestampFileTimeFormatOptional = nullptr
    );
  
    HRESULT Send_CaptureAborted_Event(
      StringType schemaDefinedArg_appExeName,
      StringType schemaDefinedArg_drsProfileName,
      StringType schemaDefinedArg_drsAppName,
      UintType schemaDefinedArg_screenResolutionX,
      UintType schemaDefinedArg_screenResolutionY,
      DxgiFormat schemaDefinedArg_colorBufferFormat,
      DxgiFormat schemaDefinedArg_depthBufferFormat,
      KindSliderEnum schemaDefinedArg_kindOfShot,
      UintType schemaDefinedArg_highresMultiplier,
      FloatType schemaDefinedArg_quality360Fov,
      FloatType schemaDefinedArg_fov,
      FloatType schemaDefinedArg_roll,
      FloatType schemaDefinedArg_lwrrentCameraPosX,
      FloatType schemaDefinedArg_lwrrentCameraPosY,
      FloatType schemaDefinedArg_lwrrentCameraPosZ,
      FloatType schemaDefinedArg_lwrrentCameraRotX,
      FloatType schemaDefinedArg_lwrrentCameraRotY,
      FloatType schemaDefinedArg_lwrrentCameraRotZ,
      FloatType schemaDefinedArg_lwrrentCameraRotW,
      FloatType schemaDefinedArg_originalCameraPosX,
      FloatType schemaDefinedArg_originalCameraPosY,
      FloatType schemaDefinedArg_originalCameraPosZ,
      FloatType schemaDefinedArg_originalCameraRotX,
      FloatType schemaDefinedArg_originalCameraRotY,
      FloatType schemaDefinedArg_originalCameraRotZ,
      FloatType schemaDefinedArg_originalCameraRotW,
      SpecialEffectsModeEnum schemaDefinedArg_specialEffectsMode,
      StringType schemaDefinedArg_effectName,
      UserConstantSliderStateEnum schemaDefinedArg_userConstant0_sliderState,
      StringType schemaDefinedArg_userConstant0_name,
      UserConstantTypeEnum schemaDefinedArg_userConstant0_type,
      StringType schemaDefinedArg_userConstant0_value,
      UserConstantSliderStateEnum schemaDefinedArg_userConstant1_sliderState,
      StringType schemaDefinedArg_userConstant1_name,
      UserConstantTypeEnum schemaDefinedArg_userConstant1_type,
      StringType schemaDefinedArg_userConstant1_value,
      UserConstantSliderStateEnum schemaDefinedArg_userConstant2_sliderState,
      StringType schemaDefinedArg_userConstant2_name,
      UserConstantTypeEnum schemaDefinedArg_userConstant2_type,
      StringType schemaDefinedArg_userConstant2_value,
      UserConstantSliderStateEnum schemaDefinedArg_userConstant3_sliderState,
      StringType schemaDefinedArg_userConstant3_name,
      UserConstantTypeEnum schemaDefinedArg_userConstant3_type,
      StringType schemaDefinedArg_userConstant3_value,
      UserConstantSliderStateEnum schemaDefinedArg_userConstant4_sliderState,
      StringType schemaDefinedArg_userConstant4_name,
      UserConstantTypeEnum schemaDefinedArg_userConstant4_type,
      StringType schemaDefinedArg_userConstant4_value,
      PackedAllUserConstants schemaDefinedArg_allUserConstants,
      UIModeType schemaDefinedArg_uiMode,
      ColorRangeType schemaDefinedArg_colorRange,
      Int64Type schemaDefinedArg_quality360resolution,
      DoubleType schemaDefinedArg_timeElapsedSinceCaptureStarted,
      BoolType schemaDefinedArg_highresEnhancementEnabled,
      
      const std::string& clientVer,
      const std::string& userId,
      const uint64_t* timestampFileTimeFormatOptional = nullptr
    );
  
    HRESULT Send_AnselUIClosed_Event(
      StringType schemaDefinedArg_appExeName,
      StringType schemaDefinedArg_drsProfileName,
      StringType schemaDefinedArg_drsAppName,
      UintType schemaDefinedArg_screenResolutionX,
      UintType schemaDefinedArg_screenResolutionY,
      DxgiFormat schemaDefinedArg_colorBufferFormat,
      DxgiFormat schemaDefinedArg_depthBufferFormat,
      KindSliderEnum schemaDefinedArg_kindOfShot,
      UintType schemaDefinedArg_highresMultiplier,
      FloatType schemaDefinedArg_quality360Fov,
      FloatType schemaDefinedArg_fov,
      FloatType schemaDefinedArg_roll,
      FloatType schemaDefinedArg_lwrrentCameraPosX,
      FloatType schemaDefinedArg_lwrrentCameraPosY,
      FloatType schemaDefinedArg_lwrrentCameraPosZ,
      FloatType schemaDefinedArg_lwrrentCameraRotX,
      FloatType schemaDefinedArg_lwrrentCameraRotY,
      FloatType schemaDefinedArg_lwrrentCameraRotZ,
      FloatType schemaDefinedArg_lwrrentCameraRotW,
      FloatType schemaDefinedArg_originalCameraPosX,
      FloatType schemaDefinedArg_originalCameraPosY,
      FloatType schemaDefinedArg_originalCameraPosZ,
      FloatType schemaDefinedArg_originalCameraRotX,
      FloatType schemaDefinedArg_originalCameraRotY,
      FloatType schemaDefinedArg_originalCameraRotZ,
      FloatType schemaDefinedArg_originalCameraRotW,
      SpecialEffectsModeEnum schemaDefinedArg_specialEffectsMode,
      StringType schemaDefinedArg_effectName,
      UserConstantSliderStateEnum schemaDefinedArg_userConstant0_sliderState,
      StringType schemaDefinedArg_userConstant0_name,
      UserConstantTypeEnum schemaDefinedArg_userConstant0_type,
      StringType schemaDefinedArg_userConstant0_value,
      UserConstantSliderStateEnum schemaDefinedArg_userConstant1_sliderState,
      StringType schemaDefinedArg_userConstant1_name,
      UserConstantTypeEnum schemaDefinedArg_userConstant1_type,
      StringType schemaDefinedArg_userConstant1_value,
      UserConstantSliderStateEnum schemaDefinedArg_userConstant2_sliderState,
      StringType schemaDefinedArg_userConstant2_name,
      UserConstantTypeEnum schemaDefinedArg_userConstant2_type,
      StringType schemaDefinedArg_userConstant2_value,
      UserConstantSliderStateEnum schemaDefinedArg_userConstant3_sliderState,
      StringType schemaDefinedArg_userConstant3_name,
      UserConstantTypeEnum schemaDefinedArg_userConstant3_type,
      StringType schemaDefinedArg_userConstant3_value,
      UserConstantSliderStateEnum schemaDefinedArg_userConstant4_sliderState,
      StringType schemaDefinedArg_userConstant4_name,
      UserConstantTypeEnum schemaDefinedArg_userConstant4_type,
      StringType schemaDefinedArg_userConstant4_value,
      PackedAllUserConstants schemaDefinedArg_allUserConstants,
      UIModeType schemaDefinedArg_uiMode,
      ColorRangeType schemaDefinedArg_colorRange,
      Int64Type schemaDefinedArg_quality360resolution,
      GamepadMappingType schemaDefinedArg_gamepadMapping,
      UintType schemaDefinedArg_gamepadProductId,
      UintType schemaDefinedArg_gamepadVendorId,
      UintType schemaDefinedArg_gamepadVersionNumber,
      BoolType schemaDefinedArg_gamepadUsedForCameraOperationDuringThisSession,
      BoolType schemaDefinedArg_gamepadUsedForUIInteractionDuringThisSession,
      DoubleType schemaDefinedArg_timeElapsedSinceSessionStarted,
      BoolType schemaDefinedArg_highresEnhancementEnabled,
      UintType schemaDefinedArg_anselSDKMajor,
      UintType schemaDefinedArg_anselSDKMinor,
      UintType schemaDefinedArg_anselSDKCommit,
      
      const std::string& clientVer,
      const std::string& userId,
      const uint64_t* timestampFileTimeFormatOptional = nullptr
    );
  
    HRESULT Send_AnselErrorOclwredFull_Event(
      StringType schemaDefinedArg_appExeName,
      StringType schemaDefinedArg_drsProfileName,
      StringType schemaDefinedArg_drsAppName,
      UintType schemaDefinedArg_screenResolutionX,
      UintType schemaDefinedArg_screenResolutionY,
      DxgiFormat schemaDefinedArg_colorBufferFormat,
      DxgiFormat schemaDefinedArg_depthBufferFormat,
      KindSliderEnum schemaDefinedArg_kindOfShot,
      UintType schemaDefinedArg_highresMultiplier,
      FloatType schemaDefinedArg_quality360Fov,
      FloatType schemaDefinedArg_fov,
      FloatType schemaDefinedArg_roll,
      FloatType schemaDefinedArg_lwrrentCameraPosX,
      FloatType schemaDefinedArg_lwrrentCameraPosY,
      FloatType schemaDefinedArg_lwrrentCameraPosZ,
      FloatType schemaDefinedArg_lwrrentCameraRotX,
      FloatType schemaDefinedArg_lwrrentCameraRotY,
      FloatType schemaDefinedArg_lwrrentCameraRotZ,
      FloatType schemaDefinedArg_lwrrentCameraRotW,
      FloatType schemaDefinedArg_originalCameraPosX,
      FloatType schemaDefinedArg_originalCameraPosY,
      FloatType schemaDefinedArg_originalCameraPosZ,
      FloatType schemaDefinedArg_originalCameraRotX,
      FloatType schemaDefinedArg_originalCameraRotY,
      FloatType schemaDefinedArg_originalCameraRotZ,
      FloatType schemaDefinedArg_originalCameraRotW,
      SpecialEffectsModeEnum schemaDefinedArg_specialEffectsMode,
      StringType schemaDefinedArg_effectName,
      UserConstantSliderStateEnum schemaDefinedArg_userConstant0_sliderState,
      StringType schemaDefinedArg_userConstant0_name,
      UserConstantTypeEnum schemaDefinedArg_userConstant0_type,
      StringType schemaDefinedArg_userConstant0_value,
      UserConstantSliderStateEnum schemaDefinedArg_userConstant1_sliderState,
      StringType schemaDefinedArg_userConstant1_name,
      UserConstantTypeEnum schemaDefinedArg_userConstant1_type,
      StringType schemaDefinedArg_userConstant1_value,
      UserConstantSliderStateEnum schemaDefinedArg_userConstant2_sliderState,
      StringType schemaDefinedArg_userConstant2_name,
      UserConstantTypeEnum schemaDefinedArg_userConstant2_type,
      StringType schemaDefinedArg_userConstant2_value,
      UserConstantSliderStateEnum schemaDefinedArg_userConstant3_sliderState,
      StringType schemaDefinedArg_userConstant3_name,
      UserConstantTypeEnum schemaDefinedArg_userConstant3_type,
      StringType schemaDefinedArg_userConstant3_value,
      UserConstantSliderStateEnum schemaDefinedArg_userConstant4_sliderState,
      StringType schemaDefinedArg_userConstant4_name,
      UserConstantTypeEnum schemaDefinedArg_userConstant4_type,
      StringType schemaDefinedArg_userConstant4_value,
      PackedAllUserConstants schemaDefinedArg_allUserConstants,
      UIModeType schemaDefinedArg_uiMode,
      ColorRangeType schemaDefinedArg_colorRange,
      Int64Type schemaDefinedArg_quality360resolution,
      GamepadMappingType schemaDefinedArg_gamepadMapping,
      UintType schemaDefinedArg_gamepadProductId,
      UintType schemaDefinedArg_gamepadVendorId,
      UintType schemaDefinedArg_gamepadVersionNumber,
      BoolType schemaDefinedArg_gamepadUsedForCameraOperationDuringThisSession,
      BoolType schemaDefinedArg_gamepadUsedForUIInteractionDuringThisSession,
      DoubleType schemaDefinedArg_timeElapsedSinceSessionStarted,
      DoubleType schemaDefinedArg_timeElapsedSinceCaptureStarted,
      AnselCaptureState schemaDefinedArg_captureStateOnError,
      ErrorType schemaDefinedArg_errorType,
      StringType schemaDefinedArg_sourceFilename,
      UintType schemaDefinedArg_sourceLine,
      StringType schemaDefinedArg_errorString,
      UintType schemaDefinedArg_errorCode,
      BoolType schemaDefinedArg_highresEnhancementEnabled,
      UintType schemaDefinedArg_anselSDKMajor,
      UintType schemaDefinedArg_anselSDKMinor,
      UintType schemaDefinedArg_anselSDKCommit,
      
      const std::string& clientVer,
      const std::string& userId,
      const uint64_t* timestampFileTimeFormatOptional = nullptr
    );
  
    HRESULT Send_AnselErrorOclwredShort_Event(
      ErrorType schemaDefinedArg_errorType,
      StringType schemaDefinedArg_sourceFilename,
      UintType schemaDefinedArg_sourceLine,
      StringType schemaDefinedArg_errorString,
      UintType schemaDefinedArg_errorCode,
      
      const std::string& clientVer,
      const std::string& userId,
      const uint64_t* timestampFileTimeFormatOptional = nullptr
    );
  
  }
}
