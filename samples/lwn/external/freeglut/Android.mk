ifdef LW_DISABLED_ANDROID_DOESNT_USE_FREEGLUT

LOCAL_PATH := $(call my-dir)
include $(CLEAR_VARS)

LOCAL_MODULE := freeglut
LOCAL_MULTILIB := 32
#LOCAL_ARM_MODE := arm

LOCAL_CFLAGS += -DFREEGLUT_GLES -DNEED_XPARSEGEOMETRY_IMPL -DFREEGLUT_PRINT_ERRORS -DFREEGLUT_PRINT_WARNINGS

FREEGLUT_SRC := freeglut/freeglut/src
FREEGLUT_INCLUDE := $(LOCAL_PATH)/$(FREEGLUT_SRC)/../include \
                    $(LOCAL_PATH)/$(FREEGLUT_SRC)/../include/GL \
                    $(LOCAL_PATH)/$(FREEGLUT_SRC)
                    
#freeglut stuff
LOCAL_SRC_FILES += $(FREEGLUT_SRC)/fg_callbacks.c
LOCAL_SRC_FILES += $(FREEGLUT_SRC)/fg_lwrsor.c
LOCAL_SRC_FILES += $(FREEGLUT_SRC)/fg_display.c
LOCAL_SRC_FILES += $(FREEGLUT_SRC)/fg_ext.c
#LOCAL_SRC_FILES += $(FREEGLUT_SRC)/fg_font.c
#LOCAL_SRC_FILES += $(FREEGLUT_SRC)/fg_font_data.c
LOCAL_SRC_FILES += $(FREEGLUT_SRC)/fg_gamemode.c
LOCAL_SRC_FILES += $(FREEGLUT_SRC)/fg_geometry.c
LOCAL_SRC_FILES += $(FREEGLUT_SRC)/fg_gl2.c
LOCAL_SRC_FILES += $(FREEGLUT_SRC)/fg_init.c
LOCAL_SRC_FILES += $(FREEGLUT_SRC)/fg_input_devices.c
LOCAL_SRC_FILES += $(FREEGLUT_SRC)/fg_joystick.c
LOCAL_SRC_FILES += $(FREEGLUT_SRC)/fg_main.c
#LOCAL_SRC_FILES += $(FREEGLUT_SRC)/fg_menu.c
LOCAL_SRC_FILES += $(FREEGLUT_SRC)/fg_misc.c
LOCAL_SRC_FILES += $(FREEGLUT_SRC)/fg_overlay.c
LOCAL_SRC_FILES += $(FREEGLUT_SRC)/fg_spaceball.c
LOCAL_SRC_FILES += $(FREEGLUT_SRC)/fg_state.c
LOCAL_SRC_FILES += $(FREEGLUT_SRC)/fg_stroke_mono_roman.c
LOCAL_SRC_FILES += $(FREEGLUT_SRC)/fg_stroke_roman.c
LOCAL_SRC_FILES += $(FREEGLUT_SRC)/fg_structure.c
LOCAL_SRC_FILES += $(FREEGLUT_SRC)/fg_teapot.c
LOCAL_SRC_FILES += $(FREEGLUT_SRC)/fg_videoresize.c
LOCAL_SRC_FILES += $(FREEGLUT_SRC)/fg_window.c
LOCAL_SRC_FILES += $(FREEGLUT_SRC)/gles_stubs.c
LOCAL_SRC_FILES += $(FREEGLUT_SRC)/android/fg_lwrsor_android.c
LOCAL_SRC_FILES += $(FREEGLUT_SRC)/android/fg_ext_android.c
LOCAL_SRC_FILES += $(FREEGLUT_SRC)/android/fg_gamemode_android.c
LOCAL_SRC_FILES += $(FREEGLUT_SRC)/android/fg_init_android.c
LOCAL_SRC_FILES += $(FREEGLUT_SRC)/android/fg_input_devices_android.c
LOCAL_SRC_FILES += $(FREEGLUT_SRC)/android/fg_joystick_android.c
LOCAL_SRC_FILES += $(FREEGLUT_SRC)/android/fg_main_android.c
LOCAL_SRC_FILES += $(FREEGLUT_SRC)/android/fg_runtime_android.c
LOCAL_SRC_FILES += $(FREEGLUT_SRC)/android/fg_spaceball_android.c
LOCAL_SRC_FILES += $(FREEGLUT_SRC)/android/fg_state_android.c
LOCAL_SRC_FILES += $(FREEGLUT_SRC)/android/fg_structure_android.c
LOCAL_SRC_FILES += $(FREEGLUT_SRC)/android/fg_window_android.c
#LOCAL_SRC_FILES += $(FREEGLUT_SRC)/android/native_app_glue/android_native_app_glue.c
LOCAL_SRC_FILES += $(FREEGLUT_SRC)/util/xparsegeometry_repl.c

LOCAL_SRC_FILES += $(FREEGLUT_SRC)/egl/fg_display_egl.c
LOCAL_SRC_FILES += $(FREEGLUT_SRC)/egl/fg_init_egl.c
LOCAL_SRC_FILES += $(FREEGLUT_SRC)/egl/fg_ext_egl.c
LOCAL_SRC_FILES += $(FREEGLUT_SRC)/egl/fg_state_egl.c
LOCAL_SRC_FILES += $(FREEGLUT_SRC)/egl/fg_structure_egl.c
LOCAL_SRC_FILES += $(FREEGLUT_SRC)/egl/fg_window_egl.c

LOCAL_C_INCLUDES += $(FREEGLUT_INCLUDE)

LOCAL_WHOLE_STATIC_LIBRARIES += android_native_app_glue

include $(BUILD_STATIC_LIBRARY)
$(call import-module,android/native_app_glue)

endif
