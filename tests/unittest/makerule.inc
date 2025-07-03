#
# LWIDIA_COPYRIGHT_BEGIN
#
# Copyright by LWPU Corporation.  All rights reserved.  All
# information contained herein is proprietary and confidential to LWPU
# Corporation.  Any use, reproduction, or disclosure without the written
# permission of LWPU Corporation is prohibited.
#
# LWIDIA_COPYRIGHT_END
#

#-------------------------------------------------------------------------------
#Add directory paths to vpaths 
#-------------------------------------------------------------------------------
vpath %.c $(dir $(CFILES)) $(dir $(MOCKED_CFILES)) $(MOCK_DIR)
vpath %.o $(BUILD_CFG)
vpath %.d $(BUILD_CFG)

POST_COMPILE_PIPE_STEP = /showIncludes | $(PERL) $(UT_DIR)/getdeps.pl $< $@ $(@:.obj=.d)

#-------------------------------------------------------------------------------
# Create output directory
#-------------------------------------------------------------------------------

$(o_dir)/dummy.txt:
	$(Q)mkdir -p $(o_dir)
	$(Q)touch $(o_dir)/dummy.txt

$(OBJS): $(o_dir)/dummy.txt

#-------------------------------------------------------------------------------
# Building object files from c files
#-------------------------------------------------------------------------------
$(o_dir)/%$(obj_suffix): %.c
ifeq ($(shell uname),Linux)
	$(Q)echo $(<F)
	$(Q)$(CC) -c $(CFLAGS) $< -o $@ -MD -MF $(@:.o=.d)
else
	$(Q)$(CC) -c $(CFLAGS) $(TARGET_CFLAGS) $< $(POST_COMPILE_PIPE_STEP)
endif

#-------------------------------------------------------------------------------
# Build the exelwtable.
#-------------------------------------------------------------------------------
$(o_dir)/$(exe_name)$(exe_suffix): $(OBJS)
	$(Q)echo creating $(@F)...
ifeq ($(shell uname),Linux)
	$(Q)$(CC) -o $(o_dir)/$(exe_name)$(exe_suffix) $(OBJS) $(LIBS)
else
	$(Q)$(LINK) $(LINKFLAGS) $(OBJS) -out:$(o_dir)/$(exe_name)$(exe_suffix) $(LIBS)
endif

# Adding dummy rules to speeden up the build process

$(o_dir)/%.d: ;
$(CMN_FILES) $(ENG_FILES): ;
%.h : ;
$(UT_DIR)/make*.inc: ;
$(UT_DIR)/Makefile: ;
$(UT_DIR)/unizen/defines.mk: ;

-include $(DEPS)
