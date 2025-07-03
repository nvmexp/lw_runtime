/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2009-2009 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

void install_will_return_always_with_readRegOpHead_null();

void install_will_return_always_more_than_once();

void install_will_return_always_node_absent_head_present();

void install_regopRead_callback_with_readRegOpHead_null();

void install_regopRead_callback_with_observerNode_present();

void install_regopRead_callback_with_readRegOpHead_not_null();

void install_regopRead_willreturn_count_with_readRegOpHead_null();

void install_regopRead_willreturn_count_with_observerNode_present();

void install_regopRead_willreturn_count_with_readRegOpHead_not_null();

void install_regopRead_willreturn_count_with_willreturn_always();

void install_regopRead_willreturn_count_with_callback();

void intall_write_mirror_with_head_null();

void install_write_mirror_head_not_null_regopnode_not_present();

void install_write_mirror_with_no_other_mirror_present();

void install_write_mirror_with_another_mirror_present();

void install_write_callback_wiht_no_head();

void install_write_callback_with_head_and_regopnode_not_present();

void install_write_callback_with_regponode_present();

void install_write_callback_with_callback_already_present();

void read_register_never_written_no_associated_node();

void read_register_never_written_associated_node();

void read_register_previously_written();

void read_reg_nth_never_written_no_head();

void read_reg_nth_never_written_head_present();

void read_reg_nth_previously_written();

void read_reg_nth_previously_written_n_greater_than_times_written();

void gpu_write_no_head_present();

void gpu_write_head_present_corresponding_node_absent();

void gpu_mirrored_write();

void gpu_write_with_callback();

void gpu_write_node_present_no_mirror_no_callback();

void gpu_read_head_absent_never_written();

void gpu_read_head_absent_previously_written();

void gpu_read_node_absent_never_written();

void gpu_read_node_absent_previously_written();

void gpu_read_callback_present();

void gpu_read_will_return();

void gpu_read_will_return_for_count();

