# SPDX-License-Identifier: BSD-3-Clause
# Copyright 2019 Mellanox Technologies, Ltd

if not is_linux
	build = false
	reason = 'only supported on Linux'
	subdir_done()
endif

fmt_name = 'mlx5_vdpa'
deps += ['hash', 'common_mlx5', 'vhost', 'pci', 'eal', 'sched']
sources = files(
	'mlx5_vdpa.c',
	'mlx5_vdpa_mem.c',
	'mlx5_vdpa_event.c',
	'mlx5_vdpa_virtq.c',
	'mlx5_vdpa_steer.c',
	'mlx5_vdpa_lm.c',
)
cflags_options = [
	'-std=c11',
	'-Wno-strict-prototypes',
	'-D_BSD_SOURCE',
	'-D_DEFAULT_SOURCE',
	'-D_XOPEN_SOURCE=600'
]
foreach option:cflags_options
	if cc.has_argument(option)
		cflags += option
	endif
endforeach

if get_option('buildtype').contains('debug')
	cflags += [ '-pedantic', '-DPEDANTIC' ]
else
	cflags += [ '-UPEDANTIC' ]
endif
