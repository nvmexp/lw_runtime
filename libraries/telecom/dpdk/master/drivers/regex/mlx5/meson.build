# SPDX-License-Identifier: BSD-3-Clause
# Copyright 2020 Mellanox Technologies, Ltd

if not is_linux
	build = false
	reason = 'only supported on Linux'
	subdir_done()
endif

fmt_name = 'mlx5_regex'
deps += ['common_mlx5', 'eal', 'regexdev']
sources = files(
	'mlx5_regex.c',
	'mlx5_rxp.c',
	'mlx5_regex_devx.c',
	'mlx5_regex_control.c',
	'mlx5_regex_fastpath.c',
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
