# SPDX-License-Identifier: BSD-3-Clause
# Copyright(c) 2018 Intel Corporation

sphinx = find_program('sphinx-build', required: get_option('enable_docs'))

if not sphinx.found()
	subdir_done()
endif

extra_sphinx_args = []
if get_option('werror')
	extra_sphinx_args += '-W'
endif

htmldir = join_paths(get_option('datadir'), 'doc', 'dpdk')
html_guides = lwstom_target('html_guides',
	input: files('index.rst'),
	output: 'html',
	command: [sphinx_wrapper, sphinx, meson.project_version(),
		meson.lwrrent_source_dir(), meson.lwrrent_build_dir(),
		extra_sphinx_args],
	depfile: '.html.d',
	build_by_default: get_option('enable_docs'),
	install: get_option('enable_docs'),
	install_dir: htmldir)

install_data(files('custom.css'),
			install_dir: join_paths(htmldir,'_static', 'css'))

doc_targets += html_guides
doc_target_names += 'HTML_Guides'
