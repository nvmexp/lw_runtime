[How to get lwplayfair-tools]
$ git clone  https://gitlab-master.lwpu.com/waqara/lwplayfair-tools.git

[Copy LwSciIpc latency data]
1. Copy lwsciipc_end2end.csv and lwsciipc_notify.csv to "lwsciipc_lwplayfair/data/" folder.

[Copy LwSciIpc json files]
1. Copy "lwsciipc_lwplayfair" to "lwplayfair-tools/" folder.
2. Please read lwplayfair-tools/README.md to setup and run lwplayfair-tool.

[lwplayfaire-tool commands for LwSciIpc]
# CD to lwplayfair-tools root folder
# Generate all latency graphs at a time
$ pipelw shell
(lwplayfair-tools) <shell-prompt>$ ./plotter.py -c lwsciipc_lwplayfair/lwsciipc_all.json

# CD to lwplayfair-tools root folder
# Generate latency graphs one by one
$ pipelw shell
(lwplayfair-tools) <shell-prompt>$ ./plotter.py -c lwsciipc_lwplayfair/lwsciipc_histogram.json
(lwplayfair-tools) <shell-prompt>$ ./plotter.py -c lwsciipc_lwplayfair/lwsciipc_boxplot.json
(lwplayfair-tools) <shell-prompt>$ ./plotter.py -c lwsciipc_lwplayfair/lwsciipc_groupedBarPlot.json
(lwplayfair-tools) <shell-prompt>$ ./plotter.py -c lwsciipc_lwplayfair/lwsciipc_notify_timeseries.json
(lwplayfair-tools) <shell-prompt>$ ./plotter.py -c lwsciipc_lwplayfair/lwsciipc_end2end_timeseries.json
