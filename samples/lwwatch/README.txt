
https://wiki.lwpu.com/engwiki/index.php/Lwwatch

lwwatch/

    bin/                  - Exelwtables

    config/               - Chip-config related files

    gpu/                  - Source files for GPU and Engines
        fifo/
        disp/
        instmem/
        misc/
        ...
    os/                   - Platform specfic implementations
        common/
        unix/
            mmap/
            jtag/
            ...
        mods/
            unix/
            win/
            ...
    common/               - Infrastructure or shared code
        halstubs.c
        backdoor.c
        help.c
        ...
    inc/                  - Headers
        hwref/
    tools/
        mcheck/

    README
