#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <direct.h> // _getcwd

#include "InteractiveWindow.h"
#include "IOStub.h"
#include "D3DCompilerHandler.h"

#include "EffectParser.h"

#include "Log.h"

int wmain(int argc, wchar_t ** argv)
{
#if 0
    shadermod::effectParser::EffectParser parser;
    shadermod::MultipassConfigParserError err = parser.parse(L"yaml_raw.txt");

    printf("\n%s\n", err.getFullErrorMessage().c_str());
    system("pause");

    return 0;
#endif
        
    const wchar_t * filename = L"externalIP\\test.bmp";
    const wchar_t * filename_shader = L"shaders\\fxaa.fx";

    wchar_t workingDir[FILENAME_MAX];
    if (!_wgetcwd( workingDir, FILENAME_MAX ))
    {
        return errno;
    }

    if (argc == 3 || argc == 4)
    {
        wchar_t filename_out[256];
        int i, filename_len = (int)wcslen(argv[1]);
        for (i = filename_len-1; i >= 0 && (argv[1][i] != L'.'); --i);
        memcpy(filename_out, argv[1], sizeof(wchar_t)* i);
        filename_out[i] = L'_';
        filename_out[i+1] = L'o';
        memcpy(filename_out + i + 2, argv[1] + i, sizeof(wchar_t) * (filename_len - i));
        filename_out[filename_len+2] = 0;

        int yamlFileArgN = argc-1;
        int j, shadername_len = (int)wcslen(argv[yamlFileArgN]);
        for (j = shadername_len - 1; j >= 0 && (argv[yamlFileArgN][j] != L'.'); --j);

        if (j >= 0 && j + 4 < shadername_len && argv[yamlFileArgN][j + 1] == L'y' && argv[yamlFileArgN][j + 2] == L'a' && argv[yamlFileArgN][j + 3] == L'm' && argv[yamlFileArgN][j + 4] == L'l')
        {
            shadermod::processFileD3D11MultiPass(argv[1], (argc == 4) ? argv[2] : nullptr, filename_out, workingDir, argv[yamlFileArgN]);
        }
        else
        {
            shadermod::processFileD3D11(argv[1], filename_out, workingDir, argv[2]);
        }
        
        system("pause");
    }
    else
    {
        shadermod::InteractiveWindow interactiveWindow(workingDir);

        interactiveWindow.spawnInteractiveWindow();
    }

    return 0;
}
