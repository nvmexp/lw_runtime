#include <iostream>
#include <conio.h>
#include <vector>
#include <fstream>
#include <locale>
#include <codecvt>

#include "..\..\Darkroom\include\darkroom\StringColwersion.h"
#include "..\..\ShaderMod\source\DenylistParser.h"

bool validateFileExists(const std::wstring& absolutePath)
{
    //Seems like all existence functions fail for files surrounded by double quotes
    return (_waccess(absolutePath.c_str(), 0) == 0);
}

int main(int argc, char *argv[])
{
    std::vector<std::string> knownPRDFiles;
    knownPRDFiles.push_back("commonProfiles.prd");
    knownPRDFiles.push_back("displayProfiles.prd");
    knownPRDFiles.push_back("dxProfiles.prd");
    knownPRDFiles.push_back("gfeProfiles.prd");
    knownPRDFiles.push_back("monitorProfiles.prd");
    knownPRDFiles.push_back("stereoProfiles.prd");

    if (argc == 1)
    {
        printf("\n");
        printf("  Usage:\n");
        printf("      This tool can be used to validate all the ansel denylist strings in a set of prd files.\n");
        printf("      You may do this one of 2 ways:\n");
        printf("          1) Highlight all prd files you want to test, and drag and drop all of them onto this exe at the same time.\n");
        printf("             Or you can list each of the files as a command line argument.\n");
        printf("          2) Drag and drop the \"app_profiles\" folder that contains the prd files onto this exe.\n");
        printf("             Or you can specify the folder as a command line argument.\n");
        printf("             For example, this folder or one of the release branch equivalents:\n");
        printf("               //sw/dev/gpu_drv/bugfix_main/drivers/common/app_profiles/\n");
        printf("             This will automatically test the following files in that folder:\n");
        for (auto knownFileItr = knownPRDFiles.begin(); knownFileItr != knownPRDFiles.end(); knownFileItr++)
        {
            printf("                %s\n", knownFileItr->c_str());
        }
        printf("      Any errors in parsing the denylists will then be printed out here.\n");
        printf("      Optionally, you can always pipe this output to a file via the command line > file.txt\n");
        printf("\n");
        printf("  Press any key to exit...\n");
        _getch();
        return 1;
    }

    std::vector<std::string> filePaths;
    for (int i = 1; i < argc; ++i)
    {
        std::string lwrPath = std::string(argv[i]);

        if (lwrPath.empty())
        {
            continue;
        }

        // Check for a prd file
        size_t lastDotPos = lwrPath.find_last_of('.');
        std::string lwrExtension = (lastDotPos != std::string::npos) ? lwrPath.substr(lastDotPos + 1) : "";
        if (lwrExtension == "prd")
        {
            if (validateFileExists(darkroom::getWstrFromUtf8(lwrPath)))
            {
                filePaths.push_back(lwrPath);
            }
        }
        else
        {
            if (lwrPath[lwrPath.size() - 1] != '\\' && lwrPath[lwrPath.size() - 1] != '/')
            {
                lwrPath += '\\';
            }
            for (auto prdFileItr = knownPRDFiles.begin(); prdFileItr != knownPRDFiles.end(); prdFileItr++)
            {
                std::string lwrPRDFilepath = lwrPath + *prdFileItr;
                if (validateFileExists(darkroom::getWstrFromUtf8(lwrPRDFilepath)))
                {
                    filePaths.push_back(lwrPRDFilepath);
                }
            }
        }
    }

    std::cout << "Processing the following files:\n";
    for (auto fileNameItr = filePaths.begin(); fileNameItr != filePaths.end(); fileNameItr++)
    {
        std::cout << "  " << *fileNameItr << '\n';
    }
    std::cout << '\n';

    bool allDenylistsAreValid = true;
    for (auto fileNameItr = filePaths.begin(); fileNameItr != filePaths.end(); fileNameItr++)
    {
        std::cout << '\n';
        std::cout << "Now looking for denylist strings in:\n";
        std::cout << "  " << *fileNameItr << '\n';

        // open as a byte stream
        std::wifstream fin(*fileNameItr, std::ios::binary);

        std::wstring wline = L"";
        if (!std::getline(fin, wline))
        {
            continue;
        }

        bool utf16BOM = false;
        if (wline.size() >= 2 && wline[0] == 0xFF && wline[1] == 0xFE)
        {
            utf16BOM = true;
        }

        // Return to start of file.
        fin.clear();
        fin.seekg(0, std::ios::beg);

        if (utf16BOM)
        {
            // apply BOM-sensitive UTF-16 facet
            fin.imbue(std::locale(fin.getloc(),
                new std::codecvt_utf16<wchar_t, 0x10ffff, std::consume_header>));
        }

        if (fin.is_open())
        {
            AnselDenylist lwrDenylist;
            std::unordered_set<std::wstring> denylistStrings;

            int lwrLine = 0;
            while (std::getline(fin, wline))
            {
                std::string line = darkroom::getUtf8FromWstr(wline);
                lwrLine++;

                if (line.find("ANSEL_DENYLIST_") != std::string::npos)
                {
                    size_t posOfFirstQuote = line.find('"');
                    size_t posOfLastQuote = line.find_last_of('"');
                    if (posOfFirstQuote == std::string::npos || posOfLastQuote == std::string::npos)
                    {
                        std::cout << '\n';
                        std::cout << "ERROR: Ill formatted denylist string setting on line " << lwrLine << ". Setting must be enclosed in quotes(\"):\n" << line << '\n';
                        std::cout << '\n';
                        allDenylistsAreValid = false;
                        continue;
                    }
                    size_t posOfDenylistStringStart = posOfFirstQuote + 1;
                    size_t lenOfDenylistString = (posOfLastQuote)-posOfDenylistStringStart;
                    std::string lwrDenylistString = line.substr(posOfDenylistStringStart, lenOfDenylistString);
                    std::cout << "    Line " << lwrLine << ": " << line << '\n';
                    denylistStrings.insert(darkroom::getWstrFromUtf8(lwrDenylistString));
                }
            }
            fin.close();

            if (denylistStrings.empty())
            {
                std::cout << "    No denylist strings found...\n";
            }
            else
            {
                std::cout << '\n';
                std::cout << "  Now validating each of these strings...\n";
                std::cout << '\n';

                lwrDenylist.Initialize(L"", denylistStrings);
                bool theseDenylistsAreValid = lwrDenylist.CheckThatAllDenylistsAreValid();
                allDenylistsAreValid &= theseDenylistsAreValid;
                if (!theseDenylistsAreValid)
                {
                    std::cout << "ERROR: Denylist validation failed for:\n";
                    std::cout << "         " << *fileNameItr << "\n";
                }
            }
        }
        else
        {
            std::cout << "ERROR: failed to open file: " << *fileNameItr << '\n';
        }
    }

    std::cout << '\n';
    if (allDenylistsAreValid)
    {
        std::cout << "Validation complete. No denylist string errors found.\n";
    }
    else
    {
        std::cout << "ERROR: Errors were encountered when validating all denylist strings.\n";
    }

    std::cout << '\n';
    std::cout << "Press any key to exit...\n";
    _getch();

    return 0;
}