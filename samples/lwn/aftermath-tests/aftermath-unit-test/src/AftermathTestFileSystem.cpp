/*
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include <string>

#include <nn/fs.h>
#include <nn/fs/fs_Debug.h>
#define NN_ENABLE_HTC
#include <nn/htc.h>
#include <nn/settings/fwdbg/settings_SettingsGetterApi.h>

#include <AftermathTest.h>
#include <AftermathTestLogging.h>
#include <AftermathTestUtils.h>

#include <AftermathFileSystem.h>

using namespace Aftermath;

namespace AftermathTest {

static bool ResolveElwVarsInPath(std::string& path)
{
    struct ScopedHtc {
        ScopedHtc()
        {
            nn::htc::Initialize();
        }
        ~ScopedHtc()
        {
            nn::htc::Finalize();
        }
    } scopedHtc;

    std::string resolvedPath;

    for (const char* src = path.data(); *src != '\0'; ++src) {
        if (*src == '%') {
            ++src;
            const char *elwVarStart = src;
            for (; *src != '\0' && *src != '%'; ++src) {
            }
            std::string elwVar;
            elwVar.assign(elwVarStart, src - elwVarStart);

            size_t elwVarSize = 0;
            nn::Result result = nn::htc::GetElwironmentVariableLength(&elwVarSize, elwVar.c_str());
            if (result.IsFailure() || elwVarSize == 0)
            {
                return false;
            }
            std::string elwVarValue;
            elwVarValue.resize(elwVarSize - 1); // exclude 0-termination
            result = nn::htc::GetElwironmentVariable(&elwVarSize, &elwVarValue[0], elwVarSize, elwVar.c_str());
            if (result.IsFailure() || elwVarSize == 0)
            {
                return false;
            }
            resolvedPath.append(elwVarValue);
        } else {
            resolvedPath.push_back(*src);
        }
    }

    path.swap(resolvedPath);

    return true;
}

static bool CheckDirExistsAndAccessible(const std::string& dirPath)
{
    nn::fs::DirectoryHandle directoryHandle;
    const nn::Result result = nn::fs::OpenDirectory(&directoryHandle, dirPath.c_str(), nn::fs::OpenDirectoryMode_All);
    if (result.IsSuccess()) {
        nn::fs::CloseDirectory(directoryHandle);
    }
    return result.IsSuccess();
}

static std::string GetCrashDumpDirBackupPath(const std::string& crashDumpPath)
{
    return crashDumpPath + ".AftermathTestBackup";
}

static bool BackupCrashDumpDir(const std::string& crashDumpPath)
{
    const std::string crashDumpBackupPath = GetCrashDumpDirBackupPath(crashDumpPath);

    // Remove any old backup directory from earlier failed tests
    if (CheckDirExistsAndAccessible(crashDumpBackupPath)) {
        const nn::Result result = nn::fs::DeleteDirectoryRelwrsively(crashDumpBackupPath.c_str());
        TEST(result.IsSuccess());
    }

    // Rename the crash dump dir to avoid clobbering real dump data
    if (CheckDirExistsAndAccessible(crashDumpPath)) {
        const nn::Result result = nn::fs::RenameDirectory(crashDumpPath.c_str(), crashDumpBackupPath.c_str());
        TEST(result.IsSuccess());
    }

    return true;
}

static bool RestoreCrashDumpDir(const std::string& crashDumpPath)
{
    const std::string crashDumpBackupPath = GetCrashDumpDirBackupPath(crashDumpPath);

    // Delete the crash dump directory the test created
    if (CheckDirExistsAndAccessible(crashDumpPath)) {
        const nn::Result result = nn::fs::DeleteDirectoryRelwrsively(crashDumpPath.c_str());
        TEST(result.IsSuccess());
    }

    // Rename back the previously renamed original crash dump directory
    if (CheckDirExistsAndAccessible(crashDumpBackupPath)) {
        const nn::Result result = nn::fs::RenameDirectory(crashDumpBackupPath.c_str(), crashDumpPath.c_str());
        TEST(result.IsSuccess());
    }

    return true;
}

static bool GetHostCrashDumpDir(std::string& crashDumpPath)
{
    crashDumpPath.clear();

    const size_t pathLength = nn::settings::fwdbg::GetSettingsItemValueSize("snap_shot_dump", "output_dir");
    TEST(pathLength > 0);

    std::string path;
    path.resize(pathLength - 1); // exclude 0-termination
    TEST_EQ(nn::settings::fwdbg::GetSettingsItemValue(&path[0], pathLength, "snap_shot_dump", "output_dir"), pathLength);
    TEST(!path.empty());

    TEST(ResolveElwVarsInPath(path));
    TEST(!path.empty());

    crashDumpPath.swap(path);

    return true;
}

static bool BackupHostCrashDumpDir()
{
    std::string crashDumpPath;
    TEST(GetHostCrashDumpDir(crashDumpPath));

    Utils::ScopedHostRootMount hostRootMount;
    TEST(hostRootMount.Ready());

    TEST(BackupCrashDumpDir(crashDumpPath));

    return true;
}

static bool RestoreHostCrashDumpDir()
{
    std::string crashDumpPath;
    TEST(GetHostCrashDumpDir(crashDumpPath));

    Utils::ScopedHostRootMount hostRootMount;
    TEST(hostRootMount.Ready());

    TEST(RestoreCrashDumpDir(crashDumpPath));

    return true;
}

static bool CheckHostCrashDumpDirExists()
{
    std::string crashDumpPath;
    TEST(GetHostCrashDumpDir(crashDumpPath));

    Utils::ScopedHostRootMount hostRootMount;
    TEST(hostRootMount.Ready());

    TEST(CheckDirExistsAndAccessible(crashDumpPath));

    return true;
}

static bool HasSdCard()
{
    Utils::ScopedSdCardMount sdCardMount;
    return !sdCardMount.NoCard();
}

static bool GetSdCardCrashDumpDir(std::string& crashDumpPath)
{
    crashDumpPath = "AmTestSD:/NXDMP";
    return true;
}

static bool BackupSdCardCrashDumpDir()
{
    std::string crashDumpPath;
    TEST(GetSdCardCrashDumpDir(crashDumpPath));

    Utils::ScopedSdCardMount sdCardMount;
    TEST(sdCardMount.Ready());

    TEST(BackupCrashDumpDir(crashDumpPath));

    return true;
}

static bool RestoreSdCardCrashDumpDir()
{
    std::string crashDumpPath;
    TEST(GetSdCardCrashDumpDir(crashDumpPath));

    Utils::ScopedSdCardMount sdCardMount;
    TEST(sdCardMount.Ready());

    TEST(RestoreCrashDumpDir(crashDumpPath));

    return true;
}

static bool CheckSdCardCrashDumpDirExists()
{
    std::string crashDumpPath;
    TEST(GetSdCardCrashDumpDir(crashDumpPath));

    Utils::ScopedSdCardMount sdCardMount;
    TEST(sdCardMount.Ready());

    TEST(CheckDirExistsAndAccessible(crashDumpPath));

    return true;
}

static bool TestFileSystemHost()
{
    // Start without an existing host crash dump dir
    TEST(BackupHostCrashDumpDir());

    // Create host crash dump dir
    {
        CrashDumpFileSystem fs(CrashDumpFileSystem::Mode::ForceHost);
        TEST(fs.Ready());
        TEST(CheckHostCrashDumpDirExists());
    }

    // Use existing host crash dump dir
    {
        CrashDumpFileSystem fs(CrashDumpFileSystem::Mode::ForceHost);
        TEST(fs.Ready());
        TEST(CheckHostCrashDumpDirExists());
    }

    // Remove the new crash dump dir the test created and restore the one
    // backup
    TEST(RestoreHostCrashDumpDir());

    return true;
}

static bool TestFileSystemSdCard()
{
    // Start without an existing crash dump dir on the SD card
    TEST(BackupSdCardCrashDumpDir());

    // Create crash dump dir on the SD card
    {
        CrashDumpFileSystem fs(CrashDumpFileSystem::Mode::ForceSdCard);
        TEST(fs.Ready());
        TEST(CheckSdCardCrashDumpDirExists());
    }

    // Use existing crash dump dir on the Sd card
    {
        CrashDumpFileSystem fs(CrashDumpFileSystem::Mode::ForceSdCard);
        TEST(fs.Ready());
        TEST(CheckSdCardCrashDumpDirExists());
    }

    // Remove the new crash dump dir the test created and restore the one
    // backup
    TEST(RestoreSdCardCrashDumpDir());

    return true;
}

static bool TestFileSystem(const Options& options)
{
    (void)options;

    // Host test
    bool result = TestFileSystemHost();

    // SD card test
    if (HasSdCard()) {
        result &= TestFileSystemSdCard();
    }

    return result;
}

// Integration test - requires active TM (and optional SD card)!
AFTERMATH_DEFINE_TEST(FileSystem, INTEGRATION,
    LwError Execute(const Options& options) {
        return TestFileSystem(options) ? LwSuccess : LwError_IlwalidState;
    }
);

} // namespace AftermathTest
