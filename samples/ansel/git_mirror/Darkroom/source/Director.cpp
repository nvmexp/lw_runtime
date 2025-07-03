#include "darkroom/Director.h"
#include "darkroom/InternalLimits.h"
#include "darkroom/StringColwersion.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <functional>
#include <sstream>

#pragma warning(push)
#pragma warning(disable:4668)
#define WINDOWS_LEAN_AND_MEAN
#include <Windows.h>
#pragma warning(pop)

namespace
{
    const float kPi = 3.14159265358979323846f;

    template<typename T>
    bool in_range(const T x, const T l, const T h)
    {
        return x >= l && x <= h;
    }

    template<class T>
    std::string to_string(T i)
    {
        std::stringstream ss;
        std::string s;
        ss << i;
        s = ss.str();

        return s;
    }
    
    // Generate capture plan for a given 360 ShotDescription
    darkroom::Error generateSphericalShots(
        std::vector<darkroom::CameraSpec>& sequence,
        std::vector<darkroom::TileUV>& sequenceTileUVs,
        std::vector<std::string>& sequenceNames,
        const darkroom::ShotDescription& desc,
        const char* captureFileName,
        const char* suffix,
        float eyeOffset)
    {
        using darkroom::Error;
        using darkroom::SphericalTileName;
        using darkroom::Position;
        using darkroom::Angles;
        using darkroom::Projection;
        using darkroom::TileUV;

        // The following two lines illustrate Hugin command line instructions
        // pano_modify --calwas=AUTO --crop=AUTO -o c:\pano\out.pto c:\pano\capture.txt
        // hugin_exelwtor /s /p=final c:\pano\out.pto
        //
        uint32_t findex = 0;
        const std::wstring captureFile = std::wstring(desc.path) + darkroom::getWstrFromUtf8(captureFileName);
        std::ofstream out(captureFile);

        if (!out)
            return Error::kCouldntCreateFile;

        const float height = float(desc.bmpHeight);
        const float width = float(desc.bmpWidth);

        float horizontalFov = desc.horizontalFov;

        if (horizontalFov > s_maxSphericalTilesHorizontalFov)
            horizontalFov = s_maxSphericalTilesHorizontalFov;

        size_t estimatedEquirectWidth = desc.panoWidth;

        if (estimatedEquirectWidth == 0)
            estimatedEquirectWidth = size_t((360.0f / horizontalFov) * width + 0.5f) & ~1u; // round it to 2 so that height is always exactly 2x times lower than width

        const size_t estimatedEquirectHeight = estimatedEquirectWidth / 2;

        out << "p f2 w" << estimatedEquirectWidth << " h" << estimatedEquirectHeight << " v360  n\"TIFF_m\"" << std::endl;


        // Callwlate vertical fov: 2 x atan( tan(horFov/2) x height / width )
        const float hfovRadians = horizontalFov * kPi / 180.0f;
        const float verticalFov = 2.0f * atanf(tanf(hfovRadians / 2.0f) * height / width) * 180.0f / kPi; //in degrees

        // Figure out start pitch
        const int pitchSteps = int(180.0f / verticalFov * desc.overlap + 0.5f); //round to the nearest int
        const float pitchStep = 180.0f / float(pitchSteps);
        const float startPitch = 90.0f - (pitchStep * 0.5f);

        const float halfVertFov = verticalFov * 0.5f;

        for (int i = 0; i < pitchSteps; i++)
        {
            const float pitch = startPitch - pitchStep * i;
            // we want the bottom of the pitch frame so add or subtract half the vertical fov
            float compPitch = pitch;
            if (compPitch > halfVertFov)
                compPitch -= halfVertFov;
            else if (compPitch < -halfVertFov)
                compPitch += halfVertFov;

            // Figure out the cirlwmference ratio based on current pitch
            const float radius = cosf(compPitch * kPi / 180.0f);
            const float cirlwmference = 2.0f * kPi * radius * 180 / kPi; //the PIs cancel out of course, just there for clarity

            int yawSteps = int(cirlwmference / horizontalFov * desc.overlap + 0.5f); //round to the nearest int
            if (height / width > 1.0f)
                yawSteps += 1;
            const float yawStep = 360.0f / float(yawSteps);
            const float startYaw = (cirlwmference - yawStep) * 0.5f;

            for (int j = 0; j < yawSteps; j++)
            {
                const float yaw = startYaw - float(j) * yawStep;
                // Gradually reduce eye separation when the pitch is close to 90 (up or down)
                float pitchFalloff = (90.0f - std::abs(pitch)) / 90.0f;
                pitchFalloff = 1.0f - pitchFalloff;
                //pitchFalloff *= pitchFalloff;// * pitchFalloff;
                pitchFalloff = powf(pitchFalloff, 1.5f);
                pitchFalloff = 1.0f - pitchFalloff;

                const float minFalloff = 0.2f;
                pitchFalloff = pitchFalloff * (1.0f - minFalloff) + minFalloff;

                sequence.push_back(
                    std::make_tuple(
                        Position(0.0f, eyeOffset * pitchFalloff, 0.0f),
                        Angles(0.0f, pitch * kPi / 180.0f, yaw * kPi / 180.0f),
                        Projection(0.0f, 0.0f),
                        horizontalFov));

                
                sequenceTileUVs.push_back(TileUV(0.0f, 0.0f, 1.0f, 1.0f));

                const std::string name = SphericalTileName + to_string(findex++) + suffix;

                out << "o f0 w" << desc.bmpWidth << " h" << desc.bmpHeight
                    << " n\"" << name << "\" r0 p" << pitch << " y" << yaw << " v" << horizontalFov << std::endl;

                sequenceNames.push_back(name.c_str());
            }
        }

        return Error::kSuccess;
    }

    // Given the two pair of sequences for both eyes, interleave them into the final sequence
    darkroom::Error interleaveSequences(std::vector<darkroom::CameraSpec>& outputCameras, std::vector<darkroom::TileUV>& outputTileUVs, std::vector<std::string>& outputNames,
        std::vector<darkroom::CameraSpec>& inputCamerasLeft, std::vector<darkroom::TileUV>& inputTileUVsLeft, std::vector<std::string>& inputNamesLeft,
        std::vector<darkroom::CameraSpec>& inputCamerasRight, std::vector<darkroom::TileUV>& inputTileUVsRight, std::vector<std::string>& inputNamesRight)
    {
        using darkroom::Error;

        if (inputCamerasLeft.size() != inputCamerasRight.size() ||
            inputTileUVsLeft.size() != inputTileUVsRight.size() ||
            inputCamerasLeft.size() != inputNamesLeft.size())
            return Error::kIlwalidArgument;

        for (size_t i = 0; i < inputCamerasLeft.size(); ++i)
        {
            outputCameras.push_back(inputCamerasLeft[i]);
            outputCameras.push_back(inputCamerasRight[i]);
            outputTileUVs.push_back(inputTileUVsLeft[i]);
            outputTileUVs.push_back(inputTileUVsRight[i]);
            outputNames.push_back(inputNamesLeft[i]);
            outputNames.push_back(inputNamesRight[i]);
        }

        return Error::kSuccess;
    }
}

namespace darkroom
{
    double CameraDirector::estimateTileHorizontalFovSpherical(const uint32_t panoramaWidth, const uint32_t tileWidth)
    {
        return 360.0 * double(tileWidth) / double(panoramaWidth);
    }

    size_t CameraDirector::estimateSphericalPanoramaWidth(const double horizontalFov, const uint32_t tileWidth)
    {
        return uint32_t((360.0 / horizontalFov) * tileWidth) & ~1u;
    }


    bool CameraDirector::validatePathWritable(const ShotDescription& desc) const
    {
        const auto validatePathWithProbe = [](const std::wstring& path)
        {
            const std::wstring probeFile = path + L".probe";
            HANDLE hFile = CreateFile(probeFile.c_str(), GENERIC_WRITE, 0, NULL, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, NULL);
            if (hFile != ILWALID_HANDLE_VALUE)
            {
                CloseHandle(hFile);
                DeleteFile(probeFile.c_str());
                return true;
            }
            else
                return false;
        };

        if (!desc.path.empty())
            if (!validatePathWithProbe(std::wstring(desc.path)))
                return false;

        if (!desc.targetPath.empty())
            if (!validatePathWithProbe(std::wstring(desc.targetPath)))
                return false;

        return true;
    }

    bool CameraDirector::validateFreeSpace(const ShotDescription& desc) const
    {
        // Just comparing free space on drive and required space for captures + output is not quite enough:
        // we don't want to leave several bytes free before we delete temporary tiles as this might lead to 
        // a bad behaviour. So actually we require MINIMUM_FREE_SPACE_AFTER_CAPTURE bytes of free space more.
        const auto estimates = estimateCaptureTask(desc);
        ULARGE_INTEGER freeBytes, totalBytes, totalFreeBytes;
        GetDiskFreeSpaceEx(desc.path.c_str(), &freeBytes, &totalBytes, &totalFreeBytes);
        if (estimates.inputDatasetSizeTotalInBytes + estimates.outputSizeInBytes + MinimumFreeSpaceAfterCapture < freeBytes.QuadPart )
            return true;

        return false;
    }
        
    bool CameraDirector::validateShotDescription(const ShotDescription& desc)
    {
            if (desc.type != ShotDescription::EShotType::REGULAR &&
            desc.type != ShotDescription::EShotType::SPHERICAL_MONO_PANORAMA &&
            desc.type != ShotDescription::EShotType::SPHERICAL_STEREO_PANORAMA &&
            desc.type != ShotDescription::EShotType::HIGHRES &&
            desc.type != ShotDescription::EShotType::STEREO_REGULAR)
            return false;

        if (desc.type == ShotDescription::EShotType::HIGHRES &&
            (!in_range(desc.horizontalFov, s_minHighresHorizontalFov, s_maxHighresHorizontalFov) ||
            !in_range(desc.highresMultiplier, s_minHighresMultiplier, s_maxHighresMultiplier)))
            return false;

        if (desc.type == ShotDescription::EShotType::STEREO_REGULAR &&
            desc.eyeSeparation <= 0.0f)
            return false;

        // horizontalFov less than 25 degrees will cause capturing more than 300 shots, which will take more than 10 seconds
        // at a generous 30 frames per second capture rate
        // 140 degrees on the other hand is quite lowres, so that requires just 6 shots and is very fast
        if ((desc.type == ShotDescription::EShotType::SPHERICAL_MONO_PANORAMA || desc.type == ShotDescription::EShotType::SPHERICAL_STEREO_PANORAMA) &&
            (!in_range(desc.overlap, s_minSphericalTilesOverlap, s_maxSphericalTilesOverlap) ||	!(desc.horizontalFov >= s_minSphericalTilesHorizontalFov)))
            return false;

        return true;
    }

    CaptureTaskEstimates CameraDirector::estimateCaptureTask(const ShotDescription& desc)
    {
        CaptureTaskEstimates est = { 0 };
        if (validateShotDescription(desc))
        {
            if (desc.type == ShotDescription::EShotType::HIGHRES)
            {
                est.inputDatasetFrameCount = (2 * desc.highresMultiplier - 1) * (2 * desc.highresMultiplier - 1);
                if (desc.produceRegularImage)
                    est.inputDatasetFrameCount += 1;
                if (desc.generateThumbnail)
                    est.inputDatasetFrameCount += 1;

                est.inputDatasetFrameSizeInBytes = desc.bmpWidth * desc.bmpHeight * 3;
                est.inputDatasetSizeTotalInBytes = est.inputDatasetFrameCount * est.inputDatasetFrameSizeInBytes;
                est.outputResolutionX = desc.bmpWidth * desc.highresMultiplier;
                est.outputResolutionY = desc.bmpHeight * desc.highresMultiplier;
                est.outputMPixels = est.outputResolutionX * est.outputResolutionY;
                est.outputSizeInBytes = est.outputMPixels * 3;
                est.stitcherMemoryRequirementsInBytes = est.inputDatasetSizeTotalInBytes + est.outputSizeInBytes;
            }
            else if (desc.type == ShotDescription::EShotType::SPHERICAL_MONO_PANORAMA)
            {
                est.inputDatasetFrameCount = numSphericalShots(desc);
                est.inputDatasetFrameSizeInBytes = desc.bmpWidth * desc.bmpHeight * 3;
                est.outputResolutionX = estimateSphericalPanoramaWidth(desc.horizontalFov, desc.bmpWidth);
                est.outputResolutionY = est.outputResolutionX / 2;
                est.outputMPixels = est.outputResolutionX * est.outputResolutionY;
                est.outputSizeInBytes = est.outputMPixels * 3;
                est.stitcherMemoryRequirementsInBytes = est.inputDatasetSizeTotalInBytes + est.outputSizeInBytes;

                if (desc.generateThumbnail)
                {
                    auto descThumbnail = desc;
                    descThumbnail.panoWidth = 2048u;
                    descThumbnail.horizontalFov = float(estimateTileHorizontalFovSpherical(descThumbnail.panoWidth, descThumbnail.bmpWidth));
                    est.inputDatasetFrameCount += numSphericalShots(descThumbnail);
                    est.outputSizeInBytes += 2048u * 1024u;
                }

                est.inputDatasetSizeTotalInBytes = est.inputDatasetFrameCount * est.inputDatasetFrameSizeInBytes;
            }
            else if (desc.type == ShotDescription::EShotType::SPHERICAL_STEREO_PANORAMA)
            {
                auto newDesc = desc;
                newDesc.type = ShotDescription::EShotType::SPHERICAL_MONO_PANORAMA;
                newDesc.generateThumbnail = false;
                est = estimateCaptureTask(newDesc);
                est.inputDatasetFrameCount *= 2;
                est.inputDatasetFrameSizeInBytes *= 2;
                est.stitcherMemoryRequirementsInBytes = est.inputDatasetSizeTotalInBytes + est.outputSizeInBytes;
                est.outputResolutionY *= 2;
                est.outputMPixels = est.outputResolutionX * est.outputResolutionY;
                est.outputSizeInBytes = est.outputMPixels * 3;
                if (desc.generateThumbnail)
                {
                    auto descThumbnail = desc;
                    descThumbnail.panoWidth = 2048u;
                    descThumbnail.horizontalFov = float(estimateTileHorizontalFovSpherical(descThumbnail.panoWidth, descThumbnail.bmpWidth));
                    est.inputDatasetFrameCount += numSphericalShots(descThumbnail);
                    est.outputSizeInBytes += 2048u * 1024u;
                }
                est.inputDatasetSizeTotalInBytes = est.inputDatasetFrameCount * est.inputDatasetFrameSizeInBytes;
            }
            else if (desc.type == ShotDescription::EShotType::STEREO_REGULAR)
            {
                est.inputDatasetFrameCount = 2;
                est.inputDatasetFrameSizeInBytes = desc.bmpWidth * desc.bmpHeight * 3 * 2;
                est.inputDatasetSizeTotalInBytes = est.inputDatasetFrameCount * est.inputDatasetFrameSizeInBytes;
                est.outputResolutionX = desc.bmpWidth * 2;
                est.outputResolutionY = desc.bmpHeight;
                est.outputMPixels = est.outputResolutionX * est.outputResolutionY * 2;
                if (desc.generateThumbnail)
                {
                    est.inputDatasetFrameCount += 1;
                    est.outputMPixels += est.outputResolutionX * est.outputResolutionY;
                }
                est.outputSizeInBytes = est.outputMPixels * 3;
                est.stitcherMemoryRequirementsInBytes = 0;
            }
            else if (desc.type == ShotDescription::EShotType::REGULAR)
            {
                est.inputDatasetFrameCount = 1;
                est.inputDatasetFrameSizeInBytes = desc.bmpWidth * desc.bmpHeight * 3;
                est.inputDatasetSizeTotalInBytes = est.inputDatasetFrameCount * est.inputDatasetFrameSizeInBytes;
                est.outputResolutionX = desc.bmpWidth;
                est.outputResolutionY = desc.bmpHeight;
                est.outputMPixels = est.outputResolutionX * est.outputResolutionY;
                est.outputSizeInBytes = est.outputMPixels * 3;
                est.stitcherMemoryRequirementsInBytes = 0;
            }
        }
        return est;
    }

    // This function pushes series of camera transforms and shot names into m_sequence and m_sequenceNames.
    // This function should generate camera transforms in the current camera frame, not world coordinates or angles
    Error CameraDirector::startCaptureTask(const ShotDescription& desc)
    {
        if (!validateShotDescription(desc))
            return Error::kIlwalidArgument;

        if (!validateFreeSpace(desc))
            return Error::kNotEnoughFreeSpace;

        if (!validatePathWritable(desc))
            return Error::kTargetPathNotWriteable;

        m_sequencePath = std::wstring(desc.path);

        if (desc.type == ShotDescription::EShotType::REGULAR)
        {
            // no need for director for that, but it's easy to support it
            m_sequence.push_back(std::make_tuple(Position(0.0f, 0.0f, 0.0f),
                Angles(0.0f, 0.0f, 0.0f),
                Projection(0.0f, 0.0f),
                0.0f));
            m_sequenceTileUVs.push_back(TileUV(0.0f, 0.0f, 1.0f, 1.0f));
            m_sequenceNames.push_back("regular.bmp");
        }
        else if (desc.type == ShotDescription::EShotType::SPHERICAL_MONO_PANORAMA)
        {
            const Error retcode = generateSphericalShots(m_sequence, m_sequenceTileUVs, m_sequenceNames, desc, "captures.txt", "", 0.0f);
            if (retcode != Error::kSuccess)
                return retcode;

            if (desc.generateThumbnail)
            {
                auto descThumbnail = desc;
                descThumbnail.panoWidth = 2048u;
                descThumbnail.horizontalFov = float(estimateTileHorizontalFovSpherical(descThumbnail.panoWidth, descThumbnail.bmpWidth));
                const Error retcode = generateSphericalShots(m_sequence, m_sequenceTileUVs, m_sequenceNames, descThumbnail, "capturesThumbnail.txt", darkroom::ThumbnailName.c_str(), 0.0f);
                if (retcode != Error::kSuccess)
                    return retcode;
            }
        }
        else if (desc.type == ShotDescription::EShotType::SPHERICAL_STEREO_PANORAMA)
        {
            std::vector<CameraSpec> sequenceLeft, sequenceRight;
            std::vector<TileUV> sequenceTileUVsLeft, sequenceTileUVsRight;
            std::vector<std::string> sequenceNamesLeft, sequenceNamesRight;
            Error retcode = generateSphericalShots(sequenceLeft, sequenceTileUVsLeft, sequenceNamesLeft, desc, "capturesL.txt", "L", -desc.eyeSeparation * 0.5f);
            if (retcode != Error::kSuccess)
                return retcode;

            retcode = generateSphericalShots(sequenceRight, sequenceTileUVsRight, sequenceNamesRight, desc, "capturesR.txt", "R", desc.eyeSeparation * 0.5f);
            if (retcode != Error::kSuccess)
                return retcode;

            retcode = interleaveSequences(m_sequence, m_sequenceTileUVs, m_sequenceNames, sequenceLeft, sequenceTileUVsLeft, sequenceNamesLeft, sequenceRight, sequenceTileUVsRight, sequenceNamesRight);
            if (retcode != Error::kSuccess)
                return retcode;

            if (desc.generateThumbnail)
            {
                auto descThumbnail = desc;
                descThumbnail.panoWidth = 2048u;
                descThumbnail.horizontalFov = float(estimateTileHorizontalFovSpherical(descThumbnail.panoWidth, descThumbnail.bmpWidth));
                const Error retcode = generateSphericalShots(m_sequence, m_sequenceTileUVs, m_sequenceNames, descThumbnail, "capturesThumbnail.txt", darkroom::ThumbnailName.c_str(), 0.0f);
                if (retcode != Error::kSuccess)
                    return retcode;
            }
        }
        else if (desc.type == ShotDescription::EShotType::STEREO_REGULAR)
        {
            const float offset = desc.eyeSeparation * 0.5f;
            m_sequence.push_back(std::make_tuple(Position(0.0f, -offset, 0.0f), Angles(0.0f, 0.0f, 0.0f), Projection(0.0f, 0.0f), 0.0f));
            m_sequenceTileUVs.push_back(TileUV(0.0f, 0.0f, 1.0f, 1.0f));
            m_sequenceNames.push_back(RegularStereoTileName + "L");
            m_sequence.push_back(std::make_tuple(Position(0.0f, offset, 0.0f), Angles(0.0f, 0.0f, 0.0f), Projection(0.0f, 0.0f), 0.0f));
            m_sequenceTileUVs.push_back(TileUV(0.0f, 0.0f, 1.0f, 1.0f));
            m_sequenceNames.push_back(RegularStereoTileName + "R");
            if (desc.generateThumbnail)
            {
                m_sequence.push_back(std::make_tuple(Position(0.0f, 0.0f, 0.0f), Angles(0.0f, 0.0f, 0.0f), Projection(0.0f, 0.0f), 0.0f));
                m_sequenceTileUVs.push_back(TileUV(0.0f, 0.0f, 1.0f, 1.0f));
                m_sequenceNames.push_back(darkroom::ThumbnailName);
            }
        }
        else if (desc.type == ShotDescription::EShotType::HIGHRES)
        {
            const auto shotCount = 2 * desc.highresMultiplier - 1u;
            const auto uvInterval = 1.0f / (2.0f * desc.highresMultiplier);
            const float fov = (360.0f / kPi) * atan(tan(kPi * desc.horizontalFov / 360.0f) / float(desc.highresMultiplier));

            if (desc.produceRegularImage)
            {
                m_sequence.push_back(std::make_tuple(Position(0.0f, 0.0f, 0.0f), Angles(0.0f, 0.0f, 0.0f), Projection(0.0f, 0.0f), desc.horizontalFov));
                m_sequenceTileUVs.push_back(TileUV(0.0f, 0.0f, 1.0f, 1.0f));
                m_sequenceNames.push_back(RegularName);
            }

            if (desc.generateThumbnail)
            {
                m_sequence.push_back(std::make_tuple(Position(0.0f, 0.0f, 0.0f), Angles(0.0f, 0.0f, 0.0f), Projection(0.0f, 0.0f), desc.horizontalFov));
                m_sequenceTileUVs.push_back(TileUV(0.0f, 0.0f, 1.0f, 1.0f));
                m_sequenceNames.push_back(darkroom::ThumbnailName);
            }

            for (auto i = 0u; i < shotCount; ++i)
            {
                for (auto j = 0u; j < shotCount; ++j)
                {
                    const auto xoffset = float(int(j) - int(shotCount / 2));
                    const auto yoffset = float(int((shotCount - i - 1)) - int(shotCount / 2));

                    const float tlV = float(i) * uvInterval;
                    const float tlU = float(j) * uvInterval;
                    const float brU = tlU + 1.0f / float(desc.highresMultiplier);
                    const float brV = tlV + 1.0f / float(desc.highresMultiplier);

                    m_sequence.push_back(std::make_tuple(Position(0.0f, 0.0f, 0.0f), Angles(0.0f, 0.0f, 0.0f), Projection(xoffset, yoffset), fov));
                    m_sequenceTileUVs.push_back(TileUV(tlU, tlV, brU, brV));
                    const std::string name = HighresTileName + to_string(i) + "-" + to_string(j);
                    m_sequenceNames.push_back(name);
                }
            }
        }
        //shuffleCameras();

        return Error::kSuccess;
    }

    void CameraDirector::shuffleCameras()
    {
        if (!m_sequence.empty() && (m_sequence.size() == m_sequenceNames.size()) && (m_sequence.size() == m_sequenceTileUVs.size()))
        {
            struct Entry {
                CameraSpec camera;
                TileUV tileUV;
                std::string name;
            };

            std::vector<Entry> entries;
            for (size_t i = 0u; i < m_sequence.size(); ++i)
                entries.push_back({ m_sequence[i], m_sequenceTileUVs[i], m_sequenceNames[i] });

            m_sequence.clear();
            m_sequenceTileUVs.clear();
            m_sequenceNames.clear();

            std::random_shuffle(entries.begin(), entries.end());

            for (auto& e : entries)
            {
                m_sequence.push_back(e.camera);
                m_sequenceTileUVs.push_back(e.tileUV);
                m_sequenceNames.push_back(e.name);
            }
        }
    }

    bool CameraDirector::abortCaptureTask()
    {
        bool wasNotEmpty = !m_sequence.empty() || !m_sequenceNames.empty();
        m_sequence.clear();
        m_sequenceNames.clear();
        return wasNotEmpty;
    }


    size_t CameraDirector::numSphericalShots(const ShotDescription& desc)
    {
        float horizontalFov = desc.horizontalFov;

        if (horizontalFov > 360.0f)
            horizontalFov = 360.0f;

        if (horizontalFov > s_maxSphericalTilesHorizontalFov)
            horizontalFov = s_maxSphericalTilesHorizontalFov;

        const float hfovRadians = horizontalFov * kPi / 180.0f;
        const float verticalFov = 2.0f * atanf(tanf(hfovRadians / 2.0f) * float(desc.bmpHeight) / float(desc.bmpWidth)) * 180.0f / kPi; //in degrees
        //Figure out start pitch
        const size_t pitchSteps = size_t(180.0f / verticalFov * desc.overlap + 0.5f); //round to the nearest int
        const float pitchStep = 180.0f / float(pitchSteps);
        const float startPitch = 90.0f - (pitchStep * 0.5f);

        const float halfVertFov = verticalFov * 0.5f;
        size_t shotCount = 0;
        for (size_t i = 0; i < pitchSteps; i++)
        {
            const float pitch = startPitch - (float(i) * pitchStep);
            //we want the bottom of the pitch frame so add or subtract half the vertical fov
            float compPitch = pitch;
            if (compPitch > halfVertFov)
                compPitch -= halfVertFov;
            else if (compPitch < -halfVertFov)
                compPitch += halfVertFov;

            //Figure out the cirlwmference ratio based on current pitch
            const float radius = cosf(compPitch * kPi / 180.0f);
            const float cirlwmference = 2.0f * kPi * radius * 180 / kPi; //the PIs cancel out of course, just there for clarity

            shotCount += int(cirlwmference / horizontalFov * desc.overlap + 0.5f); //round to the nearest int
            if (desc.bmpHeight > desc.bmpWidth)
                shotCount += 1;
        }
        return shotCount;
    }

    bool CameraDirector::isCamerasSequenceEmpty() const
    {
        return m_sequence.empty();
    }

    bool CameraDirector::isCameraNamesSequenceEmpty() const
    {
        return m_sequenceNames.empty();
    }

    std::string CameraDirector::nextShotName()
    {
        std::string result;
        if (m_sequenceNames.empty())
            return result;

        result = m_sequenceNames.back();
        m_sequenceNames.pop_back();
        return result;
    }

    bool CameraDirector::nextShotTileUV(float* tlU, float* tlV, float* brU, float* brV)
    {
        using std::get;

        if (!tlU || !tlV || !brU || !brV || m_sequenceTileUVs.empty())
            return false;

        auto tileUV = m_sequenceTileUVs.back();

        *tlU = get<0>(tileUV);
        *tlV = get<1>(tileUV);
        *brU = get<2>(tileUV);
        *brV = get<3>(tileUV);

        m_sequenceTileUVs.pop_back();

        return true;
    }

    bool CameraDirector::nextCamera(float* px, float* py, float* pz,
        float* rx, float* ry, float* rz,
        float* sox, float* soy, float* fov)
    {
        using std::get;

        if (!px || !py || !pz || !rx || !ry || !rz || !fov || m_sequence.empty())
            return false;
        
        auto cam = m_sequence.back();
        *px = get<0>(get<0>(cam));
        *py = get<1>(get<0>(cam));
        *pz = get<2>(get<0>(cam));

        *rx = get<0>(get<1>(cam));
        *ry = get<1>(get<1>(cam));
        *rz = get<2>(get<1>(cam));

        *sox = get<0>(get<2>(cam));
        *soy = get<1>(get<2>(cam));

        *fov = get<3>(cam);
        m_sequence.pop_back();
        return true;
    }

    const wchar_t* CameraDirector::getSequencePath() const
    {
        return m_sequencePath.c_str();
    }
}
