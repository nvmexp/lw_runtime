#pragma once

#include <stdint.h>
#include "Errors.h"

#include <vector>
#include <tuple>
#include <string>

namespace darkroom
{
    const std::string SphericalTileName = "sphere-";
    const std::string HighresTileName = "highres-";
    const std::string RegularName = "regular";
    const std::string ThumbnailName = "thumbnail";
    const std::string RegularStereoTileName = "shot-";
    const std::wstring gThumbnailSuffix = L" Thumbnail";

    typedef std::tuple<float, float, float> Position; // forward, right, up
    typedef std::tuple<float, float, float> Angles; // roll, pitch, yaw
    typedef std::tuple<float, float> Projection; // screenOffsetX, screenOffsetY
    typedef std::tuple<float, float, float, float> TileUV; // tileTopLeftU, tileTopLeftV, tileBottomRightU, tileBottomRightV
    typedef std::tuple<Position, Angles, Projection, float> CameraSpec;

    const uint64_t MinimumFreeSpaceAfterCapture = 64 * 1024ull * 1024ull;
    const float DefaultRecommendedSphericalOverlap = 1.5f;

    struct ShotDescription
    {
        enum class EShotType
        {
            REGULAR,              // what you see is what you get
            SPHERICAL_MONO_PANORAMA,   // capture full spherical panorama or part of it
            SPHERICAL_STEREO_PANORAMA,      // will create two sets of spherical panorama shots with some eye separation
            HIGHRES,              // use off-axis projection to produce high-resolution screenshot that is stitched using several shots
            STEREO_REGULAR,       // will create 1 set of 2 shots with some eye separation
        };

        EShotType       type;                 // capture type
        float           horizontalFov;        // horizontal FOV in degrees, used to change SPHERICAL_PANORAMA resolution (single tile zoom) or to 
                                              // specify HIGHRES result FOV
        uint32_t        bmpWidth;             //
        uint32_t        bmpHeight;            //
        uint32_t        panoWidth;            // optional: specify panorama width directly, instead of callwlating it. Should be initialized to 0 otherwise
        float           overlap;              // SPHERICAL_PANORAMA allows one to specify camera overlap 
        unsigned int    highresMultiplier;    // limited to 2-6 
        float           eyeSeparation;        //
        float           zFar;                 // this is needed to callwlate the off axis projection x component
        bool            produceRegularImage   // this is to produce a regular image along with the tiles needed for a multipart capture.
                                     = false; // it can be used to transfer frequencies from regular to super-resolution or to produce tonemapped result for HDR captures
        bool            generateThumbnail     // this is to generate JPG/PNG thumbnails for EXR captures
                                    = false;
        std::wstring    path;                 // save path for the image sequence and supplementary data
        std::wstring    targetPath;           // save path for the final image
    };

    struct CaptureTaskEstimates
    {
        uint64_t inputDatasetSizeTotalInBytes;
        uint64_t inputDatasetFrameCount;
        uint64_t inputDatasetFrameSizeInBytes;
        uint64_t stitcherMemoryRequirementsInBytes;
        uint64_t outputResolutionX;
        uint64_t outputResolutionY;
        uint64_t outputMPixels;
        uint64_t outputSizeInBytes;
    };

    /* Class that provides a list of Camera objects for a given capture mode and its settings */
    class CameraDirector
    {
    public:
        /*
        Submit a ShotDescription to fill the CameraDirector object with Camera objects
        The ShotDescription object passed should also pass validateShotDescription without errors

        returns 0 in case of success, otherwise error code (see Errors.h)
        */
        Error startCaptureTask(const ShotDescription& desc);

        /*
        Submit a ShotDescription to fill the CameraDirector object with Camera objects
        The ShotDescription object passed should also pass validateShotDescription without errors
        */
        static CaptureTaskEstimates estimateCaptureTask(const ShotDescription& desc);

        /*
        Given the a tile width and tile fov, estimate panorama width
        Could be useful to fill ShotDescription structure
        */
        static size_t estimateSphericalPanoramaWidth(const double horizontalFov, const uint32_t tileWidth);

        /*
        Given the panorama width and a tile width, estimate horizontal fov for a tile.
        Could be useful to fill ShotDescription structure
        */
        static double estimateTileHorizontalFovSpherical(const uint32_t panoramaWidth, const uint32_t tileWidth);

        /*
        Get all the camera parameters to continue performing the capture task, remove Camera object from queue
        */
        bool nextCamera(float* forward, float* right, float* up,
            float* roll, float* pitch, float* yaw,
            float* screenOffsetX, float* screenOffsetY, float* horizontalFov);
        /*
        Get a shot name for the last image to capture and remove it from the sequence
        */
        std::string nextShotName();
        /*
        Get tile UV coordinates to support image space filters for multi-part shots
        */
        bool nextShotTileUV(float* tlU, float* tlV, float* brU, float* brV);

        /*
        Check if Camera queue is empty
        */
        bool isCamerasSequenceEmpty() const;

        /*
        Check if Shot Names queue is empty
        */
        bool isCameraNamesSequenceEmpty() const;

        /*
        Abort current capture task. Return true in case there was active capture task, otherwise return false.
        */
        bool abortCaptureTask();

        /*
        Get path where shots will be saved to
        */
        const wchar_t* getSequencePath() const;

        /*
        Validates if we have enough of free space
        */
        bool validateFreeSpace(const ShotDescription& desc) const;
    private:
        /*
        Validates a ShotDescription object
        */
        static bool validateShotDescription(const ShotDescription& desc);

        /*
        Validates if we have enough of free space
        */
        bool validatePathWritable(const ShotDescription& desc) const;

        /*
        Return number of shots required to capture a 360 panorama
        */
        static size_t numSphericalShots(const ShotDescription& desc);

        /*
        Shuffle capture plan randomly
        */
        void shuffleCameras();

    protected:
        // Position, Angles, Projection, horizontal FOV
        std::vector<CameraSpec> m_sequence;
        std::vector<TileUV> m_sequenceTileUVs;
        std::vector<std::string> m_sequenceNames;
        std::wstring m_sequencePath;
    };
}
