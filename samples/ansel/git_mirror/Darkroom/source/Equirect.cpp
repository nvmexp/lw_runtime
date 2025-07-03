#pragma warning(disable: 4514 4711 4710)
#include "darkroom/Equirect.h"
#include "darkroom/Errors.h"

#define _USE_MATH_DEFINES
#include <math.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <numeric>
#include <set>
#include <tuple>
#include <vector>

#pragma warning(disable:4571) // Informational: catch(...) semantics changed since Visual C++ 7.1; structured exceptions (SEH) are no longer caught
#pragma warning(disable:4265) // class has virtual functions, but destructor is not virtual - lambdas cause that

#include <thread>

namespace darkroom
{
    namespace
    {
        const auto cPi = static_cast<float>(M_PI);
        const auto cPi2 = cPi / 2.0f;

        struct Vec3 {
            Vec3() {}
            Vec3(float theta, float phi, float length)
            {
                x = length * sinf(theta) * cosf(phi);
                y = length * sinf(phi);
                z = length * cosf(theta) * cosf(phi);
            }
            static Vec3 fromXYZ(float x, float y, float z)
            {
                Vec3 result;
                result.x = x;
                result.y = y;
                result.z = z;
                return result;
            }

            Vec3 operator+(const Vec3& other) const { return Vec3::fromXYZ(x + other.x, y + other.y, z + other.z); }
            Vec3 operator-(const Vec3& other) const { return Vec3::fromXYZ(x - other.x, y - other.y, z - other.z); }
            float operator*(const Vec3& other) const { return x * other.x + y * other.y + z * other.z; }
            Vec3 operator*(float m) const { return Vec3::fromXYZ(x * m, y * m, z * m); }
            float magnitude() const { return *this * *this; }

            float x, y, z;
        };

        template<typename T> T clamp(T x, T min, T max) { return x > max ? max : x < min ? min : x; }

        template<typename T>
        void sampleNearest(const T* image, uint32_t width, uint32_t height, float x, float y, float& r, float& g, float& b)
        {
            const uint32_t tx = static_cast<uint32_t>(x * (width - 1)), ty = static_cast<uint32_t>(y * (height - 1));
            const uint32_t pos = 3 * (ty * width + tx);
            b = image[pos];
            g = image[pos + 1];
            r = image[pos + 2];
        }

        template<typename T>
        void sampleBilinear(const T* image, uint32_t width, uint32_t height, float x, float y, float& r, float& g, float& b)
        {
            struct BGR { T b, g, r; };

            const float txFloat = x * (width - 1);
            const float tyFloat = y * (height - 1);
            const uint32_t tx = static_cast<uint32_t>(txFloat), ty = static_cast<uint32_t>(tyFloat);
            const float wx = txFloat - tx;
            const float wy = tyFloat - ty;
            const float iwx = 1.0f - wx;
            const float iwy = 1.0f - wy;

            const float w1 = iwx * iwy;
            const float w2 = wx * iwy;
            const float w3 = iwx * wy;
            const float w4 = wx * wy;

            const uint32_t pos = 3 * (ty * width + tx);
            const uint32_t right = tx < (width - 1) ? 3u : 0u;
            const uint32_t down = ty < (height - 1) ? 3u * width : 0u;
            const BGR* p0 = reinterpret_cast<const BGR*>(image + pos);
            const BGR* p1 = reinterpret_cast<const BGR*>(image + pos + right);
            const BGR* p2 = reinterpret_cast<const BGR*>(image + pos + down);
            const BGR* p3 = reinterpret_cast<const BGR*>(image + pos + down + right);

            b = clamp(p0->b * w1 + p1->b * w2 + p2->b * w3 + p3->b * w4, 0.0f, 255.0f);
            g = clamp(p0->g * w1 + p1->g * w2 + p2->g * w3 + p3->g * w4, 0.0f, 255.0f);
            r = clamp(p0->r * w1 + p1->r * w2 + p2->r * w3 + p3->r * w4, 0.0f, 255.0f);
        }

        template<typename T>
        void sampleBilwbic(const T* image, uint32_t width, uint32_t height, float x, float y, float& r, float& g, float& b)
        {
            struct BGR { T b, g, r; };

            auto LwbicHermite = [](float A, float B, float C, float D, float t)
            {
                float a = -A / 2.0f + (3.0f * B) / 2.0f - (3.0f * C) / 2.0f + D / 2.0f;
                float b = A - (5.0f * B) / 2.0f + 2.0f * C - D / 2.0f;
                float c = -A / 2.0f + C / 2.0f;
                float d = B;

                return a * t * t * t + b * t * t + c * t + d;
            };

            const float txFloat = x * (width - 1);
            const float tyFloat = y * (height - 1);

            const uint32_t tx = static_cast<uint32_t>(txFloat), ty = static_cast<uint32_t>(tyFloat);

            const float wx = txFloat - tx;
            const float wy = tyFloat - ty;

            const uint32_t pos = 3 * (ty * width + tx);

            const int32_t left = tx != 0 ? -3 : 0;
            const int32_t up = ty != 0 ? -3 * int32_t(width) : 0;
            const uint32_t right = tx < (width - 1) ? 3u : 0u;
            const uint32_t right2 = tx < (width - 2) ? 6u : 0u;
            const uint32_t down = ty < (height - 1) ? 3u * width : 0u;
            const uint32_t down2 = ty < (height - 2) ? 6u * width : 0u;

            const BGR* p00 = reinterpret_cast<const BGR*>(image + pos + left + up);
            const BGR* p10 = reinterpret_cast<const BGR*>(image + pos + up);
            const BGR* p20 = reinterpret_cast<const BGR*>(image + pos + right + up);
            const BGR* p30 = reinterpret_cast<const BGR*>(image + pos + right2 + up);

            const BGR* p01 = reinterpret_cast<const BGR*>(image + pos + left);
            const BGR* p11 = reinterpret_cast<const BGR*>(image + pos);
            const BGR* p21 = reinterpret_cast<const BGR*>(image + pos + right);
            const BGR* p31 = reinterpret_cast<const BGR*>(image + pos + right2);

            const BGR* p02 = reinterpret_cast<const BGR*>(image + pos + left + down);
            const BGR* p12 = reinterpret_cast<const BGR*>(image + pos + down);
            const BGR* p22 = reinterpret_cast<const BGR*>(image + pos + right + down);
            const BGR* p32 = reinterpret_cast<const BGR*>(image + pos + right2 + down);

            const BGR* p03 = reinterpret_cast<const BGR*>(image + pos + left + down2);
            const BGR* p13 = reinterpret_cast<const BGR*>(image + pos + down2);
            const BGR* p23 = reinterpret_cast<const BGR*>(image + pos + right + down2);
            const BGR* p33 = reinterpret_cast<const BGR*>(image + pos + right2 + down2);

            // blue
            {
                const float row0 = LwbicHermite(p00->b, p10->b, p20->b, p30->b, wx);
                const float row1 = LwbicHermite(p01->b, p11->b, p21->b, p31->b, wx);
                const float row2 = LwbicHermite(p02->b, p12->b, p22->b, p32->b, wx);
                const float row3 = LwbicHermite(p03->b, p13->b, p23->b, p33->b, wx);
                b = clamp(LwbicHermite(row0, row1, row2, row3, wy), 0.0f, 255.0f);
            }
            // green
            {
                const float row0 = LwbicHermite(p00->g, p10->g, p20->g, p30->g, wx);
                const float row1 = LwbicHermite(p01->g, p11->g, p21->g, p31->g, wx);
                const float row2 = LwbicHermite(p02->g, p12->g, p22->g, p32->g, wx);
                const float row3 = LwbicHermite(p03->g, p13->g, p23->g, p33->g, wx);
                g = clamp(LwbicHermite(row0, row1, row2, row3, wy), 0.0f, 255.0f);
            }
            // red
            {
                const float row0 = LwbicHermite(p00->r, p10->r, p20->r, p30->r, wx);
                const float row1 = LwbicHermite(p01->r, p11->r, p21->r, p31->r, wx);
                const float row2 = LwbicHermite(p02->r, p12->r, p22->r, p32->r, wx);
                const float row3 = LwbicHermite(p03->r, p13->r, p23->r, p33->r, wx);
                r = clamp(LwbicHermite(row0, row1, row2, row3, wy), 0.0f, 255.0f);
            }
        }

        template<typename T>
        void sampleLanczos(const T* image, uint32_t width, uint32_t height, float x, float y, float& r, float& g, float& b)
        {
            const uint32_t lanczosSupport = 3;
            auto lanczosKernel = [](float x, uint32_t a)
            {
                if (x > -std::numeric_limits<float>::epsilon() && x < std::numeric_limits<float>::epsilon())
                    return 1.0f;
                else if (x >= a || x <= -static_cast<int32_t>(a))
                    return 0.0f;
                else
                {
                    const float pix = static_cast<float>(M_PI * x);
                    return static_cast<float>((a * sinf(pix) * sin(pix / a)) / (pix * pix));
                }
            };

            const float txFloat = x * (width - 1);
            const float tyFloat = y * (height - 1);

            const uint32_t tx = static_cast<uint32_t>(txFloat), ty = static_cast<uint32_t>(tyFloat);
            const uint32_t startX = std::max(0u, tx - lanczosSupport + 1), endX = std::min(width - 1, tx + lanczosSupport);
            const uint32_t startY = std::max(0u, ty - lanczosSupport + 1), endY = std::min(height - 1, ty + lanczosSupport);

            float rt[2 * lanczosSupport] = { 0.0f },
                gt[2 * lanczosSupport] = { 0.0f },
                bt[2 * lanczosSupport] = { 0.0f },
                filterSumX = 0.0f,
                filterSumY = 0.0f,
                lanczosValuesX[2 * lanczosSupport] = { 0.0f };

            for (auto i = startX; i <= endX; ++i)
            {
                const auto lanczosValue = lanczosKernel(txFloat - i, lanczosSupport);
                lanczosValuesX[i - startX] = lanczosValue;
                filterSumX += lanczosValue;
            }

            filterSumX = 1.0f / filterSumX;

            for (auto j = startY; j <= endY; ++j)
            {
                for (auto i = startX; i <= endX; ++i)
                {
                    const auto lanczosValue = lanczosValuesX[i - startX];
                    const uint32_t pos = 3 * (j * width + i);
                    bt[j - startY] += image[pos] * lanczosValue;
                    gt[j - startY] += image[pos + 1] * lanczosValue;
                    rt[j - startY] += image[pos + 2] * lanczosValue;
                }
            }

            float rf, gf, bf;
            rf = gf = bf = 0.0f;

            for (auto i = startY; i <= endY; ++i)
            {
                const auto lanczosValue = lanczosKernel(tyFloat - i, lanczosSupport);
                filterSumY += lanczosValue;
                bf += bt[i - startY] * lanczosValue * filterSumX;
                gf += gt[i - startY] * lanczosValue * filterSumX;
                rf += rt[i - startY] * lanczosValue * filterSumX;
            }
            for (auto i = 0; i < 2 * lanczosSupport; ++i)
            {
                bf /= filterSumY;
                gf /= filterSumY;
                rf /= filterSumY;
            }

            b = clamp(bf, 0.0f, 255.0f);
            g = clamp(gf, 0.0f, 255.0f);
            r = clamp(rf, 0.0f, 255.0f);
        }

        class ThreadBarrier
        {
        public:
            ThreadBarrier(unsigned int c) :
                m_threshold(c),
                m_count(c),
                m_generation(0) {
            }

            void wait()
            {
                const unsigned int gen = m_generation;
                std::unique_lock<std::mutex> lock(m_mutex);
                if (--m_count == 0)
                {
                    m_generation += 1;
                    m_count = m_threshold;
                    m_cv.notify_all();
                }
                else
                    m_cv.wait(lock, [this, gen] { return gen != m_generation; });
            }
        private:
            std::mutex m_mutex;
            std::condition_variable m_cv;
            unsigned int m_threshold;
            unsigned int m_count;
            unsigned int m_generation;
        };

    }

    template<typename T>
    Error sphericalEquirect(const T* const * const tiles,
        LoadTilesEquirectCallback loadTilesCallback,
        UnloadTilesEquirectCallback unloadTilesCallback,
        const TileInfo* tileInfo,
        const unsigned int tileCount,
        const unsigned int w, const unsigned int h,
        T* const result,
        const unsigned int wr, const unsigned int hr,
        const unsigned int threadCount)
    {
        if (tileInfo == nullptr) return Error::kIlwalidArgument;
        if (tiles == nullptr) return Error::kIlwalidArgument;
        if (tileCount == 0) return Error::kIlwalidArgument;
        if (w == 0) return Error::kIlwalidArgument;
        if (h == 0) return Error::kIlwalidArgument;
        if (result == nullptr) return Error::kIlwalidArgument;
        if (wr == 0) return Error::kIlwalidArgument;
        if (hr == 0) return Error::kIlwalidArgument;

        using std::get;

        struct TileGeometry
        {
            Vec3 OA;
            Vec3 AD;
            Vec3 AB;
            Vec3 tileNormal;
            float d;
            size_t tileNo;
            float recipWidth, recipHeight;
        };

        struct PitchRangeIndex 
        { 
            unsigned int start, end; 
            bool operator==(const PitchRangeIndex& other) const
            {
                return start == other.start && end == other.end;
            }
        };

        std::vector<TileGeometry> tileGeometry;
        std::vector<PitchRangeIndex> tilePitchRange(hr, { tileCount + 1,0 });

        for (auto i = 0u; i < tileCount; ++i)
        {
            const float hfov = tileInfo[i].horizontalFov;
            const float vfov = 2.0f * atanf(tanf(hfov / 2.0f) * h / w);
            const float hfov2 = hfov / 2.0f;
            const float vfov2 = vfov / 2.0f;
            const float tans = tanf(vfov2) * tanf(vfov2) / tanf(hfov2) * tanf(hfov2);
            const float s = sqrtf((1.0f + tans) / (1.0f + tans + tanf(hfov2) * tanf(hfov2)));
            const float OC = cosf(vfov2) * s;
            const float yaw = tileInfo[i].yaw, pitch = tileInfo[i].pitch;
            const float topTilePhi = asinf(sinf(pitch + vfov2) * s);
            const float bottomTilePhi = asinf(sinf(pitch - vfov2) * s);
            const float topTileTheta = asinf(clamp(tanf(hfov2) * cosf(vfov2) * s / cosf(topTilePhi), -1.0f, 1.0f));
            const float bottomTileTheta = asinf(clamp(tanf(hfov2) * cosf(vfov2) * s / cosf(bottomTilePhi), -1.0f, 1.0f));

            const float bottomAngle = std::min(bottomTilePhi, pitch - vfov2);
            const float topAngle = std::max(topTilePhi, pitch + vfov2);
            const auto startf = (clamp(bottomAngle / cPi2, -1.0f, 1.0f) + 1.0f) * 0.5f;
            const auto endf = (clamp((topAngle) / cPi2, -1.0f, 1.0f) + 1.0f) * 0.5f;
            const size_t start = size_t(hr * startf);
            const size_t end = size_t(hr * endf);

            for (auto k = start; k < end; ++k)
            {
                tilePitchRange[k].start = std::min(tilePitchRange[k].start, i);
                tilePitchRange[k].end = std::max(tilePitchRange[k].end, i);
            }

            TileGeometry info;
            info.tileNormal = Vec3(yaw, pitch, 1.0f);
            if (pitch + vfov2 >= cPi2)
            {
                info.OA = Vec3(yaw + topTileTheta - cPi, topTilePhi, 1.0f);
                const auto OB = Vec3(yaw - topTileTheta + cPi, topTilePhi, 1.0f);
                const auto OD = Vec3(yaw - bottomTileTheta, bottomTilePhi, 1.0f);
                info.AD = OD - info.OA;
                info.AB = OB - info.OA;
            }
            else
            {
                info.OA = Vec3(yaw - topTileTheta, topTilePhi, 1.0f);
                const auto OB = Vec3(yaw + topTileTheta, topTilePhi, 1.0f);
                Vec3 OD;
                if (pitch - vfov2 <= -cPi2) OD = Vec3(yaw + bottomTileTheta - cPi, bottomTilePhi, 1.0f);
                else OD = Vec3(yaw - bottomTileTheta, bottomTilePhi, 1.0f);
                info.AD = OD - info.OA;
                info.AB = OB - info.OA;
            }
            const Vec3 tileCenter = info.tileNormal * OC;
            info.d = -1.0f / (info.tileNormal * tileCenter);
            info.tileNo = i;
            info.recipWidth = 1.0f / info.AB.magnitude();
            info.recipHeight = 1.0f / info.AD.magnitude();
            tileGeometry.push_back(info);
        }

        std::vector<PitchRangeIndex> tilePitchRangeCondensed(hr);
        const auto endIt = std::unique_copy(tilePitchRange.cbegin(), tilePitchRange.cend(), tilePitchRangeCondensed.begin());
        tilePitchRangeCondensed.resize(size_t(std::distance(tilePitchRangeCondensed.begin(), endIt)));

        struct StreamingInfo
        {
            int64_t row;
            std::vector<size_t> loadList, unloadList;
        };

        std::vector<StreamingInfo> streamingInfo;

        std::set<uint32_t> lastTilesSet;
        for (const auto& tileStartEnd : tilePitchRangeCondensed)
        {
            std::vector<uint32_t> lwrrentTilesSequence(tileStartEnd.end - tileStartEnd.start + 1u);
            std::iota(lwrrentTilesSequence.begin(), lwrrentTilesSequence.end(), tileStartEnd.start);
            std::set<uint32_t> lwrrentTilesSet(lwrrentTilesSequence.cbegin(), lwrrentTilesSequence.cend());
            const size_t maxSize = std::max(lastTilesSet.size(), lwrrentTilesSet.size());
            std::vector<size_t> loadList(maxSize), unloadList(maxSize);
            const auto loadListEnd = std::set_difference(lwrrentTilesSet.cbegin(), lwrrentTilesSet.cend(),
                lastTilesSet.cbegin(), lastTilesSet.cend(),
                loadList.begin());
            const auto unloadListEnd = std::set_difference(lastTilesSet.cbegin(), lastTilesSet.cend(),
                lwrrentTilesSet.cbegin(), lwrrentTilesSet.cend(),
                unloadList.begin());
            loadList.resize(size_t(std::distance(loadList.begin(), loadListEnd)));
            unloadList.resize(size_t(std::distance(unloadList.begin(), unloadListEnd)));
            lastTilesSet = lwrrentTilesSet;
            streamingInfo.push_back({ std::distance(tilePitchRange.cbegin(), 
                                                    std::find(tilePitchRange.cbegin(), 
                                                                tilePitchRange.cend(), 
                                                                tileStartEnd)), 
                                        loadList, 
                                        unloadList 
            });
        }

        const unsigned int workerCount = threadCount == 0 ? std::thread::hardware_conlwrrency() : threadCount;
        const unsigned int cacheLineSize = 64u; // on architectures we're interested in

        size_t streamingInfoIndex = 0u;

        ThreadBarrier loadUnloadBarrierStart(workerCount), loadUnloadBarrierEnd(workerCount);

        const size_t syncRowCount = streamingInfo.size();

        auto threadedMap = [&](unsigned int offset)
        {
            for (unsigned int i = 0u; i < hr; ++i)
            {
                // sync all threads on the next sync row and do tiles loading/unloading
                if (streamingInfoIndex < syncRowCount && i == streamingInfo[streamingInfoIndex].row)
                {
                    loadUnloadBarrierStart.wait();

                    // first worker thread manages tiles loading/unloading
                    if (offset == 0u)
                    {
                        const auto loadList = streamingInfo[streamingInfoIndex].loadList;
                        const auto unloadList = streamingInfo[streamingInfoIndex].unloadList;
                        if (unloadTilesCallback && !unloadList.empty())
                            unloadTilesCallback(unloadList);
                        if (loadTilesCallback && !loadList.empty())
                            loadTilesCallback(loadList);

                        streamingInfoIndex += 1u;
                    }

                    loadUnloadBarrierEnd.wait();
                }

                for (unsigned int j = offset; j < wr; j += workerCount * cacheLineSize)
                {
                    auto threadPixelGroup = cacheLineSize;
                    if (j + threadPixelGroup > wr)
                        threadPixelGroup -= j + threadPixelGroup - wr;
                    for (auto k = 0u; k < threadPixelGroup; ++k)
                    {
                        const auto rowPosition = j + k;
                        const auto x = (2.0f * rowPosition) / (wr - 1) - 1.0f, y = (2.0f * i) / (hr - 1) - 1.0f;
                        const float theta = x * cPi, phi = y * cPi2;
                        const size_t resultPos = 3 * (i * wr + rowPosition);

                        const Vec3 R(theta, phi, 1.0f);
                        float r = 0.0f, g = 0.0f, b = 0.0f, totalWeight = 0.0f;
                        const auto start = tilePitchRange[i].start, end = tilePitchRange[i].end;
                        for (auto k = start; k <= end; ++k)
                        {
                            const auto& tile = tileGeometry[k];
                            const float tileNormalR = tile.tileNormal * R;
                            const float recipT = -tile.d * tileNormalR;

                            if (recipT > 1.0f)
                            {
                                const float t = 1.0f / recipT;
                                const Vec3 tileIntersection = Vec3::fromXYZ(R.x * t, R.y * t, R.z * t);
                                const Vec3 AT = tileIntersection - tile.OA;

                                const float ADProj = AT * tile.AD * tile.recipHeight, ABProj = AT * tile.AB * tile.recipWidth;

                                if (ADProj >= 0.0f && ADProj <= 1.0f)
                                {
                                    if (ABProj >= 0.0f && ABProj <= 1.0f)
                                    {
                                        const float tyFloat = (1.0f - ADProj) * h, txFloat = ABProj * w;
                                        const float txNormalized = fabsf((2.0f * txFloat) / w - 1.0f);
                                        const float tyNormalized = fabsf((2.0f * tyFloat) / h - 1.0f);
                                        // callwlate equirect weight
                                        float weight = 1.0f - std::max(txNormalized, tyNormalized);
                                        weight *= weight * weight;

                                        weight *= tileInfo[k].blendFactor;
                                        // perform sampling
                                        float rs, gs, bs;
                                        sampleBilinear(tiles[tile.tileNo], w, h, ABProj, 1.0f - ADProj, rs, gs, bs);

                                        b += bs * weight;
                                        g += gs * weight;
                                        r += rs * weight;

                                        totalWeight += weight;
                                    }
                                }
                            }
                        }
                        if (totalWeight > 0.0f)
                        {
                            result[resultPos] = static_cast<T>(b / totalWeight);
                            result[resultPos + 1] = static_cast<T>(g / totalWeight);
                            result[resultPos + 2] = static_cast<T>(r / totalWeight);
                        }
                    }
                }
            }
        };

        std::vector<std::thread> workers;
        for (auto i = 0u; i < workerCount; ++i)
            workers.push_back(std::thread(threadedMap, i * cacheLineSize));

        for (auto& w : workers)
            w.join();

        return Error::kSuccess;
    }

    template Error sphericalEquirect<unsigned char>(const unsigned char* const * const tiles,
        LoadTilesEquirectCallback loadTilesCallback,
        UnloadTilesEquirectCallback unloadTilesCallback,
        const TileInfo* tileInfo,
        const unsigned int tileCount,
        const unsigned int tileWidth, const unsigned int tileHeight,
        unsigned char* const result,
        const unsigned int resultWidth, const unsigned int resultHeight,
        const unsigned int threadCount);

    template Error sphericalEquirect<float>(const float* const * const tiles,
        LoadTilesEquirectCallback loadTilesCallback,
        UnloadTilesEquirectCallback unloadTilesCallback,
        const TileInfo* tileInfo,
        const unsigned int tileCount,
        const unsigned int tileWidth, const unsigned int tileHeight,
        float* const result,
        const unsigned int resultWidth, const unsigned int resultHeight,
        const unsigned int threadCount);
}
