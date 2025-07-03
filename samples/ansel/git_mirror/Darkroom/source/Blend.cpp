#pragma warning(disable: 4514 4711 4710)
#pragma warning(disable:4571) // Informational: catch(...) semantics changed since Visual C++ 7.1; structured exceptions (SEH) are no longer caught
#pragma warning(disable:4265) // class has virtual functions, but destructor is not virtual - lambdas cause that
#include "darkroom/Blend.h"
#include "darkroom/Errors.h"
#include "darkroom/InternalLimits.h"

#include <vector>
#include <tuple>
#include <cstring>
#include <algorithm>
#include <array>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <thread>

namespace darkroom
{
    namespace
    {
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
    Error blendHighres(const T* const * const tiles,
        const unsigned int tileCount,
        const unsigned int w, const unsigned int ht,
        T* result,
        const unsigned int outputPitch,
        LoadTilesCallback loadTilesCallback,
        UnloadTilesCallback unloadTilesCallback,
        RemapOutputCallback<T> remapCallback,
        ProgressCallback progressCallback,
        bool conserveMemory,
        BufferFormat format,
        const unsigned int threadCount)
    {
        // TODO: process odd height correctly
        const unsigned int h = ht & ~1u; // a bit of cheating for odd tile height (for now)

        if (w == 0 || h == 0)
            return Error::kWrongSize;

        if (tileCount == 0)
            return Error::kWrongSize;

        if (isHighresAllowed(tileCount) != Error::kSuccess)
            return Error::kIlwalidArgumentCount;

        if (conserveMemory && !(loadTilesCallback && unloadTilesCallback && remapCallback))
            return Error::kIlwalidArgument;

        const unsigned int cacheLineSize = 64u; // on architectures we're interested in
        const unsigned int multiplier = static_cast<unsigned int>(sqrtf(float(tileCount)) / 2.0f + 1.0f);
        const unsigned int wr = static_cast<unsigned int>(w * multiplier);
        const unsigned int hr = static_cast<unsigned int>(h * multiplier);
        const unsigned int halfTileHeight = h / 2;
        const float recipH = 1.0f / h;
        const float recipW = 1.0f / w;

        const unsigned int tileRowCount = static_cast<unsigned int>(sqrtf(float(tileCount)));
        const unsigned int tileColCount = tileRowCount;

        std::vector<std::thread> workers;
        const unsigned int workerCount = threadCount == 0 ? std::thread::hardware_conlwrrency() : threadCount;

        ThreadBarrier barrier(workerCount), loadedBarrier(workerCount);

        std::atomic<unsigned int> lwrrentRow;
        std::atomic_init(&lwrrentRow, 0);
        std::mutex waitForNewRowsLoaded;

        std::vector<size_t> tileNumbers;

        if (conserveMemory)
        {
            // load first half-row
            tileNumbers.reserve(tileRowCount);

            for (size_t i = 0; i < tileRowCount / 2 + 1; ++i)
                tileNumbers.push_back(i);

            loadTilesCallback(tileNumbers);
            result = remapCallback(0, 3 * halfTileHeight * wr * sizeof(T), nullptr);
        }

        auto doStrip = [&](unsigned int offset, unsigned int lwrrentHalf)
        {
            for (auto i = 0u; i < hr; ++i)
            {
                const auto start = lwrrentHalf == 0 ? offset : offset + wr / 2;
                const auto end = lwrrentHalf == 0 ? wr / 2 : wr;
                for (auto j = start; j < end; j += workerCount * cacheLineSize)
                {
                    auto threadPixelGroup = cacheLineSize;
                    if (j + threadPixelGroup > end)
                        threadPixelGroup -= j + threadPixelGroup - end;
                    for (auto k = 0u; k < threadPixelGroup; ++k)
                    {
                        const auto rowPosition = j + k;
                        auto resultPos = i * size_t(outputPitch) / sizeof(T) + 3 * rowPosition;
                        if (conserveMemory)
                            resultPos = (i - lwrrentRow * halfTileHeight) * size_t(outputPitch) / sizeof(T) + 3 * rowPosition;
                        size_t row1No = 0;
                        bool single = true;
                        if (i > hr - halfTileHeight)
                            row1No = tileRowCount - 1;
                        else if (i > halfTileHeight)
                        {
                            row1No = 2 * (i - halfTileHeight) / h;
                            single = false;
                        }

                        auto row1Pos = i % h;
                        if (row1No % 2)
                            row1Pos += halfTileHeight;
                        const auto row2No = row1No == tileRowCount - 1 ? row1No : row1No + 1;

                        auto blendHorizontal = [&](size_t rowNo, unsigned int rowPos, T& r, T& g, T& b)
                        {
                            size_t tile1No = 0;
                            bool single = true;
                            if (rowPosition > wr - w / 2)
                                tile1No = tileColCount - 1;
                            else if (rowPosition > w / 2)
                            {
                                tile1No = 2 * (rowPosition - w / 2) / w;
                                single = false;
                            }

                            auto tile1Pos = rowPosition % w;
                            if (tile1No % 2)
                                tile1Pos += w / 2;
                            const auto tile2No = tile1No == tileColCount - 1 ? tile1No : tile1No + 1;

                            const T* tile1r = tiles[rowNo * tileColCount + tile1No] + 3 * (rowPos * w + tile1Pos);
                            if (single)
                            {
                                r = tile1r[0];
                                g = tile1r[1];
                                b = tile1r[2];
                            }
                            else
                            {
                                const auto tile2Pos = tile1Pos - w / 2;
                                const T* tile2r = tiles[rowNo * tileColCount + tile2No] + 3 * (rowPos * w + tile2Pos);
                                const float weight = 1.0f - 2.0f * float(tile2Pos) * recipW;
                                r = static_cast<T>(weight * tile1r[0] + (1.0f - weight) * tile2r[0]);
                                g = static_cast<T>(weight * tile1r[1] + (1.0f - weight) * tile2r[1]);
                                b = static_cast<T>(weight * tile1r[2] + (1.0f - weight) * tile2r[2]);
                            }
                        };

                        if (single)
                        {
                            T r1, g1, b1;
                            blendHorizontal(row1No, row1Pos, r1, g1, b1);
                            result[resultPos + 1] = g1;

                            if (format == BufferFormat::RGB8 || format == BufferFormat::RGB32)
                            {
                                result[resultPos + 0] = r1;
                                result[resultPos + 2] = b1;
                            }
                            else if (format == BufferFormat::BGR8 || format == BufferFormat::BGR32)
                            {
                                result[resultPos + 0] = b1;
                                result[resultPos + 2] = r1;
                            }
                        }
                        else
                        {
                            const auto row2Pos = row1Pos - halfTileHeight;
                            T r1, g1, b1, r2, g2, b2;
                            blendHorizontal(row1No, row1Pos, r1, g1, b1);
                            blendHorizontal(row2No, row2Pos, r2, g2, b2);

                            const float weight = 1.0f - 2.0f * float(row2Pos) * recipH;
                            result[resultPos + 1] = static_cast<T>(weight * g1 + (1.0f - weight) * g2);
                            if (format == BufferFormat::RGB8 || format == BufferFormat::RGB32)
                            {
                                result[resultPos + 0] = static_cast<T>(weight * r1 + (1.0f - weight) * r2);
                                result[resultPos + 2] = static_cast<T>(weight * b1 + (1.0f - weight) * b2);
                            }
                            else if (format == BufferFormat::BGR8 || format == BufferFormat::BGR32)
                            {
                                result[resultPos + 0] = static_cast<T>(weight * b1 + (1.0f - weight) * b2);
                                result[resultPos + 2] = static_cast<T>(weight * r1 + (1.0f - weight) * r2);
                            }
                        }
                    }
                }

                if (conserveMemory)
                {
                    // check if we've finished current row, if yes, wait for other threads
                    if (i == halfTileHeight * (lwrrentRow + 1) - 1)
                    {
                        barrier.wait();

                        if (offset == 0)
                            waitForNewRowsLoaded.lock();

                        barrier.wait();
                        // only 1st worker manages tiles loading/unloading, others just wait in the lock below
                        if (offset == 0)
                        {
                            lwrrentRow += 1;
                            if (lwrrentRow > 1)
                            {
                                tileNumbers.clear();
                                for (size_t i = 0; i < tileRowCount / 2 + 1; ++i)
                                    tileNumbers.push_back(i + (lwrrentRow - 2) * tileRowCount + lwrrentHalf * (tileRowCount / 2));
                                unloadTilesCallback(tileNumbers);
                            }
                            // generate new tile numbers
                            tileNumbers.clear();
                            
                            if (lwrrentRow < tileRowCount)
                            {
                                for (size_t i = 0; i < tileRowCount / 2 + 1; ++i)
                                    tileNumbers.push_back(i + lwrrentRow * tileRowCount + lwrrentHalf * (tileRowCount / 2));
                                // load tiles here
                                std::sort(tileNumbers.begin(), tileNumbers.end());
                                loadTilesCallback(tileNumbers);
                            }

                            if (lwrrentRow <= tileRowCount)
                            {
                                const uint64_t start = (3ull * h * wr / 2) * lwrrentRow * sizeof(T);
                                const uint64_t end = (3ull * h * wr / 2) * (lwrrentRow + 1) * sizeof(T);

                                result = remapCallback(start, end, result);
                            }

                            if (progressCallback)
                                progressCallback(0.5f * float(i) / float(hr) + 0.5f * lwrrentHalf);

                            waitForNewRowsLoaded.unlock();
                        }

                        std::unique_lock<std::mutex> lock(waitForNewRowsLoaded);
                    }
                }
            }
        };

        // first process left half of the final image
        for (auto i = 0u; i < workerCount; ++i)
            workers.push_back(std::thread(doStrip, i * cacheLineSize, 0));
        for (auto& w : workers)
            w.join();

        // report 50% ready
        if (progressCallback)
            progressCallback(0.5f);

        if (conserveMemory)
        {
            lwrrentRow = 0;
            tileNumbers.clear();
            // load first half-row of the second (right) half
            tileNumbers.reserve(tileRowCount);

            for (size_t i = tileRowCount / 2; i < tileRowCount; ++i)
                tileNumbers.push_back(i);
                
            loadTilesCallback(tileNumbers);
            const size_t start = (3 * h * wr / 2) * lwrrentRow * sizeof(T);
            const size_t end = (3 * h * wr / 2) * (lwrrentRow + 1) * sizeof(T);
            result = remapCallback(start, end, result);
        }

        workers.clear();
        // then process right half
        for (auto i = 0u; i < workerCount; ++i)
            workers.push_back(std::thread(doStrip, i * cacheLineSize, 1));
        for (auto& w : workers)
            w.join();

        // report 100% ready
        if (progressCallback)
            progressCallback(1.0f);

        return darkroom::Error::kSuccess;
    }

    template Error blendHighres<unsigned char>(const unsigned char* const * const tiles,
        const unsigned int tileCount,
        const unsigned int tileWidth,
        const unsigned int tileHeight,
        unsigned char* result,
        const unsigned int outputPitch,
        LoadTilesCallback loadTilesCallback,
        UnloadTilesCallback unloadTilesCallback,
        RemapOutputCallback<unsigned char> remapCallback,
        ProgressCallback progressCallback,
        bool conserveMemory,
        BufferFormat format,
        const unsigned int threadCount);

    template Error blendHighres<float>(const float* const * const tiles,
        const unsigned int tileCount,
        const unsigned int tileWidth,
        const unsigned int tileHeight,
        float* result,
        const unsigned int outputPitch,
        LoadTilesCallback loadTilesCallback,
        UnloadTilesCallback unloadTilesCallback,
        RemapOutputCallback<float> remapCallback,
        ProgressCallback progressCallback,
        bool conserveMemory,
        BufferFormat format,
        const unsigned int threadCount);

    Error isHighresAllowed(unsigned int tileCount)
    {
        std::vector<unsigned int> allowedArgc;
        allowedArgc.reserve(s_maxHighresMultiplier - s_minHighresMultiplier);

        for (unsigned int i = s_minHighresMultiplier; i <= s_maxHighresMultiplier; ++i)
        {
            const auto tilesInRow = i * 2u - 1u;
            allowedArgc.push_back(tilesInRow * tilesInRow);
        }

        if (std::find(allowedArgc.begin(), allowedArgc.end(), tileCount) == allowedArgc.end())
            return Error::kIlwalidArgumentCount;

        return Error::kSuccess;
    }
}
