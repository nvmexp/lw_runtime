#define NOMINMAX
#include "darkroom/ImageOperations.h"
#include "darkroom/Errors.h"

#include <algorithm>

#pragma warning(push)
#pragma warning(disable : 4668)
#include <Windows.h>
#pragma warning(pop)
#include <immintrin.h>

namespace
{
    enum class Codepath
    {
        kScalar,
        kSSE2
    };

    template <typename T, std::size_t N = 16>
    class AlignmentAllocator {
    public:
        typedef T value_type;
        typedef std::size_t size_type;
        typedef std::ptrdiff_t difference_type;
        typedef T * pointer;
        typedef const T * const_pointer;
        typedef T & reference;
        typedef const T & const_reference;
    public:
        inline AlignmentAllocator() throw () { }
        template <typename T2>
        inline AlignmentAllocator(const AlignmentAllocator<T2, N> &) throw () { }
        inline ~AlignmentAllocator() throw () { }
        inline pointer adress(reference r) { return &r; }
        inline const_pointer adress(const_reference r) const { return &r; }
        inline pointer allocate(size_type n) { return (pointer)_aligned_malloc(n * sizeof(value_type), N); }
        inline void deallocate(pointer p, size_type) { _aligned_free(p); }
        inline void construct(pointer p, const value_type & wert) { new (p) value_type(wert); }
        inline void destroy(pointer p) 
        { 
            _CRT_UNUSED(p);
            p->~value_type(); 
        }
        inline size_type max_size() const throw () { return size_type(-1) / sizeof(value_type); }

        template <typename T2>
        struct rebind { typedef AlignmentAllocator<T2, N> other; };

        bool operator!=(const AlignmentAllocator<T, N>& other) const { return !(*this == other); }
        bool operator==(const AlignmentAllocator<T, N>& other) const { return true; }
    };
}

namespace darkroom
{
    namespace SSEHelpers
    {
#define _PS_CONST(Name, Val) \
        static const __declspec(align(16)) float _ps_##Name[4] = { (float)Val, (float)Val, (float)Val, (float)Val }
#define _PI32_CONST(Name, Val) \
        static const __declspec(align(16)) int _pi32_##Name[4] = { Val, Val, Val, Val }
#define _PS_CONST_TYPE(Name, Type, Val) \
        static const __declspec(align(16)) Type _ps_##Name[4] = { Val, Val, Val, Val }

        _PS_CONST(1  , 1.0f);
        _PS_CONST(0p5, 0.5f);
        /* the smallest non denormalized float number */
        _PS_CONST_TYPE(min_norm_pos, int, 0x00800000);
        _PS_CONST_TYPE(mant_mask, int, 0x7f800000);
        _PS_CONST_TYPE(ilw_mant_mask, int, ~0x7f800000);

        _PS_CONST_TYPE(sign_mask, int, (int)0x80000000);
        _PS_CONST_TYPE(ilw_sign_mask, int, ~0x80000000);

        _PI32_CONST(1, 1);
        _PI32_CONST(ilw1, ~1);
        _PI32_CONST(2, 2);
        _PI32_CONST(4, 4);
        _PI32_CONST(0x7f, 0x7f);

        _PS_CONST(cephes_SQRTHF, 0.707106781186547524);
        _PS_CONST(cephes_log_p0, 7.0376836292E-2);
        _PS_CONST(cephes_log_p1, - 1.1514610310E-1);
        _PS_CONST(cephes_log_p2, 1.1676998740E-1);
        _PS_CONST(cephes_log_p3, - 1.2420140846E-1);
        _PS_CONST(cephes_log_p4, + 1.4249322787E-1);
        _PS_CONST(cephes_log_p5, - 1.6668057665E-1);
        _PS_CONST(cephes_log_p6, + 2.0000714765E-1);
        _PS_CONST(cephes_log_p7, - 2.4999993993E-1);
        _PS_CONST(cephes_log_p8, + 3.3333331174E-1);
        _PS_CONST(cephes_log_q1, -2.12194440e-4);
        _PS_CONST(cephes_log_q2, 0.693359375);

        typedef __m128 v4sf;
        typedef __m128i v4si;
        v4sf log_ps(v4sf x)
        {
            v4si emm0;
            v4sf one = *(v4sf*)_ps_1;

            v4sf ilwalid_mask = _mm_cmple_ps(x, _mm_setzero_ps());

            x = _mm_max_ps(x, *(v4sf*)_ps_min_norm_pos);  /* cut off denormalized stuff */

            emm0 = _mm_srli_epi32(_mm_castps_si128(x), 23);
            /* keep only the fractional part */
            x = _mm_and_ps(x, *(v4sf*)_ps_ilw_mant_mask);
            x = _mm_or_ps(x, *(v4sf*)_ps_0p5);

            emm0 = _mm_sub_epi32(emm0, *(v4si*)_pi32_0x7f);
            v4sf e = _mm_cvtepi32_ps(emm0);

            e = _mm_add_ps(e, one);

            /* part2: 
            if( x < SQRTHF ) {
            e -= 1;
            x = x + x - 1.0;
            } else { x = x - 1.0; }
            */
            v4sf mask = _mm_cmplt_ps(x, *(v4sf*)_ps_cephes_SQRTHF);
            v4sf tmp = _mm_and_ps(x, mask);
            x = _mm_sub_ps(x, one);
            e = _mm_sub_ps(e, _mm_and_ps(one, mask));
            x = _mm_add_ps(x, tmp);

            v4sf z = _mm_mul_ps(x,x);

            v4sf y = *(v4sf*)_ps_cephes_log_p0;
            y = _mm_mul_ps(y, x);
            y = _mm_add_ps(y, *(v4sf*)_ps_cephes_log_p1);
            y = _mm_mul_ps(y, x);
            y = _mm_add_ps(y, *(v4sf*)_ps_cephes_log_p2);
            y = _mm_mul_ps(y, x);
            y = _mm_add_ps(y, *(v4sf*)_ps_cephes_log_p3);
            y = _mm_mul_ps(y, x);
            y = _mm_add_ps(y, *(v4sf*)_ps_cephes_log_p4);
            y = _mm_mul_ps(y, x);
            y = _mm_add_ps(y, *(v4sf*)_ps_cephes_log_p5);
            y = _mm_mul_ps(y, x);
            y = _mm_add_ps(y, *(v4sf*)_ps_cephes_log_p6);
            y = _mm_mul_ps(y, x);
            y = _mm_add_ps(y, *(v4sf*)_ps_cephes_log_p7);
            y = _mm_mul_ps(y, x);
            y = _mm_add_ps(y, *(v4sf*)_ps_cephes_log_p8);
            y = _mm_mul_ps(y, x);

            y = _mm_mul_ps(y, z);


            tmp = _mm_mul_ps(e, *(v4sf*)_ps_cephes_log_q1);
            y = _mm_add_ps(y, tmp);


            tmp = _mm_mul_ps(z, *(v4sf*)_ps_0p5);
            y = _mm_sub_ps(y, tmp);

            tmp = _mm_mul_ps(e, *(v4sf*)_ps_cephes_log_q2);
            x = _mm_add_ps(x, y);
            x = _mm_add_ps(x, tmp);
            x = _mm_or_ps(x, ilwalid_mask); // negative arg will be NAN
            return x;
        }

        _PS_CONST(exp_hi, 88.3762626647949f);
        _PS_CONST(exp_lo, -88.3762626647949f);

        _PS_CONST(cephes_LOG2EF, 1.44269504088896341);
        _PS_CONST(cephes_exp_C1, 0.693359375);
        _PS_CONST(cephes_exp_C2, -2.12194440e-4);

        _PS_CONST(cephes_exp_p0, 1.9875691500E-4);
        _PS_CONST(cephes_exp_p1, 1.3981999507E-3);
        _PS_CONST(cephes_exp_p2, 8.3334519073E-3);
        _PS_CONST(cephes_exp_p3, 4.1665795894E-2);
        _PS_CONST(cephes_exp_p4, 1.6666665459E-1);
        _PS_CONST(cephes_exp_p5, 5.0000001201E-1);

        v4sf exp_ps(v4sf x)
        {
            v4sf tmp = _mm_setzero_ps(), fx;
            v4si emm0;

            v4sf one = *(v4sf*)_ps_1;

            x = _mm_min_ps(x, *(v4sf*)_ps_exp_hi);
            x = _mm_max_ps(x, *(v4sf*)_ps_exp_lo);

            /* express exp(x) as exp(g + n*log(2)) */
            fx = _mm_mul_ps(x, *(v4sf*)_ps_cephes_LOG2EF);
            fx = _mm_add_ps(fx, *(v4sf*)_ps_0p5);

            /* how to perform a floorf with SSE: just below */
            emm0 = _mm_cvttps_epi32(fx);
            tmp  = _mm_cvtepi32_ps(emm0);

            /* if greater, substract 1 */
            v4sf mask = _mm_cmpgt_ps(tmp, fx);    
            mask = _mm_and_ps(mask, one);
            fx = _mm_sub_ps(tmp, mask);

            tmp = _mm_mul_ps(fx, *(v4sf*)_ps_cephes_exp_C1);
            v4sf z = _mm_mul_ps(fx, *(v4sf*)_ps_cephes_exp_C2);
            x = _mm_sub_ps(x, tmp);
            x = _mm_sub_ps(x, z);

            z = _mm_mul_ps(x,x);

            v4sf y = *(v4sf*)_ps_cephes_exp_p0;
            y = _mm_mul_ps(y, x);
            y = _mm_add_ps(y, *(v4sf*)_ps_cephes_exp_p1);
            y = _mm_mul_ps(y, x);
            y = _mm_add_ps(y, *(v4sf*)_ps_cephes_exp_p2);
            y = _mm_mul_ps(y, x);
            y = _mm_add_ps(y, *(v4sf*)_ps_cephes_exp_p3);
            y = _mm_mul_ps(y, x);
            y = _mm_add_ps(y, *(v4sf*)_ps_cephes_exp_p4);
            y = _mm_mul_ps(y, x);
            y = _mm_add_ps(y, *(v4sf*)_ps_cephes_exp_p5);
            y = _mm_mul_ps(y, z);
            y = _mm_add_ps(y, x);
            y = _mm_add_ps(y, one);

            /* build 2^n */
            emm0 = _mm_cvttps_epi32(fx);
            emm0 = _mm_add_epi32(emm0, *(v4si*)_pi32_0x7f);
            emm0 = _mm_slli_epi32(emm0, 23);
            v4sf pow2n = _mm_castsi128_ps(emm0);

            y = _mm_mul_ps(y, pow2n);
            return y;
        }

#undef _PS_CONST
#undef _PI32_CONST
#undef _PS_CONST_TYPE
    }

    template <typename T>
    Error crop(std::vector<T>& output, const std::vector<T>& input, unsigned int width, unsigned int height, 
            unsigned int x, unsigned int y, unsigned int cropWidth, unsigned int cropHeight)
    {
        if ((x + cropWidth > width) || (y + cropHeight > height))
            return Error::kWrongSize;

        output.clear();
        output.resize(size_t(cropWidth) * size_t(cropHeight) * 3);
        size_t k = 0;
        for (size_t i = y; i < y + cropHeight; ++i)
        {
            for (size_t j = x; j < x + cropWidth; ++j)
            {
                output[k++] = input[3 * (i * width + j) + 0];
                output[k++] = input[3 * (i * width + j) + 1];
                output[k++] = input[3 * (i * width + j) + 2];
            }
        }

        return Error::kSuccess;
    }

    template <typename T>
    Error appendVertically(std::vector<T>& output, const std::vector<T>& input1, const std::vector<T>& input2)
    {
        if (input1.empty() || input2.empty())
            return Error::kImageTooSmall;

        if (input1.size() != input2.size())
            return Error::kImageDimsNotEqual;

        output.clear();
        output.reserve(input1.size() * 2);
        output.insert(output.end(), input1.begin(), input1.end());
        output.insert(output.end(), input2.begin(), input2.end());

        return Error::kSuccess;
    }

    template <typename T>
    Error appendHorizontally(std::vector<T>& output, const std::vector<T>& input1, const std::vector<T>& input2, unsigned int width, unsigned int height)
    {
        if (input1.empty() || input2.empty())
            return Error::kImageTooSmall;

        if (input1.size() != input2.size())
            return Error::kImageDimsNotEqual;

        output.clear();
        output.resize(input1.size() * 2);

        const auto inputPitch = width * 3;
        const auto outputPitch = inputPitch * 2;

        for (auto j = 0u; j < height; ++j)
        {
            const auto outputPos = output.begin() + off_t(outputPitch * j);
            const auto input1Pos = input1.cbegin() + off_t(inputPitch * j);
            const auto input2Pos = input2.cbegin() + off_t(inputPitch * j);
            std::copy(input1Pos, input1Pos + off_t(inputPitch), outputPos);
            std::copy(input2Pos, input2Pos + off_t(inputPitch), outputPos + off_t(inputPitch));
        }

        return Error::kSuccess;
    }

    template <typename T, typename S, typename Q, Codepath codepath>
    Error downscaleAclwmulateHelper(const T* input, S* outputAclwm, Q* outputSum,
        unsigned int inputChannels, unsigned int inputWidth, unsigned int inputHeight,
        unsigned int outputChannels, unsigned int outputWidth, unsigned int outputHeight,
        unsigned int xOffset, unsigned int yOffset, unsigned int windowWidth,
        unsigned int windowHeight) 
    {
        // this code path only supports 3 channel input/output
        if ((inputChannels != outputChannels) ||
            (inputChannels != 3))
            return darkroom::Error::kIlwalidArgument;

        const auto height = std::min(inputHeight, windowHeight);
        const auto width = std::min(inputWidth, windowWidth);

        for (uint32_t j = 0; j < height; ++j)
        {
            const auto ny = outputHeight * (j + yOffset) / inputHeight;
            const auto rowOffset = ny * outputWidth;
            for (uint32_t i = 0; i < width; ++i)
            {
                const auto nx = uint64_t(outputWidth) * (i + xOffset) / inputWidth;
                const auto outputCoords = rowOffset + nx;
                const auto outputCoordsColor = inputChannels * outputCoords;
                const auto inputCoordsColor = inputChannels * (uint64_t(j) * inputWidth + i);
                for (uint32_t c = 0; c < inputChannels; ++c)
                    outputAclwm[size_t(outputCoordsColor + c)] += input[inputCoordsColor + c];
                outputSum[size_t(outputCoords)] += 1;
            }
        }

        return darkroom::Error::kSuccess;
    }

    template <>
    Error downscaleAclwmulateHelper<unsigned char, float, float, Codepath::kSSE2>(
        const unsigned char* input, float* outputAclwm, float* outputSum,
        unsigned int inputChannels, unsigned int inputWidth, unsigned int inputHeight,
        unsigned int outputChannels, unsigned int outputWidth, unsigned int outputHeight,
        unsigned int xOffset, unsigned int yOffset, unsigned int windowWidth,
        unsigned int windowHeight)
    {
        // CPU should support SSE2
        if (!IsProcessorFeaturePresent(PF_XMMI64_INSTRUCTIONS_AVAILABLE))
            return darkroom::Error::kOperationFailed;

        // this code path only supports 3 channel input/output
        // and input pointer should be 16-bytes aligned
        if (((size_t(input) & 0xF) != 0) ||
            (inputChannels != outputChannels) ||
            (inputChannels != 3))
            return darkroom::Error::kIlwalidArgument;

        std::vector<uint32_t, AlignmentAllocator<float>> offsets(16);
        std::vector<uint32_t, AlignmentAllocator<float>> values(48);
        const float widthRatio = float(outputWidth) / float(inputWidth);
        const float heightRatio = float(outputHeight) / float(inputHeight);

        const bool inputWidthIsMultipleOf16 = !(inputWidth & 0xF);
        const auto height = std::min(inputHeight, windowHeight);
        const auto width = std::min(inputWidth, windowWidth);
        for (uint32_t j = 0; j < height; ++j)
        {
            const auto ny = outputHeight * (j + yOffset) / inputHeight;
            const auto rowOffset = ny * outputWidth;
            for (uint32_t i = 0; i < width; i += 16u)
            {
                const auto addr = input + inputChannels * (j * inputWidth + i);
                const __m128i zero = _mm_setzero_si128();
                // unpack 48 bytes worth of pixel data (16 pixels x RGB) unto 48 uint32_t and store it into 'values' array
                const auto pixels_batch_1 = _mm_loadu_si128((__m128i*)addr);
                const __m128i pixels_batch_1_u16lo = _mm_unpacklo_epi8(pixels_batch_1, zero);
                const __m128i pixels_batch_1_u16hi = _mm_unpackhi_epi8(pixels_batch_1, zero);
                const auto pixels_batch_2 = _mm_loadu_si128(((__m128i*)addr) + 1);
                const __m128i pixels_batch_2_u16lo = _mm_unpacklo_epi8(pixels_batch_2, zero);
                const __m128i pixels_batch_2_u16hi = _mm_unpackhi_epi8(pixels_batch_2, zero);
                const auto pixels_batch_3 = _mm_loadu_si128(((__m128i*)addr) + 2);
                const __m128i pixels_batch_3_u16lo = _mm_unpacklo_epi8(pixels_batch_3, zero);
                const __m128i pixels_batch_3_u16hi = _mm_unpackhi_epi8(pixels_batch_3, zero);

                _mm_storeu_si128((__m128i*)values.data(), _mm_unpacklo_epi16(pixels_batch_1_u16lo, zero));
                _mm_storeu_si128(((__m128i*)values.data()) + 1, _mm_unpackhi_epi16(pixels_batch_1_u16lo, zero));
                _mm_storeu_si128(((__m128i*)values.data()) + 2, _mm_unpacklo_epi16(pixels_batch_1_u16hi, zero));
                _mm_storeu_si128(((__m128i*)values.data()) + 3, _mm_unpackhi_epi16(pixels_batch_1_u16hi, zero));
                _mm_storeu_si128(((__m128i*)values.data()) + 4, _mm_unpacklo_epi16(pixels_batch_2_u16lo, zero));
                _mm_storeu_si128(((__m128i*)values.data()) + 5, _mm_unpackhi_epi16(pixels_batch_2_u16lo, zero));
                _mm_storeu_si128(((__m128i*)values.data()) + 6, _mm_unpacklo_epi16(pixels_batch_2_u16hi, zero));
                _mm_storeu_si128(((__m128i*)values.data()) + 7, _mm_unpackhi_epi16(pixels_batch_2_u16hi, zero));
                _mm_storeu_si128(((__m128i*)values.data()) + 8, _mm_unpacklo_epi16(pixels_batch_3_u16lo, zero));
                _mm_storeu_si128(((__m128i*)values.data()) + 9, _mm_unpackhi_epi16(pixels_batch_3_u16lo, zero));
                _mm_storeu_si128(((__m128i*)values.data()) + 10, _mm_unpacklo_epi16(pixels_batch_3_u16hi, zero));
                _mm_storeu_si128(((__m128i*)values.data()) + 11, _mm_unpackhi_epi16(pixels_batch_3_u16hi, zero));

                // callwlate thumbnail pixel offsets for all 16 pixels
                const auto pixel_x = _mm_set1_ps(float(i + xOffset));
                const auto widthRatioV = _mm_set1_ps(widthRatio);
                const auto nx1t = _mm_cvtps_epi32(_mm_mul_ps(widthRatioV, _mm_add_ps(pixel_x, _mm_setr_ps(0.0f, 1.0f, 2.0f, 3.0f))));
                const auto nx2t = _mm_cvtps_epi32(_mm_mul_ps(widthRatioV, _mm_add_ps(pixel_x, _mm_setr_ps(4.0f, 5.0f, 6.0f, 7.0f))));
                const auto nx3t = _mm_cvtps_epi32(_mm_mul_ps(widthRatioV, _mm_add_ps(pixel_x, _mm_setr_ps(8.0f, 9.0f, 10.0f, 11.0f))));
                const auto nx4t = _mm_cvtps_epi32(_mm_mul_ps(widthRatioV, _mm_add_ps(pixel_x, _mm_setr_ps(12.0f, 13.0f, 14.0f, 15.0f))));
                const auto rowOffsetV = _mm_set1_epi32(int(rowOffset));

                const auto nx1 = _mm_add_epi32(nx1t, rowOffsetV);
                const auto nx2 = _mm_add_epi32(nx2t, rowOffsetV);
                const auto nx3 = _mm_add_epi32(nx3t, rowOffsetV);
                const auto nx4 = _mm_add_epi32(nx4t, rowOffsetV);
                _mm_storeu_si128((__m128i*)offsets.data(), nx1);
                _mm_storeu_si128(((__m128i*)offsets.data()) + 1, nx2);
                _mm_storeu_si128(((__m128i*)offsets.data()) + 2, nx3);
                _mm_storeu_si128(((__m128i*)offsets.data()) + 3, nx4);

                // accumulate RGB data
                for (size_t k = 0u; k < offsets.size(); ++k)
                {
                    const auto index = offsets[k];
                    const auto outputOffset = index * inputChannels;
                    const auto valueIndex = k * 3u;
                    outputSum[index] += 1u;
                    outputAclwm[outputOffset] += values[valueIndex];
                    outputAclwm[outputOffset + 1] += values[valueIndex + 1];
                    outputAclwm[outputOffset + 2] += values[valueIndex + 2];
                }
            }
            // do the last N (<16) pixels in scalar mode
            if (!inputWidthIsMultipleOf16)
            {
                for (uint32_t i = inputWidth & 0xFFFFFFF0u; i < inputWidth; ++i)
                {
                    const auto nx = uint64_t(outputWidth) * i / inputWidth;
                    const auto outputCoords = rowOffset + nx;
                    const auto outputCoordsColor = inputChannels * outputCoords;
                    const auto inputCoordsColor = inputChannels * (uint64_t(j) * inputWidth + i);
                    for (uint32_t c = 0; c < inputChannels; ++c)
                        outputAclwm[size_t(outputCoordsColor + c)] += input[inputCoordsColor + c];
                    outputSum[size_t(outputCoords)] += 1u;
                }
            }
        }
        return darkroom::Error::kSuccess;
    }


    template <typename T, typename S, typename Q, Codepath path>
    Error downscaleAverageHelper(T* output, S* outputAclwm, Q* outputSum, unsigned int outputChannels, unsigned int outputWidth, unsigned int outputHeight)
    {
        for (uint32_t j = 0; j < outputHeight; ++j)
        {
            for (uint32_t i = 0; i < outputWidth; ++i)
            {
                const auto coords = j * outputWidth + i;
                const auto outputCoords = outputChannels * coords;
                for (uint32_t c = 0; c < outputChannels; ++c)
                {
                    const auto outputCoordsColor = outputCoords + c;
                    if (outputSum[coords])
                        output[outputCoordsColor] = static_cast<T>(outputAclwm[outputCoordsColor] / outputSum[coords]);
                }
            }
        }
        return Error::kSuccess;
    }

    template <>
    Error downscaleAverageHelper<unsigned char, float, float, Codepath::kSSE2>(unsigned char* output, float* outputAclwm, float* outputSum, unsigned int outputChannels, unsigned int outputWidth, unsigned int outputHeight)
    {
        // CPU should support SSE2
        if (!IsProcessorFeaturePresent(PF_XMMI64_INSTRUCTIONS_AVAILABLE))
            return darkroom::Error::kOperationFailed;

        // divide aclwmulated RGB data by the number of samples taken at each pixel
        const bool outputWidthIsMultipleOf4 = !(outputWidth & 0xC);
        for (uint32_t j = 0; j < outputHeight; ++j)
        {
            for (uint32_t i = 0; i < outputWidth; i += 4)
            {
                const auto coords = j * outputWidth + i;
                const auto inputCoords = outputChannels * coords;
                const auto outputCoords = outputChannels * coords;
                const auto rgbr = _mm_load_ps((outputAclwm + inputCoords));
                const auto gbrg = _mm_load_ps((outputAclwm + inputCoords + 4));
                const auto brgb = _mm_load_ps((outputAclwm + inputCoords + 8));
                const auto delims = _mm_load_ps((outputSum + coords));
                const auto rgbrDelim = _mm_shuffle_ps(delims, delims, _MM_SHUFFLE(1, 0, 0, 0));
                const auto gbrgDelim = _mm_shuffle_ps(delims, delims, _MM_SHUFFLE(2, 2, 1, 1));
                const auto brgbDelim = _mm_shuffle_ps(delims, delims, _MM_SHUFFLE(3, 3, 3, 2));
                const auto rgbrResult = _mm_cvtps_epi32(_mm_div_ps(rgbr, rgbrDelim));
                const auto gbrgResult = _mm_cvtps_epi32(_mm_div_ps(gbrg, gbrgDelim));
                const auto brgbResult = _mm_cvtps_epi32(_mm_div_ps(brgb, brgbDelim));
                const auto rgbrResult16 = _mm_packus_epi32(rgbrResult, rgbrResult);
                const auto gbrgResult16 = _mm_packus_epi32(gbrgResult, gbrgResult);
                const auto brgbResult16 = _mm_packus_epi32(brgbResult, brgbResult);
                const auto rgbrResult8 = _mm_packus_epi16(rgbrResult16, rgbrResult16);
                const auto gbrgResult8 = _mm_packus_epi16(gbrgResult16, gbrgResult16);
                const auto brgbResult8 = _mm_packus_epi16(brgbResult16, brgbResult16);
                const uint32_t rgbrResultInt = uint32_t(_mm_cvtsi128_si32(rgbrResult8));
                const uint32_t gbrgResultInt = uint32_t(_mm_cvtsi128_si32(gbrgResult8));
                const uint32_t brgbResultInt = uint32_t(_mm_cvtsi128_si32(brgbResult8));
                *(unsigned int*)(output + outputCoords) = rgbrResultInt;
                *(unsigned int*)(output + outputCoords + 4) = gbrgResultInt;
                *(unsigned int*)(output + outputCoords + 8) = brgbResultInt;
            }
            if (!outputWidthIsMultipleOf4)
            {
                // process last N (< 4) pixels of the row in scalar mode
                for (uint32_t i = outputWidth & 0xFFFFFFFC; i < outputWidth; ++i)
                {
                    const auto coords = j * outputWidth + i;
                    const auto outputCoords = outputChannels * coords;
                    for (uint32_t c = 0; c < outputChannels; ++c)
                    {
                        const auto outputCoordsColor = outputCoords + c;
                        if (outputSum[coords])
                            output[outputCoordsColor] = static_cast<unsigned char>(outputAclwm[outputCoordsColor] / outputSum[coords]);
                    }
                }
            }
        }
        return Error::kSuccess;
    }

    template <typename T, typename S, typename Q>
    Error downscaleAclwmulate(const T* input, S* outputAclwm, Q* outputSum,
        unsigned int inputChannels, unsigned int inputWidth, unsigned int inputHeight,
        unsigned int outputChannels, unsigned int outputWidth, unsigned int outputHeight,
        unsigned int xOffset, unsigned int yOffset, unsigned int windowWidth,
        unsigned int windowHeight)
    {
        const auto err = downscaleAclwmulateHelper<T, S, Q, Codepath::kSSE2>(
            input, outputAclwm, outputSum, inputChannels, inputWidth, inputHeight,
            outputChannels, outputWidth, outputHeight, xOffset, yOffset, windowWidth, windowHeight);
        if (err != Error::kSuccess)
            return downscaleAclwmulateHelper<T, S, Q, Codepath::kScalar>(
                input, outputAclwm, outputSum, inputChannels, inputWidth, inputHeight,
                outputChannels, outputWidth, outputHeight, xOffset, yOffset, windowWidth, windowHeight);
        return err;
    }

    template <>
    Error downscaleAclwmulate(const float* input, float* outputAclwm, float* outputSum,
        unsigned int inputChannels, unsigned int inputWidth, unsigned int inputHeight,
        unsigned int outputChannels, unsigned int outputWidth, unsigned int outputHeight,
        unsigned int xOffset, unsigned int yOffset, unsigned int windowWidth,
        unsigned int windowHeight)
    {
        return downscaleAclwmulateHelper<float, float, float, Codepath::kScalar>(
            input, outputAclwm, outputSum, inputChannels, inputWidth, inputHeight,
            outputChannels, outputWidth, outputHeight, xOffset, yOffset, windowWidth, windowHeight);
    }

    template <typename T, typename S>
    Error downscaleAverage(T* output, S* outputAclwm, float* outputSum, unsigned int outputChannels, unsigned int outputWidth, unsigned int outputHeight)
    {
        const auto err = downscaleAverageHelper<T, S, float, Codepath::kSSE2>(output, outputAclwm, outputSum, outputChannels, outputWidth, outputHeight);
        if (err != Error::kSuccess)
            return downscaleAverageHelper<T, S, float, Codepath::kScalar>(output, outputAclwm, outputSum, outputChannels, outputWidth, outputHeight);
        return err;
    }

    template <typename T, typename Q>
    Error downscale(T* output, const Q* input,
        unsigned int inputChannels, unsigned int inputWidth, unsigned int inputHeight,
        unsigned int outputChannels, unsigned int outputWidth, unsigned int outputHeight,
        unsigned int xOffset, unsigned int yOffset, unsigned int windowWidth,
        unsigned int windowHeight) 
    {
        std::vector<float, AlignmentAllocator<float>> outputAclwm(outputWidth * outputHeight * inputChannels, 0);
        std::vector<float, AlignmentAllocator<float>> outputSum(outputWidth * outputHeight, 0);
        auto err = downscaleAclwmulate(input, outputAclwm.data(), outputSum.data(), inputChannels, 
            inputWidth, inputHeight, outputChannels, outputWidth, outputHeight, xOffset, yOffset, windowWidth, windowHeight);
        if (err != Error::kSuccess)
            return err;
        return downscaleAverage(output, outputAclwm.data(), outputSum.data(), outputChannels, outputWidth, outputHeight);
    }

    // tonemapping
    template<TonemapOperator op, Codepath path>
    Error tonemapHelper(float* input, unsigned char* output, unsigned int width, unsigned int height, unsigned int channels)
    {
        UNREFERENCED_PARAMETER(input);
        UNREFERENCED_PARAMETER(output);
        UNREFERENCED_PARAMETER(width);
        UNREFERENCED_PARAMETER(height);
        UNREFERENCED_PARAMETER(channels);
        return Error::kOperationFailed;
    }
    
    template<>
    Error tonemapHelper<TonemapOperator::kFilmic, Codepath::kScalar>(float* input, unsigned char* output, unsigned int width, unsigned int height, unsigned int channels)
    {
        const uint64_t inputLength = width * height * channels;
        for (uint64_t i = 0u; i < inputLength; ++i)
        {
            //float linColor = input[i];//powf(input[i], 1.0f/2.2f);
            float linColor = powf(input[i], 1.0f/2.2f);
            float x = std::max(0.0f, linColor - 0.004f);
            // Gamma correction is included into this formula
            x = (x*(6.2f*x+0.5f))/(x*(6.2f*x+1.7f)+0.06f);
            output[i] = static_cast<unsigned char>(255 * std::max(0.0f, std::min(1.0f, x)));
        }
        return Error::kSuccess;
    }

    template<>
    Error tonemapHelper<TonemapOperator::kFilmic, Codepath::kSSE2>(float* input, unsigned char* output, unsigned int width, unsigned int height, unsigned int channels)
    {
        // CPU should support SSE2
        if (!IsProcessorFeaturePresent(PF_XMMI64_INSTRUCTIONS_AVAILABLE))
            return darkroom::Error::kOperationFailed;

        const uint64_t inputLength = width * height * channels;
        const auto inputLengthFloor16 = inputLength & 0xFFFFFFFFFFFFFFF0ull;
        const auto inputLengthMod16 = inputLength - inputLengthFloor16;
        for (uint64_t i = 0; i < inputLengthFloor16; i += 16)
        {
            auto filmicTonemap = [](const float * data)
            {
                const __m128 f0_0 = _mm_set_ps1(0.0f);
                const __m128 f0_004 = _mm_set_ps1(0.004f);
                const __m128 f0_06 = _mm_set_ps1(0.06f);
                const __m128 f0_5 = _mm_set_ps1(0.5f);
                const __m128 f1_0 = _mm_set_ps1(1.0f);
                const __m128 f1_7 = _mm_set_ps1(1.7f);
                const __m128 f6_2 = _mm_set_ps1(6.2f);
                const __m128 f255_0 = _mm_set_ps1(255.0f);

                const __m128 dataSSE = _mm_load_ps(data);
                const __m128 dataLinear = SSEHelpers::exp_ps(_mm_mul_ps(SSEHelpers::log_ps(dataSSE), _mm_set_ps1(1.0f/2.2f)));

                //float x = std::max(0.0f, linColor - 0.004f);
                const __m128 mm_x = _mm_max_ps(_mm_sub_ps(dataLinear, f0_004), f0_0);

                // Gamma correction is included into this formula
                //x = (x*(6.2f*x+0.5f)) / (x*(6.2f*x+1.7f)+0.06f);
                const __m128 num = _mm_mul_ps(mm_x, _mm_add_ps(_mm_mul_ps(mm_x, f6_2), f0_5));
                const __m128 denum = _mm_rcp_ps(_mm_add_ps(_mm_mul_ps(mm_x, _mm_add_ps(_mm_mul_ps(mm_x, f6_2), f1_7)), f0_06));
                const __m128 result = _mm_mul_ps(num, denum);
                const __m128 resultClamped = _mm_min_ps(_mm_max_ps(result, f0_0), f1_0);
                const __m128i resultClamped255 = _mm_cvtps_epi32(_mm_mul_ps(resultClamped, f255_0));
                const __m128i resultClamped255pack1 = _mm_packus_epi32(resultClamped255, resultClamped255);
                const __m128i resultClamped255pack2 = _mm_packus_epi16(resultClamped255pack1, resultClamped255pack1);
                return _mm_cvtsi128_si32(resultClamped255pack2);
            };

            const float * base = input + i;
            const auto result1 = filmicTonemap(base);
            const auto result2 = filmicTonemap(base + 4);
            const auto result3 = filmicTonemap(base + 8);
            const auto result4 = filmicTonemap(base + 12);

            const auto outputUint = reinterpret_cast<int*>(output + i);
            outputUint[0] = result1;
            outputUint[1] = result2;
            outputUint[2] = result3;
            outputUint[3] = result4;
        }

        for (uint64_t i = inputLengthFloor16; i < inputLength; ++i)
        {
            float linColor = powf(input[i], 1.0f/2.2f);
            float x = std::max(0.0f, linColor - 0.004f);
            // Gamma correction is included into this formula
            x = (x*(6.2f*x+0.5f))/(x*(6.2f*x+1.7f)+0.06f);
            output[i] = static_cast<unsigned char>(255 * std::max(0.0f, std::min(1.0f, x)));
        }
        return Error::kSuccess;
    }

    template<>
    Error tonemapHelper<TonemapOperator::kFilmicLinear, Codepath::kScalar>(float* input, unsigned char* output, unsigned int width, unsigned int height, unsigned int channels)
    {
        const uint64_t inputLength = width * height * channels;
        for (uint64_t i = 0u; i < inputLength; ++i)
        {
            float x = std::max(0.0f, input[i] - 0.004f);
            // Gamma correction is included into this formula
            x = (x*(5.9f*x+0.42f))/(x*(5.4f*x+3.0f)+0.01f);
            output[i] = static_cast<unsigned char>(255 * std::max(0.0f, std::min(1.0f, x)));
        }
        return Error::kSuccess;
    }

    template<>
    Error tonemapHelper<TonemapOperator::kFilmicLinear, Codepath::kSSE2>(float* input, unsigned char* output, unsigned int width, unsigned int height, unsigned int channels)
    {
        // CPU should support SSE2
        if (!IsProcessorFeaturePresent(PF_XMMI64_INSTRUCTIONS_AVAILABLE))
            return darkroom::Error::kOperationFailed;

        const uint64_t inputLength = width * height * channels;
        const auto inputLengthFloor16 = inputLength & 0xFFFFFFFFFFFFFFF0ull;
        const auto inputLengthMod16 = inputLength - inputLengthFloor16;
        for (uint64_t i = 0; i < inputLengthFloor16; i += 16)
        {
            auto filmicTonemap = [](const float * data)
            {
                const __m128 f0_0 = _mm_set_ps1(0.0f);
                const __m128 f1_0 = _mm_set_ps1(1.0f);
                const __m128 f255_0 = _mm_set_ps1(255.0f);

                const __m128 dataSSE = _mm_load_ps(data);

                //float x = std::max(0.0f, linColor - 0.004f);
                const __m128 mm_x = _mm_max_ps(_mm_sub_ps(dataSSE, _mm_set_ps1(0.004f)), f0_0);

                // Gamma correction is included into this formula
                //x = (x*(5.9f*x+0.42f))/(x*(5.4f*x+3.0f)+0.01f);
                const __m128 num = _mm_mul_ps(mm_x, _mm_add_ps(_mm_mul_ps(mm_x, _mm_set_ps1(6.9f)), _mm_set_ps1(0.42f)));
                const __m128 denum = _mm_rcp_ps(_mm_add_ps(_mm_mul_ps(mm_x, _mm_add_ps(_mm_mul_ps(mm_x, _mm_set_ps1(5.0f)), _mm_set_ps1(3.0f))), _mm_set_ps1(0.01f)));

                const __m128 result = _mm_mul_ps(num, denum);
                const __m128 resultClamped = _mm_min_ps(_mm_max_ps(result, f0_0), f1_0);

                const __m128i resultClamped255 = _mm_cvtps_epi32(_mm_mul_ps(resultClamped, f255_0));
                const __m128i resultClamped255pack1 = _mm_packus_epi32(resultClamped255, resultClamped255);
                const __m128i resultClamped255pack2 = _mm_packus_epi16(resultClamped255pack1, resultClamped255pack1);
                return _mm_cvtsi128_si32(resultClamped255pack2);
            };

            const float * base = input + i;
            const auto result1 = filmicTonemap(base);
            const auto result2 = filmicTonemap(base + 4);
            const auto result3 = filmicTonemap(base + 8);
            const auto result4 = filmicTonemap(base + 12);

            const auto outputUint = reinterpret_cast<int*>(output + i);
            outputUint[0] = result1;
            outputUint[1] = result2;
            outputUint[2] = result3;
            outputUint[3] = result4;
        }

        for (uint64_t i = inputLengthFloor16; i < inputLength; ++i)
        {
            float x = std::max(0.0f, input[i] - 0.004f);
            // Gamma correction is included into this formula
            x = (x*(6.2f*x+0.5f))/(x*(6.2f*x+1.7f)+0.5f);
            output[i] = static_cast<unsigned char>(255 * std::max(0.0f, std::min(1.0f, x)));
        }
        return Error::kSuccess;
    }

    template<>
    Error tonemapHelper<TonemapOperator::kReinhardSimple, Codepath::kScalar>(float* input, unsigned char* output, unsigned int width, unsigned int height, unsigned int channels)
    {
        const uint64_t inputLength = width * height * channels;
        for (uint64_t i = 0u; i < inputLength; ++i)
        {
            float n = input[i];
            // no gamma correction
            n = n / (n + 1.0f);
            output[i] = static_cast<unsigned char>(255 * std::max(0.0f, std::min(1.0f, n)));
        }
        return Error::kSuccess;
    }

    template<>
    Error tonemapHelper<TonemapOperator::kReinhardSimple, Codepath::kSSE2>(float* input, unsigned char* output, unsigned int width, unsigned int height, unsigned int channels)
    {
        // CPU should support SSE2
        if (!IsProcessorFeaturePresent(PF_XMMI64_INSTRUCTIONS_AVAILABLE))
            return darkroom::Error::kOperationFailed;

        const uint64_t inputLength = width * height * channels;
        const auto inputLengthFloor16 = inputLength & 0xFFFFFFFFFFFFFFF0ull;
        const auto inputLengthMod16 = inputLength - inputLengthFloor16;
        const auto zeros = _mm_set_ps1(0.0f);
        const auto ones = _mm_set_ps1(1.0f);
        const auto mult255 = _mm_set_ps1(255.0f);
        for (uint64_t i = 0; i < inputLengthFloor16; i += 16)
        {
            const auto base = input + i;
            const auto pack1 = _mm_load_ps(base);
            const auto pack1plus1 = _mm_add_ps(pack1, ones);
            const auto pack1plus1recip = _mm_rcp_ps(pack1plus1);
            const auto pack1result = _mm_mul_ps(pack1, pack1plus1recip);
            const auto pack1resultClamped = _mm_min_ps(_mm_max_ps(pack1result, zeros), ones);
            const auto pack1resultClamped255 = _mm_cvtps_epi32(_mm_mul_ps(pack1resultClamped, mult255));
            const auto pack1resultClamped255pack1 = _mm_packus_epi32(pack1resultClamped255, pack1resultClamped255);
            const auto pack1resultClamped255pack2 = _mm_packus_epi16(pack1resultClamped255pack1, pack1resultClamped255pack1);
            const auto result1 = _mm_cvtsi128_si32(pack1resultClamped255pack2);
            const auto pack2 = _mm_load_ps(base + 4);
            const auto pack2plus1 = _mm_add_ps(pack2, ones);
            const auto pack2plus1recip = _mm_rcp_ps(pack2plus1);
            const auto pack2result = _mm_mul_ps(pack2, pack2plus1recip);
            const auto pack2resultClamped = _mm_min_ps(_mm_max_ps(pack2result, zeros), ones);
            const auto pack2resultClamped255 = _mm_cvtps_epi32(_mm_mul_ps(pack2resultClamped, mult255));
            const auto pack2resultClamped255pack1 = _mm_packus_epi32(pack2resultClamped255, pack2resultClamped255);
            const auto pack2resultClamped255pack2 = _mm_packus_epi16(pack2resultClamped255pack1, pack2resultClamped255pack1);
            const auto result2 = _mm_cvtsi128_si32(pack2resultClamped255pack2);
            const auto pack3 = _mm_load_ps(base + 8);
            const auto pack3plus1 = _mm_add_ps(pack3, ones);
            const auto pack3plus1recip = _mm_rcp_ps(pack3plus1);
            const auto pack3result = _mm_mul_ps(pack3, pack3plus1recip);
            const auto pack3resultClamped = _mm_min_ps(_mm_max_ps(pack3result, zeros), ones);
            const auto pack3resultClamped255 = _mm_cvtps_epi32(_mm_mul_ps(pack3resultClamped, mult255));
            const auto pack3resultClamped255pack1 = _mm_packus_epi32(pack3resultClamped255, pack3resultClamped255);
            const auto pack3resultClamped255pack2 = _mm_packus_epi16(pack3resultClamped255pack1, pack3resultClamped255pack1);
            const auto result3 = _mm_cvtsi128_si32(pack3resultClamped255pack2);
            const auto pack4 = _mm_load_ps(base + 12);
            const auto pack4plus1 = _mm_add_ps(pack4, ones);
            const auto pack4plus1recip = _mm_rcp_ps(pack4plus1);
            const auto pack4result = _mm_mul_ps(pack4, pack4plus1recip);
            const auto pack4resultClamped = _mm_min_ps(_mm_max_ps(pack4result, zeros), ones);
            const auto pack4resultClamped255 = _mm_cvtps_epi32(_mm_mul_ps(pack4resultClamped, mult255));
            const auto pack4resultClamped255pack1 = _mm_packus_epi32(pack4resultClamped255, pack4resultClamped255);
            const auto pack4resultClamped255pack2 = _mm_packus_epi16(pack4resultClamped255pack1, pack4resultClamped255pack1);
            const auto result4 = _mm_cvtsi128_si32(pack4resultClamped255pack2);
            const auto outputUint = reinterpret_cast<int*>(output + i);
            outputUint[0] = result1;
            outputUint[1] = result2;
            outputUint[2] = result3;
            outputUint[3] = result4;
        }

        for (uint64_t i = inputLengthFloor16; i < inputLength; ++i)
        {
            float n = input[i];

            // no gamma correction
            n = n / (n + 1.0f);
            output[i] = static_cast<unsigned char>(255 * std::max(0.0f, std::min(1.0f, n)));
        }
        return Error::kSuccess;
    }

    template<>
    Error tonemapHelper<TonemapOperator::kClamp, Codepath::kScalar>(float* input, unsigned char* output, unsigned int width, unsigned int height, unsigned int channels)
    {
        const uint64_t inputLength = width * height * channels;
        for (uint64_t i = 0u; i < inputLength; ++i)
            output[i] = static_cast<unsigned char>(255 * std::max(0.0f, std::min(1.0f, input[i])));
        return Error::kSuccess;
    }

    template<>
    Error tonemapHelper<TonemapOperator::kClamp, Codepath::kSSE2>(float* input, unsigned char* output, unsigned int width, unsigned int height, unsigned int channels)
    {
        // CPU should support SSE2
        if (!IsProcessorFeaturePresent(PF_XMMI64_INSTRUCTIONS_AVAILABLE))
            return darkroom::Error::kOperationFailed;

        const uint64_t inputLength = width * height * channels;
        const auto inputLengthFloor16 = inputLength & 0xFFFFFFFFFFFFFFF0ull;
        const auto inputLengthMod16 = inputLength - inputLengthFloor16;
        const auto zeros = _mm_set_ps1(0.0f);
        const auto ones = _mm_set_ps1(1.0f);
        const auto mult255 = _mm_set_ps1(255.0f);
        for (uint64_t i = 0; i < inputLengthFloor16; i += 16)
        {
            const auto base = input + i;
            const auto pack1 = _mm_load_ps(base);
            const auto pack1resultClamped = _mm_min_ps(_mm_max_ps(pack1, zeros), ones);
            const auto pack1resultClamped255 = _mm_cvtps_epi32(_mm_mul_ps(pack1resultClamped, mult255));
            const auto pack1resultClamped255pack1 = _mm_packus_epi32(pack1resultClamped255, pack1resultClamped255);
            const auto pack1resultClamped255pack2 = _mm_packus_epi16(pack1resultClamped255pack1, pack1resultClamped255pack1);
            const auto result1 = _mm_cvtsi128_si32(pack1resultClamped255pack2);
            const auto pack2 = _mm_load_ps(base + 4);
            const auto pack2resultClamped = _mm_min_ps(_mm_max_ps(pack2, zeros), ones);
            const auto pack2resultClamped255 = _mm_cvtps_epi32(_mm_mul_ps(pack2resultClamped, mult255));
            const auto pack2resultClamped255pack1 = _mm_packus_epi32(pack2resultClamped255, pack2resultClamped255);
            const auto pack2resultClamped255pack2 = _mm_packus_epi16(pack2resultClamped255pack1, pack2resultClamped255pack1);
            const auto result2 = _mm_cvtsi128_si32(pack2resultClamped255pack2);
            const auto pack3 = _mm_load_ps(base + 8);
            const auto pack3resultClamped = _mm_min_ps(_mm_max_ps(pack3, zeros), ones);
            const auto pack3resultClamped255 = _mm_cvtps_epi32(_mm_mul_ps(pack3resultClamped, mult255));
            const auto pack3resultClamped255pack1 = _mm_packus_epi32(pack3resultClamped255, pack3resultClamped255);
            const auto pack3resultClamped255pack2 = _mm_packus_epi16(pack3resultClamped255pack1, pack3resultClamped255pack1);
            const auto result3 = _mm_cvtsi128_si32(pack3resultClamped255pack2);
            const auto pack4 = _mm_load_ps(base + 12);
            const auto pack4resultClamped = _mm_min_ps(_mm_max_ps(pack4, zeros), ones);
            const auto pack4resultClamped255 = _mm_cvtps_epi32(_mm_mul_ps(pack4resultClamped, mult255));
            const auto pack4resultClamped255pack1 = _mm_packus_epi32(pack4resultClamped255, pack4resultClamped255);
            const auto pack4resultClamped255pack2 = _mm_packus_epi16(pack4resultClamped255pack1, pack4resultClamped255pack1);
            const auto result4 = _mm_cvtsi128_si32(pack4resultClamped255pack2);
            const auto outputUint = reinterpret_cast<int*>(output + i);
            outputUint[0] = result1;
            outputUint[1] = result2;
            outputUint[2] = result3;
            outputUint[3] = result4;
        }

        for (uint64_t i = inputLengthFloor16; i < inputLength; ++i)
            output[i] = static_cast<unsigned char>(255 * std::max(0.0f, std::min(1.0f, input[i])));
        return Error::kSuccess;
    }


    template<TonemapOperator op>
    Error tonemap(float* input, unsigned char* output, unsigned int width, unsigned int height, unsigned int channels)
    {
        const auto err = tonemapHelper<op, Codepath::kSSE2>(input, output, width, height, channels);
        if (err != Error::kSuccess)
            return tonemapHelper<op, Codepath::kScalar>(input, output, width, height, channels);
        return err;
    }

// explicitly instantiate functions for supported type combinations
    template Error appendHorizontally(std::vector<unsigned char>& output,
        const std::vector<unsigned char>& input1,
        const std::vector<unsigned char>& input2,
        unsigned int width, unsigned int height);
    template Error appendHorizontally(std::vector<float>& output,
        const std::vector<float>& input1,
        const std::vector<float>& input2,
        unsigned int width, unsigned int height);

    template Error appendVertically(std::vector<unsigned char>& output,
        const std::vector<unsigned char>& input1,
        const std::vector<unsigned char>& input2);
    template Error appendVertically(std::vector<float>& output,
        const std::vector<float>& input1,
        const std::vector<float>& input2);

    template Error crop(std::vector<unsigned char>& output, const std::vector<unsigned char>& input, unsigned int width, unsigned int height,
        unsigned int x, unsigned int y, unsigned int cropWidth, unsigned int cropHeight);
    template Error crop(std::vector<float>& output, const std::vector<float>& input, unsigned int width, unsigned int height,
        unsigned int x, unsigned int y, unsigned int cropWidth, unsigned int cropHeight);

    template Error downscale(unsigned char* output, const unsigned char* input,
        unsigned int inputChannels, unsigned int inputWidth, unsigned int inputHeight,
        unsigned int outputChannels, unsigned int outputWidth, unsigned int outputHeight,
        unsigned int xOffset, unsigned int yOffset, unsigned int xWidth, unsigned int xHeight);
    template Error downscale(unsigned char* output, const float* input,
        unsigned int inputChannels, unsigned int inputWidth, unsigned int inputHeight,
        unsigned int outputChannels, unsigned int outputWidth, unsigned int outputHeight,
        unsigned int xOffset, unsigned int yOffset, unsigned int xWidth, unsigned int xHeight);
    template Error downscale(float* output, const float* input,
        unsigned int inputChannels, unsigned int inputWidth, unsigned int inputHeight,
        unsigned int outputChannels, unsigned int outputWidth, unsigned int outputHeight,
        unsigned int xOffset, unsigned int yOffset, unsigned int xWidth, unsigned int xHeight);

    template Error downscaleAclwmulate(const unsigned char* input,
        float* outputAclwm, float* outputSum,
        unsigned int inputChannels, unsigned int inputWidth, unsigned int inputHeight,
        unsigned int outputChannels, unsigned int outputWidth, unsigned int outputHeight,
        unsigned int xOffset, unsigned int yOffset, unsigned int windowWidth, unsigned int windowHeight);
    template Error downscaleAverage(unsigned char* output, float* outputAclwm, float* outputSum, unsigned int outputChannels, unsigned int outputWidth, unsigned int outputHeight);
    template Error downscaleAverage(float* output, float* outputAclwm, float* outputSum, unsigned int outputChannels, unsigned int outputWidth, unsigned int outputHeight);

    template Error tonemap<TonemapOperator::kClamp>(float* input, unsigned char* output, unsigned int width, unsigned int height, unsigned int channels);
    template Error tonemap<TonemapOperator::kReinhardSimple>(float* input, unsigned char* output, unsigned int width, unsigned int height, unsigned int channels);
    template Error tonemap<TonemapOperator::kFilmic>(float* input, unsigned char* output, unsigned int width, unsigned int height, unsigned int channels);
    template Error tonemap<TonemapOperator::kFilmicLinear>(float* input, unsigned char* output, unsigned int width, unsigned int height, unsigned int channels);
}
