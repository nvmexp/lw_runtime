/* Copyright (c) 2022, LWPU CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of LWPU CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROLWREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// This sample needs at least LWCA 10.1. It demonstrates usages of the lwJPEG
// library lwJPEG encoder supports single and multiple image encode.

#include <lwda_runtime_api.h>
#include "helper_lwJPEG.hxx"


int dev_malloc(void **p, size_t s) { return (int)lwdaMalloc(p, s); }
int dev_free(void *p) { return (int)lwdaFree(p); }

bool is_interleaved(lwjpegOutputFormat_t format)
{
    if (format == LWJPEG_OUTPUT_RGBI || format == LWJPEG_OUTPUT_BGRI)
        return true;
    else
        return false;
}

struct encode_params_t {
  std::string input_dir;
  std::string output_dir;
  std::string format;
  std::string subsampling;
  int quality;
  int huf;
  int dev;
};

lwjpegEncoderParams_t encode_params;
lwjpegHandle_t lwjpeg_handle;
lwjpegJpegState_t jpeg_state;
lwjpegEncoderState_t encoder_state;

int decodeEncodeOneImage(std::string sImagePath, std::string sOutputPath, double &time, lwjpegOutputFormat_t output_format, lwjpegInputFormat_t input_format)
{
    time = 0.;
    lwdaEvent_t startEvent = NULL, stopEvent = NULL;
    float loopTime = 0;
    checkLwdaErrors(lwdaEventCreate(&startEvent, lwdaEventBlockingSync));
    checkLwdaErrors(lwdaEventCreate(&stopEvent, lwdaEventBlockingSync));

    // Get the file name, without extension.
    // This will be used to rename the output file.    
    size_t position = sImagePath.rfind("/");
    std::string sFileName = (std::string::npos == position)? sImagePath : sImagePath.substr(position + 1, sImagePath.size());
    position = sFileName.rfind(".");
    sFileName = (std::string::npos == position)? sFileName : sFileName.substr(0, position);
    position = sFileName.rfind("/");
    sFileName = (std::string::npos == position) ? sFileName : sFileName.substr(position + 1, sFileName.length());
    position = sFileName.rfind("\\");
    sFileName = (std::string::npos == position) ? sFileName : sFileName.substr(position+1, sFileName.length());

    // Read an image from disk.
    std::ifstream oInputStream(sImagePath.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
    if(!(oInputStream.is_open()))
    {
        std::cerr << "Cannot open image: " << sImagePath << std::endl;
        return 1;
    }
    
    // Get the size.
    std::streamsize nSize = oInputStream.tellg();
    oInputStream.seekg(0, std::ios::beg);

    // Image buffers. 
    unsigned char * pBuffer = NULL; 
    double encoder_time = 0.;
    
    std::vector<char> vBuffer(nSize);
    
    if (oInputStream.read(vBuffer.data(), nSize))
    {            
        unsigned char * dpImage = (unsigned char *)vBuffer.data();
        
        // Retrieve the componenet and size info.
        int nComponent = 0;
        lwjpegChromaSubsampling_t subsampling;
        int widths[LWJPEG_MAX_COMPONENT];
        int heights[LWJPEG_MAX_COMPONENT];
        if (LWJPEG_STATUS_SUCCESS != lwjpegGetImageInfo(lwjpeg_handle, dpImage, nSize, &nComponent, &subsampling, widths, heights))
        {
            std::cerr << "Error decoding JPEG header: " << sImagePath << std::endl;
            return 1;
        }

        // image information
        std::cout << "Image is " << nComponent << " channels." << std::endl;
        for (int i = 0; i < nComponent; i++)
        {
            std::cout << "Channel #" << i << " size: "  << widths[i]  << " x " << heights[i] << std::endl;    
        }
        
        switch (subsampling)
        {
            case LWJPEG_CSS_444:
                std::cout << "YUV 4:4:4 chroma subsampling" << std::endl;
                break;
            case LWJPEG_CSS_440:
                std::cout << "YUV 4:4:0 chroma subsampling" << std::endl;
                break;
            case LWJPEG_CSS_422:
                std::cout << "YUV 4:2:2 chroma subsampling" << std::endl;
                break;
            case LWJPEG_CSS_420:
                std::cout << "YUV 4:2:0 chroma subsampling" << std::endl;
                break;
            case LWJPEG_CSS_411:
                std::cout << "YUV 4:1:1 chroma subsampling" << std::endl;
                break;
            case LWJPEG_CSS_410:
                std::cout << "YUV 4:1:0 chroma subsampling" << std::endl;
                break;
            case LWJPEG_CSS_GRAY:
                std::cout << "Grayscale JPEG " << std::endl;
                break;
            case LWJPEG_CSS_UNKNOWN: 
                std::cout << "Unknown chroma subsampling" << std::endl;
                return 1;
        }

        {

            lwdaError_t eCopy = lwdaMalloc(&pBuffer, widths[0] * heights[0] * LWJPEG_MAX_COMPONENT);
            if(lwdaSuccess != eCopy) 
            {
                std::cerr << "lwdaMalloc failed for component Y: " << lwdaGetErrorString(eCopy) << std::endl;
                return 1;
            }

            lwjpegImage_t imgdesc = 
            {
                {
                    pBuffer,
                    pBuffer + widths[0]*heights[0],
                    pBuffer + widths[0]*heights[0]*2,
                    pBuffer + widths[0]*heights[0]*3
                },
                {
                    (unsigned int)(is_interleaved(output_format) ? widths[0] * 3 : widths[0]),
                    (unsigned int)widths[0],
                    (unsigned int)widths[0],
                    (unsigned int)widths[0]
                }
            };
           
            int nReturnCode = 0;

            lwdaDeviceSynchronize();

            nReturnCode = lwjpegDecode(lwjpeg_handle, jpeg_state, dpImage, nSize, output_format, &imgdesc, NULL);

            // alternatively decode by stages
            /*int nReturnCode = lwjpegDecodeCPU(lwjpeg_handle, dpImage, nSize, output_format, &imgdesc, NULL);
            nReturnCode = lwjpegDecodeMixed(lwjpeg_handle, NULL);
            nReturnCode = lwjpegDecodeGPU(lwjpeg_handle, NULL);*/
            lwdaDeviceSynchronize();

            if(nReturnCode != 0)
            {
                std::cerr << "Error in lwjpegDecode." << std::endl;
                return 1;
            }

            checkLwdaErrors(lwdaEventRecord(startEvent, NULL));
            /////////////////////// encode ////////////////////
            if (LWJPEG_OUTPUT_YUV == output_format)
            {
                checkLwdaErrors(lwjpegEncodeYUV(lwjpeg_handle,
                    encoder_state,
                    encode_params,
                    &imgdesc,
                    subsampling,
                    widths[0],
                    heights[0],
                    NULL));
            }
            else
            {
                checkLwdaErrors(lwjpegEncodeImage(lwjpeg_handle,
                    encoder_state,
                    encode_params,
                    &imgdesc,
                    input_format,
                    widths[0],
                    heights[0],
                    NULL));
            }

            std::vector<unsigned char> obuffer;
            size_t length;
            checkLwdaErrors(lwjpegEncodeRetrieveBitstream(
                lwjpeg_handle,
                encoder_state,
                NULL,
                &length,
                NULL));
            obuffer.resize(length);
            checkLwdaErrors(lwjpegEncodeRetrieveBitstream(
                lwjpeg_handle,
                encoder_state,
                obuffer.data(),
                &length,
                NULL));

            checkLwdaErrors(lwdaEventRecord(stopEvent, NULL));
            checkLwdaErrors(lwdaEventSynchronize(stopEvent));
            checkLwdaErrors(lwdaEventElapsedTime(&loopTime, startEvent, stopEvent));
            encoder_time = static_cast<double>(loopTime);

            std::string output_filename = sOutputPath + "/" + sFileName + ".jpg";
            char directory[120];
            char mkdir_cmd[256];
            std::string folder = sOutputPath;
            output_filename = folder + "/"+ sFileName +".jpg";
#if !defined(_WIN32)
            sprintf(directory, "%s", folder.c_str());
            sprintf(mkdir_cmd, "mkdir -p %s 2> /dev/null", directory);
#else
            sprintf(directory, "%s", folder.c_str());
            sprintf(mkdir_cmd, "mkdir %s 2> nul", directory);
#endif

            int ret = system(mkdir_cmd);

            std::cout << "Writing JPEG file: " << output_filename << std::endl;
            std::ofstream outputFile(output_filename.c_str(), std::ios::out | std::ios::binary);
            outputFile.write(reinterpret_cast<const char *>(obuffer.data()), static_cast<int>(length));
            
            // Free memory
            checkLwdaErrors(lwdaFree(pBuffer));
        }
    }

    time = encoder_time;

    return 0;
}

int processArgs(encode_params_t param)
{
    std::string sInputPath(param.input_dir);
    std::string sOutputPath(param.output_dir);
    std::string sFormat(param.format);
    std::string sSubsampling(param.subsampling);
    lwjpegOutputFormat_t oformat = LWJPEG_OUTPUT_RGB;
    lwjpegInputFormat_t iformat = LWJPEG_INPUT_RGB;

    int error_code = 1;

    if (sFormat == "yuv")
    {
        oformat = LWJPEG_OUTPUT_YUV;
    } 
    else if (sFormat == "rgb")
    {
        oformat = LWJPEG_OUTPUT_RGB;
        iformat = LWJPEG_INPUT_RGB;
    }
    else if (sFormat == "bgr")
    {
        oformat = LWJPEG_OUTPUT_BGR;
        iformat = LWJPEG_INPUT_BGR;
    }
    else if (sFormat == "rgbi")
    {
        oformat = LWJPEG_OUTPUT_RGBI;
        iformat = LWJPEG_INPUT_RGBI;
    }
    else if (sFormat == "bgri")
    {
        oformat = LWJPEG_OUTPUT_BGRI;
        iformat = LWJPEG_INPUT_BGRI;
    }
    else 
    {
        std::cerr << "Unknown or unsupported output format: " << sFormat << std::endl;
        return error_code;
    }

    if (sSubsampling == "444")
    {
        checkLwdaErrors(lwjpegEncoderParamsSetSamplingFactors(encode_params, LWJPEG_CSS_444, NULL));
    }
    else if (sSubsampling == "422")
    {
        checkLwdaErrors(lwjpegEncoderParamsSetSamplingFactors(encode_params, LWJPEG_CSS_422, NULL));
    }
    else if (sSubsampling == "420")
    {
        checkLwdaErrors(lwjpegEncoderParamsSetSamplingFactors(encode_params, LWJPEG_CSS_420, NULL));
    }
    else if (sSubsampling == "440")
    {
        checkLwdaErrors(lwjpegEncoderParamsSetSamplingFactors(encode_params, LWJPEG_CSS_440, NULL));
    }
    else if (sSubsampling == "411")
    {
        checkLwdaErrors(lwjpegEncoderParamsSetSamplingFactors(encode_params, LWJPEG_CSS_411, NULL));
    }
    else if (sSubsampling == "410")
    {
        checkLwdaErrors(lwjpegEncoderParamsSetSamplingFactors(encode_params, LWJPEG_CSS_410, NULL));
    }
    else if (sSubsampling == "400")
    {
        checkLwdaErrors(lwjpegEncoderParamsSetSamplingFactors(encode_params, LWJPEG_CSS_GRAY, NULL));
    }
    else 
    {
        std::cerr << "Unknown or unsupported subsampling: " << sSubsampling << std::endl;
        return error_code;
    }
    /*if( stat(sOutputPath.c_str(), &s) == 0 )
    {
        if( !(s.st_mode & S_IFDIR) )
        {
            std::cout << "Output path already exist as non-directory: " << sOutputPath << std::endl;
            return error_code;
        }
    }
    else
    {
        if (mkdir(sOutputPath.c_str(), 0775))
        {
            std::cout << "Cannot create output directory: " << sOutputPath << std::endl;
            return error_code;
        }
    }*/

    std::vector<std::string> inputFiles;
    if (readInput(sInputPath, inputFiles))
    {
        return error_code;
    }
    
    double total_time = 0., encoder_time = 0.;
    int total_images = 0;

    for (unsigned int i = 0; i < inputFiles.size(); i++)
    {
        std::string &sFileName = inputFiles[i];
        std::cout << "Processing file: " << sFileName << std::endl;
        int image_error_code = decodeEncodeOneImage(sFileName, sOutputPath, encoder_time, oformat, iformat);
        if (image_error_code)
        {
            std::cerr << "Error processing file: " << sFileName << std::endl;
            //return image_error_code;
        }
        else
        {
            total_images++;
            total_time += encoder_time;
        }                      
    }
    std::cout << "Total images processed: " << total_images << std::endl;
    std::cout << "Total time spent on encoding: " << total_time << std::endl;
    std::cout << "Avg time/image: " << total_time/total_images << std::endl;

    return 0;
}

// parse parameters
int findParamIndex(const char **argv, int argc, const char *parm) {
  int count = 0;
  int index = -1;

  for (int i = 0; i < argc; i++) {
    if (strncmp(argv[i], parm, 100) == 0) {
      index = i;
      count++;
    }
  }

  if (count == 0 || count == 1) {
    return index;
  } else {
    std::cout << "Error, parameter " << parm
              << " has been specified more than once, exiting\n"
              << std::endl;
    return -1;
  }

  return -1;
}


int main(int argc, const char *argv[]) 
{
  int pidx;

  if ((pidx = findParamIndex(argv, argc, "-h")) != -1 ||
      (pidx = findParamIndex(argv, argc, "--help")) != -1) {
    std::cout << "Usage: " << argv[0]
              << " -i images_dir  [-o output_dir] [-device=device_id]"                 
                 "[-q quality][-s 420/444] [-fmt output_format] [-huf 0]\n";
    std::cout << "Parameters: " << std::endl;
    std::cout << "\timages_dir\t:\tPath to single image or directory of images" << std::endl;
    std::cout << "\toutput_dir\t:\tWrite encoded images as jpeg to this directory" << std::endl;
    std::cout << "\tdevice_id\t:\tWhich device to use for encoding" << std::endl;
    std::cout << "\tQuality\t:\tUse image quality [default 70]" << std::endl;
    std::cout << "\tsubsampling\t:\tUse Subsampling [420, 444]" << std::endl;
    std::cout << "\toutput_format\t:\tlwJPEG output format for encoding. One "
                 "of [rgb, rgbi, bgr, bgri, yuv, y, unchanged]"
              << std::endl;
    std::cout << "\tHuffman Optimization\t:\tUse Huffman optimization [default 0]" << std::endl;
    return EXIT_SUCCESS;
  }

  encode_params_t params;

  params.input_dir = "./";
  if ((pidx = findParamIndex(argv, argc, "-i")) != -1) {
    params.input_dir = argv[pidx + 1];
  } else {
    // Search in default paths for input images.
    int found = getInputDir(params.input_dir, argv[0]);
    if (!found)
    {
      std::cout << "Please specify input directory with encoded images"<< std::endl;
      return EXIT_WAIVED;
    }
  }
  if ((pidx = findParamIndex(argv, argc, "-o")) != -1) {
    params.output_dir = argv[pidx + 1];
  } else {
      // by-default write the folder named "output" in cwd
      params.output_dir = "encode_output";
  }
  params.dev = 0;
  params.dev = findLwdaDevice(argc, argv);

  params.quality = 70;
  if ((pidx = findParamIndex(argv, argc, "-q")) != -1) {
    params.quality = std::atoi(argv[pidx + 1]);
  }

  if ((pidx = findParamIndex(argv, argc, "-s")) != -1) {
    params.subsampling = argv[pidx + 1];
  } else {
      // by-default use subsampling as 420
      params.subsampling = "420";
  }
  if ((pidx = findParamIndex(argv, argc, "-fmt")) != -1) {
    params.format = argv[pidx + 1];
  } else {
   // by-default use output format yuv
    params.format = "yuv";
  }

  params.huf = 0;
  if ((pidx = findParamIndex(argv, argc, "-huf")) != -1) {
    params.huf = std::atoi(argv[pidx + 1]);
  }

    lwdaDeviceProp props;
    checkLwdaErrors(lwdaGetDeviceProperties(&props, params.dev));

    printf("Using GPU %d (%s, %d SMs, %d th/SM max, CC %d.%d, ECC %s)\n",
         params.dev, props.name, props.multiProcessorCount,
         props.maxThreadsPerMultiProcessor, props.major, props.minor,
         props.ECCEnabled ? "on" : "off");

    lwjpegDevAllocator_t dev_allocator = {&dev_malloc, &dev_free};
    checkLwdaErrors(lwjpegCreate(LWJPEG_BACKEND_DEFAULT, &dev_allocator, &lwjpeg_handle));
    checkLwdaErrors(lwjpegJpegStateCreate(lwjpeg_handle, &jpeg_state));
    checkLwdaErrors(lwjpegEncoderStateCreate(lwjpeg_handle, &encoder_state, NULL));
    checkLwdaErrors(lwjpegEncoderParamsCreate(lwjpeg_handle, &encode_params, NULL));
    
    // sample input parameters
    checkLwdaErrors(lwjpegEncoderParamsSetQuality(encode_params, params.quality, NULL));
    checkLwdaErrors(lwjpegEncoderParamsSetOptimizedHuffman(encode_params, params.huf, NULL));

    pidx = processArgs(params);

    checkLwdaErrors(lwjpegEncoderParamsDestroy(encode_params));
    checkLwdaErrors(lwjpegEncoderStateDestroy(encoder_state));
    checkLwdaErrors(lwjpegJpegStateDestroy(jpeg_state));
    checkLwdaErrors(lwjpegDestroy(lwjpeg_handle));

    return pidx;
}
