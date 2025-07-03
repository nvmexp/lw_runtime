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

// This sample needs at least LWCA 10.0. It demonstrates usages of the lwJPEG
// library lwJPEG supports single and multiple image(batched) decode. Multiple
// images can be decoded using the API for batch mode

#include <lwda_runtime_api.h>
#include "helper_lwJPEG.hxx"

int dev_malloc(void **p, size_t s) { return (int)lwdaMalloc(p, s); }

int dev_free(void *p) { return (int)lwdaFree(p); }

int host_malloc(void** p, size_t s, unsigned int f) { return (int)lwdaHostAlloc(p, s, f); }

int host_free(void* p) { return (int)lwdaFreeHost(p); }

typedef std::vector<std::string> FileNames;
typedef std::vector<std::vector<char> > FileData;

struct decode_params_t {
  std::string input_dir;
  int batch_size;
  int total_images;
  int dev;
  int warmup;

  lwjpegJpegState_t lwjpeg_state;
  lwjpegHandle_t lwjpeg_handle;
  lwdaStream_t stream;

  // used with decoupled API
  lwjpegJpegState_t lwjpeg_decoupled_state;
  lwjpegBufferPinned_t pinned_buffers[2]; // 2 buffers for pipelining
  lwjpegBufferDevice_t device_buffer;
  lwjpegJpegStream_t  jpeg_streams[2]; //  2 streams for pipelining
  lwjpegDecodeParams_t lwjpeg_decode_params;
  lwjpegJpegDecoder_t lwjpeg_decoder;

  lwjpegOutputFormat_t fmt;
  bool write_decoded;
  std::string output_dir;

  bool pipelined;
  bool batched;
};

int read_next_batch(FileNames &image_names, int batch_size,
                    FileNames::iterator &lwr_iter, FileData &raw_data,
                    std::vector<size_t> &raw_len, FileNames &lwrrent_names) {
  int counter = 0;

  while (counter < batch_size) {
    if (lwr_iter == image_names.end()) {
      std::cerr << "Image list is too short to fill the batch, adding files "
                   "from the beginning of the image list"
                << std::endl;
      lwr_iter = image_names.begin();
    }

    if (image_names.size() == 0) {
      std::cerr << "No valid images left in the input list, exit" << std::endl;
      return EXIT_FAILURE;
    }

    // Read an image from disk.
    std::ifstream input(lwr_iter->c_str(),
                        std::ios::in | std::ios::binary | std::ios::ate);
    if (!(input.is_open())) {
      std::cerr << "Cannot open image: " << *lwr_iter
                << ", removing it from image list" << std::endl;
      image_names.erase(lwr_iter);
      continue;
    }

    // Get the size
    std::streamsize file_size = input.tellg();
    input.seekg(0, std::ios::beg);
    // resize if buffer is too small
    if (raw_data[counter].size() < file_size) {
      raw_data[counter].resize(file_size);
    }
    if (!input.read(raw_data[counter].data(), file_size)) {
      std::cerr << "Cannot read from file: " << *lwr_iter
                << ", removing it from image list" << std::endl;
      image_names.erase(lwr_iter);
      continue;
    }
    raw_len[counter] = file_size;

    lwrrent_names[counter] = *lwr_iter;

    counter++;
    lwr_iter++;
  }
  return EXIT_SUCCESS;
}

// prepare buffers for RGBi output format
int prepare_buffers(FileData &file_data, std::vector<size_t> &file_len,
                    std::vector<int> &img_width, std::vector<int> &img_height,
                    std::vector<lwjpegImage_t> &ibuf,
                    std::vector<lwjpegImage_t> &isz, FileNames &lwrrent_names,
                    decode_params_t &params) {
  int widths[LWJPEG_MAX_COMPONENT];
  int heights[LWJPEG_MAX_COMPONENT];
  int channels;
  lwjpegChromaSubsampling_t subsampling;

  for (int i = 0; i < file_data.size(); i++) {
    checkLwdaErrors(lwjpegGetImageInfo(
        params.lwjpeg_handle, (unsigned char *)file_data[i].data(), file_len[i],
        &channels, &subsampling, widths, heights));

    img_width[i] = widths[0];
    img_height[i] = heights[0];

    std::cout << "Processing: " << lwrrent_names[i] << std::endl;
    std::cout << "Image is " << channels << " channels." << std::endl;
    for (int c = 0; c < channels; c++) {
      std::cout << "Channel #" << c << " size: " << widths[c] << " x "
                << heights[c] << std::endl;
    }

    switch (subsampling) {
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
        return EXIT_FAILURE;
    }

    int mul = 1;
    // in the case of interleaved RGB output, write only to single channel, but
    // 3 samples at once
    if (params.fmt == LWJPEG_OUTPUT_RGBI || params.fmt == LWJPEG_OUTPUT_BGRI) {
      channels = 1;
      mul = 3;
    }
    // in the case of rgb create 3 buffers with sizes of original image
    else if (params.fmt == LWJPEG_OUTPUT_RGB ||
             params.fmt == LWJPEG_OUTPUT_BGR) {
      channels = 3;
      widths[1] = widths[2] = widths[0];
      heights[1] = heights[2] = heights[0];
    }

    // realloc output buffer if required
    for (int c = 0; c < channels; c++) {
      int aw = mul * widths[c];
      int ah = heights[c];
      int sz = aw * ah;
      ibuf[i].pitch[c] = aw;
      if (sz > isz[i].pitch[c]) {
        if (ibuf[i].channel[c]) {
          checkLwdaErrors(lwdaFree(ibuf[i].channel[c]));
        }
        checkLwdaErrors(lwdaMalloc(&ibuf[i].channel[c], sz));
        isz[i].pitch[c] = sz;
      }
    }
  }
  return EXIT_SUCCESS;
}

void create_decoupled_api_handles(decode_params_t& params){

  checkLwdaErrors(lwjpegDecoderCreate(params.lwjpeg_handle, LWJPEG_BACKEND_DEFAULT, &params.lwjpeg_decoder));
  checkLwdaErrors(lwjpegDecoderStateCreate(params.lwjpeg_handle, params.lwjpeg_decoder, &params.lwjpeg_decoupled_state));   
  
  checkLwdaErrors(lwjpegBufferPinnedCreate(params.lwjpeg_handle, NULL, &params.pinned_buffers[0]));
  checkLwdaErrors(lwjpegBufferPinnedCreate(params.lwjpeg_handle, NULL, &params.pinned_buffers[1]));
  checkLwdaErrors(lwjpegBufferDeviceCreate(params.lwjpeg_handle, NULL, &params.device_buffer));
  
  checkLwdaErrors(lwjpegJpegStreamCreate(params.lwjpeg_handle, &params.jpeg_streams[0]));
  checkLwdaErrors(lwjpegJpegStreamCreate(params.lwjpeg_handle, &params.jpeg_streams[1]));

  checkLwdaErrors(lwjpegDecodeParamsCreate(params.lwjpeg_handle, &params.lwjpeg_decode_params));
}

void destroy_decoupled_api_handles(decode_params_t& params){  

  checkLwdaErrors(lwjpegDecodeParamsDestroy(params.lwjpeg_decode_params));
  checkLwdaErrors(lwjpegJpegStreamDestroy(params.jpeg_streams[0]));
  checkLwdaErrors(lwjpegJpegStreamDestroy(params.jpeg_streams[1]));
  checkLwdaErrors(lwjpegBufferPinnedDestroy(params.pinned_buffers[0]));
  checkLwdaErrors(lwjpegBufferPinnedDestroy(params.pinned_buffers[1]));
  checkLwdaErrors(lwjpegBufferDeviceDestroy(params.device_buffer));
  checkLwdaErrors(lwjpegJpegStateDestroy(params.lwjpeg_decoupled_state));  
  checkLwdaErrors(lwjpegDecoderDestroy(params.lwjpeg_decoder));
}

void release_buffers(std::vector<lwjpegImage_t> &ibuf) {
  for (int i = 0; i < ibuf.size(); i++) {
    for (int c = 0; c < LWJPEG_MAX_COMPONENT; c++)
      if (ibuf[i].channel[c]) checkLwdaErrors(lwdaFree(ibuf[i].channel[c]));
  }
}

int decode_images(const FileData &img_data, const std::vector<size_t> &img_len,
                  std::vector<lwjpegImage_t> &out, decode_params_t &params,
                  double &time) {
  checkLwdaErrors(lwdaStreamSynchronize(params.stream));
  lwdaEvent_t startEvent = NULL, stopEvent = NULL;
  float loopTime = 0; 
  
  checkLwdaErrors(lwdaEventCreate(&startEvent, lwdaEventBlockingSync));
  checkLwdaErrors(lwdaEventCreate(&stopEvent, lwdaEventBlockingSync));

  if (!params.batched) {
    if (!params.pipelined)  // decode one image at a time
    {
      checkLwdaErrors(lwdaEventRecord(startEvent, params.stream));
      for (int i = 0; i < params.batch_size; i++) {
        checkLwdaErrors(lwjpegDecode(params.lwjpeg_handle, params.lwjpeg_state,
                                     (const unsigned char *)img_data[i].data(),
                                     img_len[i], params.fmt, &out[i],
                                     params.stream));
      }
      checkLwdaErrors(lwdaEventRecord(stopEvent, params.stream));
    } else {
      // use de-coupled API in pipelined mode
      checkLwdaErrors(lwdaEventRecord(startEvent, params.stream));
      checkLwdaErrors(lwjpegStateAttachDeviceBuffer(params.lwjpeg_decoupled_state, params.device_buffer));
      int buffer_index = 0;
      checkLwdaErrors(lwjpegDecodeParamsSetOutputFormat(params.lwjpeg_decode_params, params.fmt));
      for (int i = 0; i < params.batch_size; i++) {
      checkLwdaErrors(
          lwjpegJpegStreamParse(params.lwjpeg_handle, (const unsigned char *)img_data[i].data(), img_len[i], 
          0, 0, params.jpeg_streams[buffer_index]));
                                
      checkLwdaErrors(lwjpegStateAttachPinnedBuffer(params.lwjpeg_decoupled_state,
          params.pinned_buffers[buffer_index]));
      
      checkLwdaErrors(lwjpegDecodeJpegHost(params.lwjpeg_handle, params.lwjpeg_decoder, params.lwjpeg_decoupled_state, 
          params.lwjpeg_decode_params, params.jpeg_streams[buffer_index]));

      checkLwdaErrors(lwdaStreamSynchronize(params.stream));

      checkLwdaErrors(lwjpegDecodeJpegTransferToDevice(params.lwjpeg_handle, params.lwjpeg_decoder, params.lwjpeg_decoupled_state,
          params.jpeg_streams[buffer_index], params.stream));

      buffer_index = 1 - buffer_index; // switch pinned buffer in pipeline mode to avoid an extra sync

      checkLwdaErrors(lwjpegDecodeJpegDevice(params.lwjpeg_handle, params.lwjpeg_decoder, params.lwjpeg_decoupled_state,
          &out[i], params.stream));

      }
      checkLwdaErrors(lwdaEventRecord(stopEvent, params.stream));
    }
  } else {
    std::vector<const unsigned char *> raw_inputs;
    for (int i = 0; i < params.batch_size; i++) {
      raw_inputs.push_back((const unsigned char *)img_data[i].data());
    }

    checkLwdaErrors(lwdaEventRecord(startEvent, params.stream));
    checkLwdaErrors(lwjpegDecodeBatched(
        params.lwjpeg_handle, params.lwjpeg_state, raw_inputs.data(),
        img_len.data(), out.data(), params.stream));
    checkLwdaErrors(lwdaEventRecord(stopEvent, params.stream));
  
  }
  checkLwdaErrors(lwdaEventSynchronize(stopEvent));
  checkLwdaErrors(lwdaEventElapsedTime(&loopTime, startEvent, stopEvent));
  time = static_cast<double>(loopTime);

  return EXIT_SUCCESS;
}

int write_images(std::vector<lwjpegImage_t> &iout, std::vector<int> &widths,
                 std::vector<int> &heights, decode_params_t &params,
                 FileNames &filenames) {
  for (int i = 0; i < params.batch_size; i++) {
    // Get the file name, without extension.
    // This will be used to rename the output file.
    size_t position = filenames[i].rfind("/");
    std::string sFileName =
        (std::string::npos == position)
            ? filenames[i]
            : filenames[i].substr(position + 1, filenames[i].size());
    position = sFileName.rfind(".");
    sFileName = (std::string::npos == position) ? sFileName
                                                : sFileName.substr(0, position);
    std::string fname(params.output_dir + "/" + sFileName + ".bmp");

    int err;
    if (params.fmt == LWJPEG_OUTPUT_RGB || params.fmt == LWJPEG_OUTPUT_BGR) {
      err = writeBMP(fname.c_str(), iout[i].channel[0], iout[i].pitch[0],
                     iout[i].channel[1], iout[i].pitch[1], iout[i].channel[2],
                     iout[i].pitch[2], widths[i], heights[i]);
    } else if (params.fmt == LWJPEG_OUTPUT_RGBI ||
               params.fmt == LWJPEG_OUTPUT_BGRI) {
      // Write BMP from interleaved data
      err = writeBMPi(fname.c_str(), iout[i].channel[0], iout[i].pitch[0],
                      widths[i], heights[i]);
    }
    if (err) {
      std::cout << "Cannot write output file: " << fname << std::endl;
      return EXIT_FAILURE;
    }
    std::cout << "Done writing decoded image to file: " << fname << std::endl;
  }
}

double process_images(FileNames &image_names, decode_params_t &params,
                      double &total) {
  // vector for storing raw files and file lengths
  FileData file_data(params.batch_size);
  std::vector<size_t> file_len(params.batch_size);
  FileNames lwrrent_names(params.batch_size);
  std::vector<int> widths(params.batch_size);
  std::vector<int> heights(params.batch_size);
  // we wrap over image files to process total_images of files
  FileNames::iterator file_iter = image_names.begin();

  // stream for decoding
  checkLwdaErrors(
      lwdaStreamCreateWithFlags(&params.stream, lwdaStreamNonBlocking));

  int total_processed = 0;

  // output buffers
  std::vector<lwjpegImage_t> iout(params.batch_size);
  // output buffer sizes, for colwenience
  std::vector<lwjpegImage_t> isz(params.batch_size);

  for (int i = 0; i < iout.size(); i++) {
    for (int c = 0; c < LWJPEG_MAX_COMPONENT; c++) {
      iout[i].channel[c] = NULL;
      iout[i].pitch[c] = 0;
      isz[i].pitch[c] = 0;
    }
  }

  double test_time = 0;
  int warmup = 0;
  while (total_processed < params.total_images) {
    if (read_next_batch(image_names, params.batch_size, file_iter, file_data,
                        file_len, lwrrent_names))
      return EXIT_FAILURE;

    if (prepare_buffers(file_data, file_len, widths, heights, iout, isz,
                        lwrrent_names, params))
      return EXIT_FAILURE;

    double time;
    if (decode_images(file_data, file_len, iout, params, time))
      return EXIT_FAILURE;
    if (warmup < params.warmup) {
      warmup++;
    } else {
      total_processed += params.batch_size;
      test_time += time;
    }

    if (params.write_decoded)
      write_images(iout, widths, heights, params, lwrrent_names);
  }
  total = test_time;

  release_buffers(iout);

  checkLwdaErrors(lwdaStreamDestroy(params.stream));

  return EXIT_SUCCESS;
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

int main(int argc, const char *argv[]) {
  int pidx;

  if ((pidx = findParamIndex(argv, argc, "-h")) != -1 ||
      (pidx = findParamIndex(argv, argc, "--help")) != -1) {
    std::cout << "Usage: " << argv[0]
              << " -i images_dir [-b batch_size] [-t total_images] [-device= "
                 "device_id] [-w warmup_iterations] [-o output_dir] "
                 "[-pipelined] [-batched] [-fmt output_format]\n";
    std::cout << "Parameters: " << std::endl;
    std::cout << "\timages_dir\t:\tPath to single image or directory of images"
              << std::endl;
    std::cout << "\tbatch_size\t:\tDecode images from input by batches of "
                 "specified size"
              << std::endl;
    std::cout << "\ttotal_images\t:\tDecode this much images, if there are "
                 "less images \n"
              << "\t\t\t\t\tin the input than total images, decoder will loop "
                 "over the input"
              << std::endl;
    std::cout << "\tdevice_id\t:\tWhich device to use for decoding"
              << std::endl;
    std::cout << "\twarmup_iterations\t:\tRun this amount of batches first "
                 "without measuring performance"
              << std::endl;
    std::cout
        << "\toutput_dir\t:\tWrite decoded images as BMPs to this directory"
        << std::endl;
    std::cout << "\tpipelined\t:\tUse decoding in phases" << std::endl;
    std::cout << "\tbatched\t\t:\tUse batched interface" << std::endl;
    std::cout << "\toutput_format\t:\tlwJPEG output format for decoding. One "
                 "of [rgb, rgbi, bgr, bgri, yuv, y, unchanged]"
              << std::endl;
    return EXIT_SUCCESS;
  }

  decode_params_t params;

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

  params.batch_size = 1;
  if ((pidx = findParamIndex(argv, argc, "-b")) != -1) {
    params.batch_size = std::atoi(argv[pidx + 1]);
  }

  params.total_images = -1;
  if ((pidx = findParamIndex(argv, argc, "-t")) != -1) {
    params.total_images = std::atoi(argv[pidx + 1]);
  }

  params.dev = 0;
  params.dev = findLwdaDevice(argc, argv);

  params.warmup = 0;
  if ((pidx = findParamIndex(argv, argc, "-w")) != -1) {
    params.warmup = std::atoi(argv[pidx + 1]);
  }

  params.batched = false;
  if ((pidx = findParamIndex(argv, argc, "-batched")) != -1) {
    params.batched = true;
  }

  params.pipelined = false;
  if ((pidx = findParamIndex(argv, argc, "-pipelined")) != -1) {
    params.pipelined = true;
  }

  params.fmt = LWJPEG_OUTPUT_RGB;
  if ((pidx = findParamIndex(argv, argc, "-fmt")) != -1) {
    std::string sfmt = argv[pidx + 1];
    if (sfmt == "rgb")
      params.fmt = LWJPEG_OUTPUT_RGB;
    else if (sfmt == "bgr")
      params.fmt = LWJPEG_OUTPUT_BGR;
    else if (sfmt == "rgbi")
      params.fmt = LWJPEG_OUTPUT_RGBI;
    else if (sfmt == "bgri")
      params.fmt = LWJPEG_OUTPUT_BGRI;
    else if (sfmt == "yuv")
      params.fmt = LWJPEG_OUTPUT_YUV;
    else if (sfmt == "y")
      params.fmt = LWJPEG_OUTPUT_Y;
    else if (sfmt == "unchanged")
      params.fmt = LWJPEG_OUTPUT_UNCHANGED;
    else {
      std::cout << "Unknown format: " << sfmt << std::endl;
      return EXIT_FAILURE;
    }
  }

  params.write_decoded = false;
  if ((pidx = findParamIndex(argv, argc, "-o")) != -1) {
    params.output_dir = argv[pidx + 1];
    if (params.fmt != LWJPEG_OUTPUT_RGB && params.fmt != LWJPEG_OUTPUT_BGR &&
        params.fmt != LWJPEG_OUTPUT_RGBI && params.fmt != LWJPEG_OUTPUT_BGRI) {
      std::cout << "We can write ony BMPs, which require output format be "
                   "either RGB/BGR or RGBi/BGRi"
                << std::endl;
      return EXIT_FAILURE;
    }
    params.write_decoded = true;
  }

  lwdaDeviceProp props;
  checkLwdaErrors(lwdaGetDeviceProperties(&props, params.dev));

  printf("Using GPU %d (%s, %d SMs, %d th/SM max, CC %d.%d, ECC %s)\n",
         params.dev, props.name, props.multiProcessorCount,
         props.maxThreadsPerMultiProcessor, props.major, props.minor,
         props.ECCEnabled ? "on" : "off");

  lwjpegDevAllocator_t dev_allocator = {&dev_malloc, &dev_free};
  lwjpegPinnedAllocator_t pinned_allocator ={&host_malloc, &host_free};
  int flags = 0;
  checkLwdaErrors(lwjpegCreateEx(LWJPEG_BACKEND_DEFAULT, &dev_allocator,
                                &pinned_allocator,flags,  &params.lwjpeg_handle));

  checkLwdaErrors(
      lwjpegJpegStateCreate(params.lwjpeg_handle, &params.lwjpeg_state));
  checkLwdaErrors(
      lwjpegDecodeBatchedInitialize(params.lwjpeg_handle, params.lwjpeg_state,
                                    params.batch_size, 1, params.fmt));

  if(params.pipelined ){
    create_decoupled_api_handles(params);
  }
  // read source images
  FileNames image_names;
  readInput(params.input_dir, image_names);

  if (params.total_images == -1) {
    params.total_images = image_names.size();
  } else if (params.total_images % params.batch_size) {
    params.total_images =
        ((params.total_images) / params.batch_size) * params.batch_size;
    std::cout << "Changing total_images number to " << params.total_images
              << " to be multiple of batch_size - " << params.batch_size
              << std::endl;
  }

  std::cout << "Decoding images in directory: " << params.input_dir
            << ", total " << params.total_images << ", batchsize "
            << params.batch_size << std::endl;

  double total;
  if (process_images(image_names, params, total)) return EXIT_FAILURE;
  std::cout << "Total decoding time: " << total << std::endl;
  std::cout << "Avg decoding time per image: " << total / params.total_images
            << std::endl;
  std::cout << "Avg images per sec: " << params.total_images / total
            << std::endl;
  std::cout << "Avg decoding time per batch: "
            << total / ((params.total_images + params.batch_size - 1) /
                        params.batch_size)
            << std::endl;

  if(params.pipelined ){ 
    destroy_decoupled_api_handles(params);
  }

  checkLwdaErrors(lwjpegJpegStateDestroy(params.lwjpeg_state));
  checkLwdaErrors(lwjpegDestroy(params.lwjpeg_handle));

  return EXIT_SUCCESS;
}
