// Copyright (C) 2013-2020 Altera Corporation, San Jose, California, USA. All rights reserved.
// Permission is hereby granted, free of charge, to any person obtaining a copy of this
// software and associated documentation files (the "Software"), to deal in the Software
// without restriction, including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to
// whom the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.
// 
// This agreement shall be governed in all respects by the laws of the State of California and
// by the laws of the United States of America.

///////////////////////////////////////////////////////////////////////////////////
// This OpenCL application executes a 1D FFT transform on an Altera FPGA.
// The kernel is defined in a device/fft1d.cl file.  The Altera 
// Offline Compiler tool ('aoc') compiles the kernel source into a 'fft1d.aocx' 
// file containing a hardware programming image for the FPGA.  The host program 
// provides the contents of the .aocx file to the clCreateProgramWithBinary OpenCL
// API for runtime programming of the FPGA.
//
// When compiling this application, ensure that the Intel(R) FPGA SDK for OpenCL(TM)
// is properly installed.
///////////////////////////////////////////////////////////////////////////////////

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <cstring>
#include <CL/opencl.h>
#include <CL/cl_ext_intelfpga.h>
#include "AOCLUtils/aocl_utils.h"
#include "fft_config.h"

// the above header defines log of the FFT size hardcoded in the kernel
// compute N as 2^LOGN
#define N (1 << LOGN)

using namespace aocl_utils;

#define STRING_BUFFER_LEN 1024

// ACL runtime configuration
static cl_platform_id platform = NULL;
static cl_device_id device = NULL;
static cl_context context = NULL;
static cl_command_queue queue = NULL;
static cl_command_queue queue1 = NULL;
static cl_kernel kernel = NULL;
static cl_kernel kernel1 = NULL;
static cl_program program = NULL;
static cl_int status = 0;

// Control whether the emulator should be used.
static bool use_emulator = false;

// FFT operates with complex numbers - store them in a struct
typedef struct {
  double x;
  double y;
} double2;

typedef struct {
  float x;
  float y;
} float2;

// Function prototypes
bool init();
void cleanup();
static void test_fft(int iterations, int chunk_size, bool inverse);
static int coord(int iteration, int i);
static void fourier_transform_gold(bool inverse, int lognr_points, double2 * data);
static void fourier_stage(int lognr_points, double2 * data);


// Host memory buffers
float2 *h_inData, *h_outData;
double2 *h_verify;

// Device memory buffers
#if USE_SVM_API == 0
cl_mem d_inData, d_outData;
#endif /* USE_SVM_API == 0 */

// Entry point.
int main(int argc, char **argv) {
  int iterations = 2048;
  int chunksize = 512;

  Options options(argc, argv);

  // Optional argument to set the number of iterations.
  if(options.has("n")) {
    chunksize = options.get<int>("n");
  }

  // Optional argument to specify whether the emulator should be used.
  if(options.has("emulator")) {
    use_emulator = options.get<bool>("emulator");
  }

  if(!init()) {
    return false;
  }

  if (iterations <= 0) {
	printf("ERROR: Invalid number of iterations\n\nUsage: %s [-N=<#>]\n\tN: number of iterations to run (default 2000)\n", argv[0]);
	return false;
  }

  // Allocate host memory
#if USE_SVM_API == 0
  h_inData = (float2 *)alignedMalloc(sizeof(float2) * N * iterations);
  h_outData = (float2 *)alignedMalloc(sizeof(float2) * N * iterations);
#else
  h_inData = (float2 *)clSVMAlloc(context, CL_MEM_READ_WRITE, sizeof(float2) * N * iterations, 0);
  h_outData = (float2 *)clSVMAlloc(context, CL_MEM_READ_WRITE, sizeof(float2) * N * iterations, 0);
#endif /* USE_SVM_API == 0 */
  h_verify = (double2 *)alignedMalloc(sizeof(double2) * N * iterations);
  if (!(h_inData && h_outData && h_verify)) {
    printf("ERROR: Couldn't create host buffers\n");
    return false;
  }

  if ( options.has("mode")){
     if (options.get<std::string>("mode") == "normal") {
       test_fft(iterations, chunksize, false);
     } else if (options.get<std::string>("mode") == "inverse") {
       test_fft(iterations, chunksize, true);
     }
  } else{
     test_fft(iterations, chunksize, false); // test FFT transform running a sequence of iterations x 4k points transforms
     test_fft(iterations, chunksize, true); // test inverse FFT transform - same setup as above
  }

  // Free the resources allocated
  cleanup();

  return 0;
}

void test_fft(int iterations, int chunk_size, bool inverse) {
  printf("Launching");
  if (inverse) 
	printf(" inverse");
  printf(" FFT transform for %d iterations\n", iterations);
  printf(" With chunk size of %d.\n", chunk_size);

  if (iterations % chunk_size != 0) {
    printf(" ERROR: chunks must evenly divide the number of iterations.\n");
    return;
  }

#if USE_SVM_API == 1
  status = clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_WRITE,
      (void *)h_inData, sizeof(float2) * N * iterations, 0, NULL, NULL);
  checkError(status, "Failed to map input data");
#endif /* USE_SVM_API == 1 */

  // Initialize input and produce verification data
  for (int i = 0; i < iterations; i++) {
    for (int j = 0; j < N; j++) {
      h_verify[coord(i, j)].x = h_inData[coord(i, j)].x = (float)((double)rand() / (double)RAND_MAX);
      h_verify[coord(i, j)].y = h_inData[coord(i, j)].y = (float)((double)rand() / (double)RAND_MAX);
    }
  }

  // Create device buffers - assign the buffers in different banks for more efficient
  // memory access 
 
  d_inData = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float2) * N * chunk_size, NULL, &status);
  checkError(status, "Failed to allocate input device buffer\n");
  d_outData = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_CHANNEL_2_INTELFPGA, sizeof(float2) * N * chunk_size, NULL, &status);
  checkError(status, "Failed to allocate output device buffer\n");


  // Can't pass bool to device, so convert it to int
  int inverse_int = inverse;

  // Get the iterationstamp to evaluate performance
  double time = getCurrentTimestamp();

  size_t ls = N/8;
  size_t gs = chunk_size * ls;

  // Copy data from host to device
  for (int offset = 0; offset < iterations; offset += chunk_size) {
    status = clEnqueueWriteBuffer(queue1, d_inData, CL_TRUE, 0,
                                  sizeof(float2) * N * chunk_size, h_inData + offset, 0, NULL, NULL);
    checkError(status, "Failed to copy data to device");

    status = clSetKernelArg(kernel1, 0, sizeof(cl_mem), (void *)&d_inData);
    checkError(status, "Failed to set kernel1 arg 0");
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_outData);
    checkError(status, "Failed to set kernel arg 0");
    status = clSetKernelArg(kernel, 1, sizeof(cl_int), (void*)&iterations);
    checkError(status, "Failed to set kernel arg 1");
    status = clSetKernelArg(kernel, 2, sizeof(cl_int), (void*)&inverse_int);
    checkError(status, "Failed to set kernel arg 2");

    printf(inverse ? "\tInverse FFT" : "\tFFT");
    printf(" kernel initialization is complete.\n");

    size_t ls = N/8;
    size_t gs = chunk_size * ls;  // Was iterations * ls;
    status = clEnqueueNDRangeKernel(queue1, kernel1, 1, NULL, &gs, &ls, 0, NULL, NULL);
    checkError(status, "Failed to launch fetch kernel");

    // Wait for command queue to complete pending events
    status = clFinish(queue);
    checkError(status, "Failed to finish");
    status = clFinish(queue1);
    checkError(status, "Failed to finish queue1");

    //size_t ls = N/8;
    //size_t gs = chunk_size * ls;
    status = clEnqueueNDRangeKernel(queue1, kernel1, 1, NULL, &gs, &ls, 0, NULL, NULL);
    checkError(status, "Failed to launch fetch kernel");

    status = clEnqueueReadBuffer(queue, d_outData, CL_TRUE, 0,
                                 sizeof(float2) * N * chunk_size,
                                 h_outData + offset, 0, NULL, NULL);
    checkError(status, "Failed to copy data from device");
  }

  // Record execution time
  time = getCurrentTimestamp() - time;


  printf("\tProcessing time = %.4fms\n", (float)(time * 1E3));
  double gpoints_per_sec = ((double) iterations * N / time) * 1E-9;
  double gflops = 5 * N * (log((float)N)/log((float)2))/(time / iterations * 1E9);
  printf("\tThroughput = %.4f Gpoints / sec (%.4f Gflops)\n", gpoints_per_sec, gflops);

  // Pick randomly a few iterations and check SNR

  double fpga_snr = 200;
  for (int i = 0; i < iterations; i+= rand() % 20 + 1) {
    fourier_transform_gold(inverse, LOGN, h_verify + coord(i, 0));
    double mag_sum = 0;
    double noise_sum = 0;
    for (int j = 0; j < N; j++) {
      double magnitude = (double)h_verify[coord(i, j)].x * (double)h_verify[coord(i, j)].x +  
                              (double)h_verify[coord(i, j)].y * (double)h_verify[coord(i, j)].y;
      double noise = (h_verify[coord(i, j)].x - (double)h_outData[coord(i, j)].x) * (h_verify[coord(i, j)].x - (double)h_outData[coord(i, j)].x) +  
                          (h_verify[coord(i, j)].y - (double)h_outData[coord(i, j)].y) * (h_verify[coord(i, j)].y - (double)h_outData[coord(i, j)].y);

      mag_sum += magnitude;
      noise_sum += noise;
    }
    double db = 10 * log(mag_sum / noise_sum) / log(10.0);
    // find minimum SNR across all iterations
    if (db < fpga_snr) fpga_snr = db;
  }

  printf("\tSignal to noise ratio on output sample: %f --> %s\n\n", fpga_snr, fpga_snr > 125 ? "PASSED" : "FAILED");
}


/////// HELPER FUNCTIONS ///////

// provides a linear offset in the input array
int coord(int iteration, int i) {
  return iteration * N + i;
}


// Reference Fourier transform
void fourier_transform_gold(bool inverse, const int lognr_points, double2 *data) {
   const int nr_points = 1 << lognr_points;

   // The inverse requires swapping the real and imaginary component
   
   if (inverse) {
      for (int i = 0; i < nr_points; i++) {
         double tmp = data[i].x;
         data[i].x = data[i].y;
         data[i].y = tmp;;
      }
   }
   // Do a FT recursively
   fourier_stage(lognr_points, data);

   // The inverse requires swapping the real and imaginary component
   if (inverse) {
      for (int i = 0; i < nr_points; i++) {
         double tmp = data[i].x;
         data[i].x = data[i].y;
         data[i].y = tmp;;
      }
   }

   // Do the bit reversal

   double2 *temp = (double2 *)alloca(sizeof(double2) * nr_points);
   for (int i = 0; i < nr_points; i++) temp[i] = data[i];
   for (int i = 0; i < nr_points; i++) {
      int fwd = i;
      int bit_rev = 0;
      for (int j = 0; j < lognr_points; j++) {
         bit_rev <<= 1;
         bit_rev |= fwd & 1;
         fwd >>= 1;
      }
      data[i] = temp[bit_rev];
   }
}

void fourier_stage(int lognr_points, double2 *data) {
   int nr_points = 1 << lognr_points;
   if (nr_points == 1) return;
   double2 *half1 = (double2 *)alloca(sizeof(double2) * nr_points / 2);
   double2 *half2 = (double2 *)alloca(sizeof(double2) * nr_points / 2);
   for (int i = 0; i < nr_points / 2; i++) {
      half1[i] = data[2 * i];
      half2[i] = data[2 * i + 1];
   }
   fourier_stage(lognr_points - 1, half1);
   fourier_stage(lognr_points - 1, half2);
   for (int i = 0; i < nr_points / 2; i++) {
      data[i].x = half1[i].x + cos (2 * M_PI * i / nr_points) * half2[i].x + sin (2 * M_PI * i / nr_points) * half2[i].y;
      data[i].y = half1[i].y - sin (2 * M_PI * i / nr_points) * half2[i].x + cos (2 * M_PI * i / nr_points) * half2[i].y;
      data[i + nr_points / 2].x = half1[i].x - cos (2 * M_PI * i / nr_points) * half2[i].x - sin (2 * M_PI * i / nr_points) * half2[i].y;
      data[i + nr_points / 2].y = half1[i].y + sin (2 * M_PI * i / nr_points) * half2[i].x - cos (2 * M_PI * i / nr_points) * half2[i].y;
   }
}

bool init() {
  cl_int status;

  if(!setCwdToExeDir()) {
    return false;
  }

  // Get the OpenCL platform.
  if (use_emulator) {
    platform = findPlatform("Intel(R) FPGA Emulation Platform for OpenCL(TM)");
  } else {
    platform = findPlatform("Intel(R) FPGA SDK for OpenCL(TM)");
  }
  if(platform == NULL) {
    printf("ERROR: Unable to find Intel(R) FPGA OpenCL platform\n");
    return false;
  }

  // Query the available OpenCL devices.
  scoped_array<cl_device_id> devices;
  cl_uint num_devices;

  devices.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));

  // We'll just use the first device.
  device = devices[0];

  // Create the context.
  context = clCreateContext(NULL, 1, &device, &oclContextCallback, NULL, &status);
  checkError(status, "Failed to create context");

  // Create the command queue.
  queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status, "Failed to create command queue");
  queue1 = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status, "Failed to create command queue");


  // Create the program.
  std::string binary_file = getBoardBinaryFile("fft1d", device);
  printf("Using AOCX: %s\n\n", binary_file.c_str());
  program = createProgramFromBinary(context, binary_file.c_str(), &device, 1);

  // Build the program that was just created.
  status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
  checkError(status, "Failed to build program");

  // Create the kernel - name passed in here must match kernel name in the
  // original CL file, that was compiled into an AOCX file using the AOC tool
  kernel = clCreateKernel(program, "fft1d", &status);
  checkError(status, "Failed to create kernel");

  kernel1 = clCreateKernel(program, "fetch", &status);
  checkError(status, "Failed to create fetch kernel");

#if USE_SVM_API == 1
  cl_device_svm_capabilities caps = 0;

  status = clGetDeviceInfo(
    device,
    CL_DEVICE_SVM_CAPABILITIES,
    sizeof(cl_device_svm_capabilities),
    &caps,
    0
  );
  checkError(status, "Failed to get device info");

  if (!(caps & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER)) {
    printf("The host was compiled with USE_SVM_API, however the device currently being targeted does not support SVM.\n");
    // Free the resources allocated
    cleanup();
    return false;
  }
#endif /* USE_SVM_API == 1 */
  return true;
}

// Free the resources allocated during initialization
void cleanup() {
  if(kernel) 
    clReleaseKernel(kernel);  
  if(kernel1) 
    clReleaseKernel(kernel1);  
  if(program) 
    clReleaseProgram(program);
  if(queue) 
    clReleaseCommandQueue(queue);
  if(queue1) 
    clReleaseCommandQueue(queue1);
  if (h_verify)
	alignedFree(h_verify);
#if USE_SVM_API == 0
  if(h_inData)
	alignedFree(h_inData);
  if (h_outData)
	alignedFree(h_outData);
  if (d_inData)
	clReleaseMemObject(d_inData);
  if (d_outData) 
	clReleaseMemObject(d_outData);
#else
  if (h_inData)
    clSVMFree(context, h_inData);
  if (h_outData)
    clSVMFree(context, h_outData);
#endif /* USE_SVM_API == 0 */
  if(context)
    clReleaseContext(context);
}



