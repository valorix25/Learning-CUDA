/**
 * Copyright 1993-2017 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

////////////////////////////////////////////////////////////////////////////////
// These are MACA Helper functions for initialization and error checking

#ifndef COMMON_HELPER_MACA_H_
#define COMMON_HELPER_MACA_H_

#pragma once

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "helper_string.h"

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif

#ifndef __RUNTIME_TYPES_H__
#define __RUNTIME_TYPES_H__ 
#endif

// Note, it is required that your SDK sample to include the proper header
// files, please refer the MACA examples for examples of the needed MACA
// headers, which may change depending on which MACA functions are used.

// MACA Runtime error messages
#ifdef __RUNTIME_TYPES_H__
static const char *_MACAGetErrorEnum(mcError_t error) {
  return mcGetErrorName(error);
}
#endif

#ifdef MACA_RUNTIME_API
// MACA Runtime API errors
static const char *_MACAGetErrorEnum(CUresult error) {
  static char unknown[] = "<unknown>";
  const char *ret = NULL;
  cuGetErrorName(error, &ret);
  return ret ? ret : unknown;
}
#endif

#ifdef MCBLAS_API_H_
// mcBLAS API errors
static const char *_MACAGetErrorEnum(mcblasStatus_t error) {
  switch (error) {
    case MCBLAS_STATUS_SUCCESS:
      return "MCBLAS_STATUS_SUCCESS";

    case MCBLAS_STATUS_NOT_INITIALIZED:
      return "MCBLAS_STATUS_NOT_INITIALIZED";

    case MCBLAS_STATUS_ALLOC_FAILED:
      return "MCBLAS_STATUS_ALLOC_FAILED";

    case MCBLAS_STATUS_INVALID_VALUE:
      return "MCBLAS_STATUS_INVALID_VALUE";

    case MCBLAS_STATUS_ARCH_MISMATCH:
      return "MCBLAS_STATUS_ARCH_MISMATCH";

    case MCBLAS_STATUS_MAPPING_ERROR:
      return "MCBLAS_STATUS_MAPPING_ERROR";

    case MCBLAS_STATUS_EXECUTION_FAILED:
      return "MCBLAS_STATUS_EXECUTION_FAILED";

    case MCBLAS_STATUS_INTERNAL_ERROR:
      return "MCBLAS_STATUS_INTERNAL_ERROR";

    case MCBLAS_STATUS_NOT_SUPPORTED:
      return "MCBLAS_STATUS_NOT_SUPPORTED";

    case MCBLAS_STATUS_LICENSE_ERROR:
      return "MCBLAS_STATUS_LICENSE_ERROR";
  }

  return "<unknown>";
}
#endif

#ifdef _MCFFT_H_
// mcFFT API errors
static const char *_MACAGetErrorEnum(mcfftResult error) {
  switch (error) {
    case MCFFT_SUCCESS:
      return "MCFFT_SUCCESS";

    case MCFFT_INVALID_PLAN:
      return "MCFFT_INVALID_PLAN";

    case MCFFT_ALLOC_FAILED:
      return "MCFFT_ALLOC_FAILED";

    case MCFFT_INVALID_TYPE:
      return "MCFFT_INVALID_TYPE";

    case MCFFT_INVALID_VALUE:
      return "MCFFT_INVALID_VALUE";

    case MCFFT_INTERNAL_ERROR:
      return "MCFFT_INTERNAL_ERROR";

    case MCFFT_EXEC_FAILED:
      return "MCFFT_EXEC_FAILED";

    case MCFFT_SETUP_FAILED:
      return "MCFFT_SETUP_FAILED";

    case MCFFT_INVALID_SIZE:
      return "MCFFT_INVALID_SIZE";

    case MCFFT_UNALIGNED_DATA:
      return "MCFFT_UNALIGNED_DATA";

    case MCFFT_INCOMPLETE_PARAMETER_LIST:
      return "MCFFT_INCOMPLETE_PARAMETER_LIST";

    case MCFFT_INVALID_DEVICE:
      return "MCFFT_INVALID_DEVICE";

    case MCFFT_PARSE_ERROR:
      return "MCFFT_PARSE_ERROR";

    case MCFFT_NO_WORKSPACE:
      return "MCFFT_NO_WORKSPACE";

    case MCFFT_NOT_IMPLEMENTED:
      return "MCFFT_NOT_IMPLEMENTED";

    case MCFFT_LICENSE_ERROR:
      return "MCFFT_LICENSE_ERROR";

    case MCFFT_NOT_SUPPORTED:
      return "MCFFT_NOT_SUPPORTED";
  }

  return "<unknown>";
}
#endif

#ifdef MCSPARSEAPI
// mcSPARSE API errors
static const char *_MACAGetErrorEnum(mcsparseStatus_t error) {
  switch (error) {
    case MCSPARSE_STATUS_SUCCESS:
      return "MCSPARSE_STATUS_SUCCESS";

    case MCSPARSE_STATUS_NOT_INITIALIZED:
      return "MCSPARSE_STATUS_NOT_INITIALIZED";

    case MCSPARSE_STATUS_ALLOC_FAILED:
      return "MCSPARSE_STATUS_ALLOC_FAILED";

    case MCSPARSE_STATUS_INVALID_VALUE:
      return "MCSPARSE_STATUS_INVALID_VALUE";

    case MCSPARSE_STATUS_ARCH_MISMATCH:
      return "MCSPARSE_STATUS_ARCH_MISMATCH";

    case MCSPARSE_STATUS_MAPPING_ERROR:
      return "MCSPARSE_STATUS_MAPPING_ERROR";

    case MCSPARSE_STATUS_EXECUTION_FAILED:
      return "MCSPARSE_STATUS_EXECUTION_FAILED";

    case MCSPARSE_STATUS_INTERNAL_ERROR:
      return "MCSPARSE_STATUS_INTERNAL_ERROR";

    case MCSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
      return "MCSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
  }

  return "<unknown>";
}
#endif

#ifdef MCSOLVER_COMMON_H_
// cuSOLVER API errors
static const char *_MACAGetErrorEnum(mcsolverStatus_t error) {
  switch (error) {
    case MCSOLVER_STATUS_SUCCESS:
      return "MCSOLVER_STATUS_SUCCESS";
    case MCSOLVER_STATUS_NOT_INITIALIZED:
      return "MCSOLVER_STATUS_NOT_INITIALIZED";
    case MCSOLVER_STATUS_ALLOC_FAILED:
      return "MCSOLVER_STATUS_ALLOC_FAILED";
    case MCSOLVER_STATUS_INVALID_VALUE:
      return "MCSOLVER_STATUS_INVALID_VALUE";
    case MCSOLVER_STATUS_ARCH_MISMATCH:
      return "MCSOLVER_STATUS_ARCH_MISMATCH";
    case MCSOLVER_STATUS_MAPPING_ERROR:
      return "MCSOLVER_STATUS_MAPPING_ERROR";
    case MCSOLVER_STATUS_EXECUTION_FAILED:
      return "MCSOLVER_STATUS_EXECUTION_FAILED";
    case MCSOLVER_STATUS_INTERNAL_ERROR:
      return "MCSOLVER_STATUS_INTERNAL_ERROR";
    case MCSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
      return "MCSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    case MCSOLVER_STATUS_NOT_SUPPORTED:
      return "MCSOLVER_STATUS_NOT_SUPPORTED ";
    case MCSOLVER_STATUS_ZERO_PIVOT:
      return "MCSOLVER_STATUS_ZERO_PIVOT";
    case MCSOLVER_STATUS_INVALID_LICENSE:
      return "MCSOLVER_STATUS_INVALID_LICENSE";
  }

  return "<unknown>";
}
#endif

#ifdef MCRAND_H_
// mcRAND API errors
static const char *_MACAGetErrorEnum(mcrandStatus_t error) {
  switch (error) {
    case MCRAND_STATUS_SUCCESS:
      return "MCRAND_STATUS_SUCCESS";

    case MCRAND_STATUS_VERSION_MISMATCH:
      return "MCRAND_STATUS_VERSION_MISMATCH";

    case MCRAND_STATUS_NOT_INITIALIZED:
      return "MCRAND_STATUS_NOT_INITIALIZED";

    case MCRAND_STATUS_ALLOCATION_FAILED:
      return "MCRAND_STATUS_ALLOCATION_FAILED";

    case MCRAND_STATUS_TYPE_ERROR:
      return "MCRAND_STATUS_TYPE_ERROR";

    case MCRAND_STATUS_OUT_OF_RANGE:
      return "MCRAND_STATUS_OUT_OF_RANGE";

    case MCRAND_STATUS_LENGTH_NOT_MULTIPLE:
      return "MCRAND_STATUS_LENGTH_NOT_MULTIPLE";

    case MCRAND_STATUS_DOUBLE_PRECISION_REQUIRED:
      return "MCRAND_STATUS_DOUBLE_PRECISION_REQUIRED";

    case MCRAND_STATUS_LAUNCH_FAILURE:
      return "MCRAND_STATUS_LAUNCH_FAILURE";

    case MCRAND_STATUS_PREEXISTING_FAILURE:
      return "MCRAND_STATUS_PREEXISTING_FAILURE";

    case MCRAND_STATUS_INITIALIZATION_FAILED:
      return "MCRAND_STATUS_INITIALIZATION_FAILED";

    case MCRAND_STATUS_ARCH_MISMATCH:
      return "MCRAND_STATUS_ARCH_MISMATCH";

    case MCRAND_STATUS_INTERNAL_ERROR:
      return "MCRAND_STATUS_INTERNAL_ERROR";
  }

  return "<unknown>";
}
#endif

#ifdef MXJPEGAPI
// mxJPEG API errors
static const char *_MACAGetErrorEnum(mxjpegStatus_t error) {
  switch (error) {
    case MXJPEG_STATUS_SUCCESS:
      return "MXJPEG_STATUS_SUCCESS";

    case MXJPEG_STATUS_NOT_INITIALIZED:
      return "MXJPEG_STATUS_NOT_INITIALIZED";

    case MXJPEG_STATUS_INVALID_PARAMETER:
      return "MXJPEG_STATUS_INVALID_PARAMETER";

    case MXJPEG_STATUS_BAD_JPEG:
      return "MXJPEG_STATUS_BAD_JPEG";

    case MXJPEG_STATUS_JPEG_NOT_SUPPORTED:
      return "MXJPEG_STATUS_JPEG_NOT_SUPPORTED";

    case MXJPEG_STATUS_ALLOCATOR_FAILURE:
      return "MXJPEG_STATUS_ALLOCATOR_FAILURE";

    case MXJPEG_STATUS_EXECUTION_FAILED:
      return "MXJPEG_STATUS_EXECUTION_FAILED";

    case MXJPEG_STATUS_ARCH_MISMATCH:
      return "MXJPEG_STATUS_ARCH_MISMATCH";

    case MXJPEG_STATUS_INTERNAL_ERROR:
      return "MXJPEG_STATUS_INTERNAL_ERROR";
  }

  return "<unknown>";
}
#endif


template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
  if (result) {
    fprintf(stderr, "MACA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), _MACAGetErrorEnum(result), func);
    exit(EXIT_FAILURE);
  }
}

//#ifdef __RUNTIME_TYPES_H__
// This will output the proper MACA error strings in the event
// that a MACA host call returns an error
#define checkMacaErrors(val) check((val), #val, __FILE__, __LINE__)

// This will output the proper error string when calling mcGetLastError
#define getLastMacaError(msg) __getLastMacaError(msg, __FILE__, __LINE__)

inline void __getLastMacaError(const char *errorMessage, const char *file,
                               const int line) {
  mcError_t err = mcGetLastError();

  if (mcSuccess != err) {
    fprintf(stderr,
            "%s(%i) : getLastMacaError() MACA error :"
            " %s : (%d) %s.\n",
            file, line, errorMessage, static_cast<int>(err),
            mcGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

// This will only print the proper error string when calling mcGetLastError
// but not exit program incase error detected.
#define printLastMacaError(msg) __printLastMacaError(msg, __FILE__, __LINE__)

inline void __printLastMacaError(const char *errorMessage, const char *file,
                                 const int line) {
  mcError_t err = mcGetLastError();

  if (mcSuccess != err) {
    fprintf(stderr,
            "%s(%i) : getLastMacaError() MACA error :"
            " %s : (%d) %s.\n",
            file, line, errorMessage, static_cast<int>(err),
            mcGetErrorString(err));
  }
}
#endif

#ifndef MAX
#define MAX(a, b) (a > b ? a : b)
#endif

// Float To Int conversion
inline int ftoi(float value) {
  return (value >= 0 ? static_cast<int>(value + 0.5)
                     : static_cast<int>(value - 0.5));
}

// Beginning of GPU Architecture definitions
inline int _ConvertAPVer2Cores(int major, int minor) {
  // Defines for GPU Architecture types (using the AP version to determine
  // the # of cores per AP
  typedef struct {
    int AP;  // 0xMm (hexidecimal notation), M = AP Major version,
    // and m = AP minor version
    int Cores;
  } sAPtoCores;

  sAPtoCores nGpuArchCoresPerAP[] = {
      {0x30, 192},
      {0x32, 192},
      {0x35, 192},
      {0x37, 192},
      {0x50, 128},
      {0x52, 128},
      {0x53, 128},
      {0x60,  64},
      {0x61, 128},
      {0x62, 128},
      {0x70,  64},
      {0x72,  64},
      {0x75,  64},
      {0x80,  64},
      {0x86, 128},
      {0x87, 128},
      {-1, -1}};

  int index = 0;

  while (nGpuArchCoresPerAP[index].AP != -1) {
    if (nGpuArchCoresPerAP[index].AP == ((major << 4) + minor)) {
      return nGpuArchCoresPerAP[index].Cores;
    }

    index++;
  }

  // If we don't find the values, we default use the previous one
  // to run properly
  printf(
      "MapAPtoCores for AP %d.%d is undefined."
      "  Default to use %d Cores/AP\n",
      major, minor, nGpuArchCoresPerAP[index - 1].Cores);
  return nGpuArchCoresPerAP[index - 1].Cores;
}

inline const char* _ConvertAPVer2ArchName(int major, int minor) {
  // Defines for GPU Architecture types (using the AP version to determine
  // the GPU Arch name)
  typedef struct {
    int AP;  // 0xMm (hexidecimal notation), M = AP Major version,
    // and m = AP minor version
    const char* name;
  } sAPtoArchName;

  sAPtoArchName nGpuArchNameAP[] = {
      {0x30, "Kepler"},
      {0x32, "Kepler"},
      {0x35, "Kepler"},
      {0x37, "Kepler"},
      {0x50, "Maxwell"},
      {0x52, "Maxwell"},
      {0x53, "Maxwell"},
      {0x60, "Pascal"},
      {0x61, "Pascal"},
      {0x62, "Pascal"},
      {0x70, "Volta"},
      {0x72, "Xavier"},
      {0x75, "Turing"},
      {0x80, "Ampere"},
      {0x86, "Ampere"},
      {0x87, "Ampere"},
      {-1, "Graphics Device"}};

  int index = 0;

  while (nGpuArchNameAP[index].AP != -1) {
    if (nGpuArchNameAP[index].AP == ((major << 4) + minor)) {
      return nGpuArchNameAP[index].name;
    }

    index++;
  }

  // If we don't find the values, we default use the previous one
  // to run properly
  printf(
      "MapAPtoArchName for AP %d.%d is undefined."
      "  Default to use %s\n",
      major, minor, nGpuArchNameAP[index - 1].name);
  return nGpuArchNameAP[index - 1].name;
}
  // end of GPU Architecture definitions

#ifdef __MACA_RUNTIME_H__
// General GPU Device MACA Initialization
inline int gpuDeviceInit(int devID) {
  int device_count;
  checkMacaErrors(mcGetDeviceCount(&device_count));

  if (device_count == 0) {
    fprintf(stderr,
            "gpuDeviceInit() MACA error: "
            "no devices supporting MACA.\n");
    exit(EXIT_FAILURE);
  }

  if (devID < 0) {
    devID = 0;
  }

  if (devID > device_count - 1) {
    fprintf(stderr, "\n");
    fprintf(stderr, ">> %d MACA capable GPU device(s) detected. <<\n",
            device_count);
    fprintf(stderr,
            ">> gpuDeviceInit (-device=%d) is not a valid"
            " GPU device. <<\n",
            devID);
    fprintf(stderr, "\n");
    return -devID;
  }

  int computeMode = -1, major = 0, minor = 0;
  checkMacaErrors(mcDeviceGetAttribute(&computeMode, mcDevAttrComputeMode, devID));
  checkMacaErrors(mcDeviceGetAttribute(&major, mcDevAttrComputeCapabilityMajor, devID));
  checkMacaErrors(mcDeviceGetAttribute(&minor, mcDevAttrComputeCapabilityMinor, devID));
  if (computeMode == mcComputeModeProhibited) {
    fprintf(stderr,
            "Error: device is running in <Compute Mode "
            "Prohibited>, no threads can use mcSetDevice().\n");
    return -1;
  }

  if (major < 1) {
    fprintf(stderr, "gpuDeviceInit(): GPU device does not support MACA.\n");
    exit(EXIT_FAILURE);
  }

  checkMacaErrors(mcSetDevice(devID));
  printf("gpuDeviceInit() MACA Device [%d]: \"%s\n", devID, _ConvertAPVer2ArchName(major, minor));

  return devID;
}

// This function returns the best GPU (with maximum GFLOPS)
inline int gpuGetMaxGflopsDeviceId() {
  int current_device = 0, ap_per_multiproc = 0;
  int max_perf_device = 0;
  int device_count = 0;
  int devices_prohibited = 0;

  uint64_t max_compute_perf = 0;
  checkMacaErrors(mcGetDeviceCount(&device_count));

  if (device_count == 0) {
    fprintf(stderr,
            "gpuGetMaxGflopsDeviceId() MACA error:"
            " no devices supporting MACA.\n");
    exit(EXIT_FAILURE);
  }

  // Find the best MACA capable GPU device
  current_device = 0;

  while (current_device < device_count) {
    int computeMode = -1, major = 0, minor = 0;
    checkMacaErrors(mcDeviceGetAttribute(&computeMode, mcDevAttrComputeMode, current_device));
    checkMacaErrors(mcDeviceGetAttribute(&major, mcDevAttrComputeCapabilityMajor, current_device));
    checkMacaErrors(mcDeviceGetAttribute(&minor, mcDevAttrComputeCapabilityMinor, current_device));

    // If this GPU is not running on Compute Mode prohibited,
    // then we can add it to the list
    if (computeMode != mcComputeModeProhibited) {
      if (major == 9999 && minor == 9999) {
        ap_per_multiproc = 1;
      } else {
        ap_per_multiproc =
            _ConvertAPVer2Cores(major,  minor);
      }
      int multiProcessorCount = 0, clockRate = 0;
      checkMacaErrors(mcDeviceGetAttribute(&multiProcessorCount, mcDevAttrMultiProcessorCount, current_device));
      mcError_t result = mcDeviceGetAttribute(&clockRate, mcDevAttrClockRate, current_device);
      if (result != mcSuccess) {
        // If mcDevAttrClockRate attribute is not supported we
        // set clockRate as 1, to consider GPU with most APs and MACA Cores.
        if(result == mcErrorInvalidValue) {
          clockRate = 1;
        }
        else {
          fprintf(stderr, "MACA error at %s:%d code=%d(%s) \n", __FILE__, __LINE__,
            static_cast<unsigned int>(result), _MACAGetErrorEnum(result));
          exit(EXIT_FAILURE);
        }
      }
      uint64_t compute_perf = (uint64_t)multiProcessorCount * ap_per_multiproc * clockRate;

      if (compute_perf > max_compute_perf) {
        max_compute_perf = compute_perf;
        max_perf_device = current_device;
      }
    } else {
      devices_prohibited++;
    }

    ++current_device;
  }

  if (devices_prohibited == device_count) {
    fprintf(stderr,
            "gpuGetMaxGflopsDeviceId() MACA error:"
            " all devices have compute mode prohibited.\n");
    exit(EXIT_FAILURE);
  }

  return max_perf_device;
}

// Initialization code to find the best MACA Device
inline int findmcDevice(int argc, const char **argv) {
  int devID = 0;

  // If the command-line has a device number specified, use it
  if (checkCmdLineFlag(argc, argv, "device")) {
    devID = getCmdLineArgumentInt(argc, argv, "device=");

    if (devID < 0) {
      printf("Invalid command line parameter\n ");
      exit(EXIT_FAILURE);
    } else {
      devID = gpuDeviceInit(devID);

      if (devID < 0) {
        printf("exiting...\n");
        exit(EXIT_FAILURE);
      }
    }
  } else {
    // Otherwise pick the device with highest Gflops/s
    devID = gpuGetMaxGflopsDeviceId();
    checkMacaErrors(mcSetDevice(devID));
    int major = 0, minor = 0; 
    checkMacaErrors(mcDeviceGetAttribute(&major, mcDevAttrComputeCapabilityMajor, devID));
    checkMacaErrors(mcDeviceGetAttribute(&minor, mcDevAttrComputeCapabilityMinor, devID));
    printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n",
           devID, _ConvertAPVer2ArchName(major, minor), major, minor);

  }

  return devID;
}

inline int findIntegratedGPU() {
  int current_device = 0;
  int device_count = 0;
  int devices_prohibited = 0;

  checkMacaErrors(mcGetDeviceCount(&device_count));

  if (device_count == 0) {
    fprintf(stderr, "MACA error: no devices supporting MACA.\n");
    exit(EXIT_FAILURE);
  }

  // Find the integrated GPU which is compute capable
  while (current_device < device_count) {
    int computeMode = -1, integrated = -1;
    checkMacaErrors(mcDeviceGetAttribute(&computeMode, mcDevAttrComputeMode, current_device));
    checkMacaErrors(mcDeviceGetAttribute(&integrated, mcDevAttrIntegrated, current_device));
    // If GPU is integrated and is not running on Compute Mode prohibited,
    // then MACA can map to GLES resource
    if (integrated && (computeMode != mcComputeModeProhibited)) {
      checkMacaErrors(mcSetDevice(current_device));

      int major = 0, minor = 0; 
      checkMacaErrors(mcDeviceGetAttribute(&major, mcDevAttrComputeCapabilityMajor, current_device));
      checkMacaErrors(mcDeviceGetAttribute(&minor, mcDevAttrComputeCapabilityMinor, current_device));
      printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n",
             current_device, _ConvertAPVer2ArchName(major, minor), major, minor);

      return current_device;
    } else {
      devices_prohibited++;
    }

    current_device++;
  }

  if (devices_prohibited == device_count) {
    fprintf(stderr,
            "MACA error:"
            " No GLES-MACA Interop capable GPU found.\n");
    exit(EXIT_FAILURE);
  }

  return -1;
}

// General check for MACA GPU AP Capabilities
inline bool checkMacaCapabilities(int major_version, int minor_version) {
  int dev;
  int major = 0, minor = 0;

  checkMacaErrors(mcGetDevice(&dev));
  checkMacaErrors(mcDeviceGetAttribute(&major, mcDevAttrComputeCapabilityMajor, dev));
  checkMacaErrors(mcDeviceGetAttribute(&minor, mcDevAttrComputeCapabilityMinor, dev));

  if ((major > major_version) ||
      (major == major_version &&
       minor >= minor_version)) {
    printf("  Device %d: <%16s >, Compute AP %d.%d detected\n", dev,
           _ConvertAPVer2ArchName(major, minor), major, minor);
    return true;
  } else {
    printf(
        "  No GPU device was found that can support "
        "MACA compute capability %d.%d.\n",
        major_version, minor_version);
    return false;
  }
}
#endif

  // end of MACA Helper Functions
