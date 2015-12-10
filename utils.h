#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <cstdlib>

#ifdef __APPLE__
#   include <OpenCL/opencl.h>
#else
#   include <CL/cl.h>
#endif

#define CL_CHECK(expr) do { \
    cl_int status = expr; \
    if (status != CL_SUCCESS) { \
        std::cerr << "OpenCL error code " << status << " at " << __FILE__ << " [" << __LINE__ << "]: " << #expr << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0);

void CL_CALLBACK pfn_notify(const char *errinfo, const void *private_info, size_t cb, void *user_data);
unsigned next_multiple(const unsigned x, const unsigned n);
double psnr(const unsigned char * const original, const unsigned char * const result, const size_t size);

#endif

