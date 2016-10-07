#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <sstream>
#include <string>

#include <cstdio>
#include <cmath>
#include <cassert>
#include <cstring>

#include "config.h"
#include "PGM.h"
#include "utils.h"

#ifdef __APPLE__
#   include <OpenCL/opencl.h>
#else
#   include <CL/cl.h>
#endif

#ifdef _MSC_VER
#   include <direct.h>
#   define getcwd _getcwd
#   define putenv _putenv
#else
#   include <unistd.h>
#endif

#ifndef KERNEL_BUILD_OPTIONS
#   define KERNEL_BUILD_OPTIONS ""
#endif

using namespace std;

void usage(const char * const exe) {
    cout << "Usage: " << exe << " <noisy> [sigma] [original]" << endl;
    exit(EXIT_FAILURE);
}

int main(int argc, char** argv) {

    char* cwd = getcwd(NULL, 0);
    cout << "CWD: " << cwd << endl;
    free(cwd);

    string original_fn, noisy_fn;
    int sigma = (int)SIGMA;

    putenv("CUDA_CACHE_DISABLE=1");

    if (argc >= 4) {
        original_fn = argv[3];
    }
    if (argc >= 3) {
        sigma = atoi(argv[2]);
    }
    if (argc >= 2) {
        noisy_fn = argv[1];
    }

    if (noisy_fn.empty()) {
        usage(argv[0]);
    }

    PGM noisy(noisy_fn);
    PGM original(original_fn);

    if (!noisy) {
        cout << "Failed to load noisy image: " << noisy_fn << endl;
        return EXIT_FAILURE;
    }
    cout << "Noisy image: " << noisy_fn << endl;

    if (original) {
        cout << "Original image: " << original_fn << endl;
    }
    cout << "Sigma: " << sigma << endl;

    //noisy.debug();
    //noisy.debug_content();

    cout << "Getting OpenCL platforms..." << endl;

    // Get platform ID count
    cl_uint platform_id_count = 0;
    CL_CHECK(clGetPlatformIDs(0, NULL, &platform_id_count));

    // Get actual platform IDs
    cl_platform_id *platform_ids = new cl_platform_id[platform_id_count];
    CL_CHECK(clGetPlatformIDs(platform_id_count, platform_ids, NULL));

    int platform_id = 0;

    for (int i = 0; i < static_cast<int>(platform_id_count); i++) {
        size_t info_size;
        CL_CHECK(clGetPlatformInfo(platform_ids[i], CL_PLATFORM_NAME, 0, NULL, &info_size));

        char *info = new char[info_size];
        CL_CHECK(clGetPlatformInfo(platform_ids[i], CL_PLATFORM_NAME, info_size, info, NULL));

        cout << "[" << i << "] Name: " << info << endl;

#if USE_PLATFORM == PLATFORM_NVIDIA
        if (!strncmp(info, "NVIDIA", 6)) platform_id = i;
#elif USE_PLATFORM == PLATFORM_ATI
        if (!strncmp(info, "AMD", 3)) platform_id = i;
#endif
        delete[] info;
    }

    cout << "Platform selected: " << platform_id << endl;

    // Get devices
    cl_int error = CL_SUCCESS;
    cl_uint device_id_count = 0;
    CL_CHECK(clGetDeviceIDs(platform_ids[platform_id], CL_DEVICE_TYPE_CPU | CL_DEVICE_TYPE_GPU, 0, NULL, &device_id_count));
    cl_device_id *device_ids = new cl_device_id[device_id_count];
    CL_CHECK(clGetDeviceIDs(platform_ids[platform_id], CL_DEVICE_TYPE_CPU | CL_DEVICE_TYPE_GPU, device_id_count, device_ids, NULL));
    cl_context context = clCreateContext(NULL, device_id_count, device_ids, &pfn_notify, NULL, &error);
    CL_CHECK(error);

    FILE *f = fopen("bm3d.cl", "rb");
    if (f == NULL) {
        cout << "File not found: bm3d.cl" << endl;
        exit(EXIT_FAILURE);
    }
    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    rewind(f);
    char *buf = new char[size+1];
    buf[size] = '\0';
    fread(buf, sizeof(char), size, f);
    fclose(f);

    cl_uint string_count = 1;

    cl_program program = clCreateProgramWithSource(context, string_count, (const char**)&buf, NULL, &error);
    delete[] buf;
    CL_CHECK(error);

    cout << "Available devices:" << endl;
    for (int i = 0; i < static_cast<int>(device_id_count); i++) {
        size_t info_size;
        CL_CHECK(clGetDeviceInfo(device_ids[i], CL_DEVICE_NAME, 0, NULL, &info_size));
        char *info = new char[info_size];
        CL_CHECK(clGetDeviceInfo(device_ids[i], CL_DEVICE_NAME, info_size, info, NULL));
        cout << "[" << i << "] Device name: " << info << endl;
        delete[] info;
    }

#if USE_PLATFORM == PLATFORM_NVIDIA
    int device_id = 0;
#else
    // 0 = GPU, 1 = CPU usually
    int device_id = 0;
#endif
    cout << "Selected device: " << device_id << endl;

    cl_device_id device = device_ids[device_id];

    cl_uint CU_count;
    CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &CU_count, NULL));
    cout << "Compute units: " << CU_count << endl;

    cl_ulong global_mem_size;
    CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &global_mem_size, NULL));
    cout << "Global mem size available: " << global_mem_size << endl;

    string options;
    stringstream ss;
    ss << KERNEL_BUILD_OPTIONS " -cl-nv-verbose -cl-std=CL1.1 -DWIDTH=" << noisy.width << " -DHEIGHT=" << noisy.height << " -DUSE_PLATFORM=" << USE_PLATFORM << " -DSIGMA=" << sigma;
    options = ss.str();

    cout << "Starting kernel build with options: " << options << endl;
    error = clBuildProgram(program, 0, NULL, options.c_str(), NULL, NULL);

    size_t log_size;
    CL_CHECK(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size));

    char *log = new char[log_size+1];
    log[log_size] = '\0';

    CL_CHECK(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size + 1, log, NULL));

    cout << endl << "----Kernel build log----" << endl << log << endl << "----Kernel build end----" << endl << endl;

    CL_CHECK(error); // Check clBuildProgram after printing build log

    cl_kernel dist_kernel = clCreateKernel(program, "calc_distances", &error);
    CL_CHECK(error);

    cl_kernel basic_kernel = clCreateKernel(program, "bm3d_basic_filter", &error);
    CL_CHECK(error);

    cl_kernel wiener_kernel = clCreateKernel(program, "bm3d_wiener_filter", &error);
    CL_CHECK(error);

    size_t multiple;
    CL_CHECK(clGetKernelWorkGroupInfo(dist_kernel, device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &multiple, NULL));
    size_t maxWG;
    CL_CHECK(clGetKernelWorkGroupInfo(dist_kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &maxWG, NULL));

    cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &error);
    CL_CHECK(error);

    const cl_image_format image_format_in = { CL_R, CL_UNSIGNED_INT8 };
    const cl_image_format image_format_out = { CL_R, CL_UNSIGNED_INT8 };
    const size_t image_origin[3] = {0, 0, 0};
    const size_t image_region[3] = {noisy.width, noisy.height, 1};
    const size_t image_size = noisy.width * noisy.height;
#if USE_PLATFORM == PLATFORM_NVIDIA
    cl_mem noisy_image_buffer = clCreateImage2D(context, CL_MEM_READ_ONLY, &image_format_in, noisy.width, noisy.height, 0, NULL, &error);
    CL_CHECK(error);
    cl_mem basic_image_buffer = clCreateImage2D(context, CL_MEM_READ_WRITE, &image_format_out, noisy.width, noisy.height, 0, NULL, &error);
    CL_CHECK(error);
    cl_mem wiener_image_buffer = clCreateImage2D(context, CL_MEM_WRITE_ONLY, &image_format_out, noisy.width, noisy.height, 0, NULL, &error);
    CL_CHECK(error);
#else
    const cl_image_desc image_desc = { CL_MEM_OBJECT_IMAGE2D, noisy.width, noisy.height };
    cl_mem noisy_image_buffer = clCreateImage(context, CL_MEM_READ_ONLY, &image_format_in, &image_desc, NULL, &error);
    CL_CHECK(error);
    cl_mem basic_image_buffer = clCreateImage(context, CL_MEM_READ_WRITE, &image_format_out, &image_desc, NULL, &error);
    CL_CHECK(error);
    cl_mem wiener_image_buffer = clCreateImage(context, CL_MEM_WRITE_ONLY, &image_format_out, &image_desc, NULL, &error);
    CL_CHECK(error);
#endif

    // This can be changed to other powers of two for testing performance
    const size_t ls[2] = {16, 8};
    const int gx_d = next_multiple((unsigned)ceil(noisy.width / (double)STEP_SIZE), ls[0]);
    const int gy_d = next_multiple((unsigned)ceil(noisy.height / (double)STEP_SIZE), ls[1]);
    const int tot_items_d = gx_d * gy_d;

    const size_t gx = next_multiple((unsigned)ceil(noisy.width / (double)SPLIT_SIZE_X), ls[0]);
    const size_t gy = next_multiple((unsigned)ceil(noisy.height / (double)SPLIT_SIZE_Y), ls[1]);
    const size_t tot_items = gx * gy;

    cout << "----Size info----" << endl;
    cout << "Preferred work group size multiple: " << multiple << endl;
    cout << "Max work group size: " << maxWG << endl;
    cout << "Total amount of distance work items: [" << gx_d << ", " << gy_d << "] = " << tot_items_d << endl;
    cout << "Work items in a distance group: [" << ls[0] << ", " << ls[1] << "] = " << ls[0]*ls[1] << endl;
    cout << "Total amount of basic/wiener work items: [" << gx << ", " << gy << "] = " << tot_items << endl;
    cout << "Work items in a basic/wiener group: [" << ls[0] << ", " << ls[1] << "] = " << ls[0]*ls[1] << endl;
    cout << "----Size end ----" << endl << endl;

    const size_t similar_coords_size = MAX_BLOCK_COUNT_2 * tot_items_d * sizeof(cl_short) * 2;
    cout << "Similar coords size: " << similar_coords_size << endl;
    cl_mem similar_coords_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, similar_coords_size, NULL, &error);
    CL_CHECK(error);

    const size_t block_counts_size = tot_items_d * sizeof(cl_uchar);
    cout << "block_counts size: " << block_counts_size << endl;
    cl_mem block_counts_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, block_counts_size, NULL, &error);
    CL_CHECK(error);

    CL_CHECK(clEnqueueWriteImage(queue, noisy_image_buffer, CL_TRUE, image_origin, image_region, 0, 0, noisy.image, 0, NULL, NULL));

    PGM basic(noisy.width, noisy.height);
    PGM wiener(noisy.width, noisy.height);
    cout << "Output image size: " << image_size << " bytes" << endl;

    const cl_int hard_threshold = D_THRESHOLD_1 * BLOCK_SIZE_SQ;
    const cl_int wiener_threshold = D_THRESHOLD_2 * BLOCK_SIZE_SQ;
    const cl_int max_block_count_1 = MAX_BLOCK_COUNT_1;
    const cl_int max_block_count_2 = MAX_BLOCK_COUNT_2;
    const cl_int window_step_size_1 = WINDOW_STEP_SIZE_1;
    const cl_int window_step_size_2 = WINDOW_STEP_SIZE_2;

    CL_CHECK(clSetKernelArg(dist_kernel, 0, sizeof(cl_mem), &noisy_image_buffer));
    CL_CHECK(clSetKernelArg(dist_kernel, 1, sizeof(cl_mem), &similar_coords_buffer));
    CL_CHECK(clSetKernelArg(dist_kernel, 2, sizeof(cl_mem), &block_counts_buffer));
    CL_CHECK(clSetKernelArg(dist_kernel, 3, sizeof(cl_int), &hard_threshold));
    CL_CHECK(clSetKernelArg(dist_kernel, 4, sizeof(cl_int), &max_block_count_1));
    CL_CHECK(clSetKernelArg(dist_kernel, 5, sizeof(cl_int), &window_step_size_1));

    CL_CHECK(clSetKernelArg(basic_kernel, 0, sizeof(cl_mem), &noisy_image_buffer));
    CL_CHECK(clSetKernelArg(basic_kernel, 1, sizeof(cl_mem), &basic_image_buffer));
    CL_CHECK(clSetKernelArg(basic_kernel, 2, sizeof(cl_mem), &similar_coords_buffer));
    CL_CHECK(clSetKernelArg(basic_kernel, 3, sizeof(cl_mem), &block_counts_buffer));
    CL_CHECK(clSetKernelArg(basic_kernel, 4, sizeof(cl_int), &gx_d));
    CL_CHECK(clSetKernelArg(basic_kernel, 5, sizeof(cl_int), &tot_items_d));
#if USE_PLATFORM == PLATFORM_ATI
    const size_t accu_size = tot_items * SPLIT_SIZE_X * SPLIT_SIZE_Y * sizeof(cl_float);
    cout << "accu_size: " << accu_size << endl;
    cl_mem accumulator_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, accu_size, NULL, &error);
    CL_CHECK(error);
    cl_mem weight_map_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, accu_size, NULL, &error);
    CL_CHECK(error);
    CL_CHECK(clSetKernelArg(basic_kernel, 6, sizeof(cl_mem), &accumulator_buffer));
    CL_CHECK(clSetKernelArg(basic_kernel, 7, sizeof(cl_mem), &weight_map_buffer));
#endif

    cl_short *distances = new cl_short[similar_coords_size];
    for (int i = 0; i < static_cast<int>(similar_coords_size / sizeof(cl_short)); i++) {
        distances[i] = 0;
    }
    cl_short *distances2 = new cl_short[similar_coords_size];
    for (int i = 0; i < static_cast<int>(similar_coords_size / sizeof(cl_short)); i++) {
        distances2[i] = 0;
    }

    cl_event event;

#if ENABLE_PROFILING
    double total_time = 0.0;
#endif

    cout << endl << "1st step..." << endl;

    // Change these if want to use offset 
    for (int j = 0; j < 1; j++) {
        for (int i = 0; i < 1; i++) {
            size_t gs_d[2] = {gx_d, gy_d};
            size_t gs[2] = {gx, gy};
            size_t offset[2] = {i*gx, j*gy};
            assert(ls[0] * ls[1] <= maxWG);
            assert(!(gs[0] % ls[0]));
            assert(!(gs[1] % ls[1]));

            CL_CHECK(clEnqueueNDRangeKernel(queue, dist_kernel, 2, offset, gs_d, ls, 0, NULL, &event));
            cout << "Distances kernel enqueued" << endl;

#if ENABLE_PROFILING
            CL_CHECK(clWaitForEvents(1, &event));

            cl_ulong time_start, time_end;
            double exec_time;

            CL_CHECK(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL));
            CL_CHECK(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL));
            exec_time = static_cast<double>(time_end - time_start);
            total_time += exec_time;
            printf("Distances execution time: %0.3f ms\n", exec_time/1000000.0);
#endif

            CL_CHECK(clEnqueueReadBuffer(queue, similar_coords_buffer, CL_TRUE, 0, similar_coords_size, distances, 0, NULL, NULL));

            CL_CHECK(clEnqueueNDRangeKernel(queue, basic_kernel, 2, offset, gs, ls, 0, NULL, &event));
            cout << "Basic kernel enqueued" << endl;

#if ENABLE_PROFILING
            CL_CHECK(clWaitForEvents(1, &event));
            CL_CHECK(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL));
            CL_CHECK(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL));
            exec_time = static_cast<double>(time_end - time_start);
            total_time += exec_time;
            printf("Basic execution time: %0.3f ms\n", exec_time/1000000.0);
#endif
        }
    }

    cout << endl << "2nd step..." << endl;

    CL_CHECK(clSetKernelArg(dist_kernel, 0, sizeof(cl_mem), &basic_image_buffer));
    CL_CHECK(clSetKernelArg(dist_kernel, 3, sizeof(cl_int), &wiener_threshold));
    CL_CHECK(clSetKernelArg(dist_kernel, 4, sizeof(cl_int), &max_block_count_2));
    CL_CHECK(clSetKernelArg(dist_kernel, 5, sizeof(cl_int), &window_step_size_2));

    CL_CHECK(clSetKernelArg(wiener_kernel, 0, sizeof(cl_mem), &noisy_image_buffer));
    CL_CHECK(clSetKernelArg(wiener_kernel, 1, sizeof(cl_mem), &basic_image_buffer));
    CL_CHECK(clSetKernelArg(wiener_kernel, 2, sizeof(cl_mem), &wiener_image_buffer));
    CL_CHECK(clSetKernelArg(wiener_kernel, 3, sizeof(cl_mem), &similar_coords_buffer));
    CL_CHECK(clSetKernelArg(wiener_kernel, 4, sizeof(cl_mem), &block_counts_buffer));
    CL_CHECK(clSetKernelArg(wiener_kernel, 5, sizeof(cl_int), &gx_d));
    CL_CHECK(clSetKernelArg(wiener_kernel, 6, sizeof(cl_int), &tot_items_d));
#if USE_PLATFORM == PLATFORM_ATI
    CL_CHECK(clSetKernelArg(wiener_kernel, 7, sizeof(cl_mem), &accumulator_buffer));
    CL_CHECK(clSetKernelArg(wiener_kernel, 8, sizeof(cl_mem), &weight_map_buffer));
#endif

    // Change these if want to use offset
    for (int j = 0; j < 1; j++) {
        for (int i = 0; i < 1; i++) {
            size_t gs_d[2] = {gx_d, gy_d};
            size_t gs[2] = {gx, gy};
            size_t offset[2] = {i*gx, j*gy};
            assert(ls[0] * ls[1] <= maxWG);
            assert(!(gs[0] % ls[0]));
            assert(!(gs[1] % ls[1]));

            CL_CHECK(clEnqueueNDRangeKernel(queue, dist_kernel, 2, offset, gs_d, ls, 0, NULL, &event));
            cout << "Distances kernel enqueued" << endl;

#if ENABLE_PROFILING
            CL_CHECK(clWaitForEvents(1, &event));

            cl_ulong time_start, time_end;
            double exec_time;

            CL_CHECK(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL));
            CL_CHECK(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL));
            exec_time = static_cast<double>(time_end - time_start);
            total_time += exec_time;
            printf("Distances execution time: %0.3f ms\n", exec_time/1000000.0);
#endif

            CL_CHECK(clEnqueueReadBuffer(queue, similar_coords_buffer, CL_TRUE, 0, similar_coords_size, distances2, 0, NULL, NULL));

            CL_CHECK(clEnqueueNDRangeKernel(queue, wiener_kernel, 2, offset, gs, ls, 0, NULL, &event));
            cout << "Wiener kernel enqueued" << endl;

#if ENABLE_PROFILING
            CL_CHECK(clWaitForEvents(1, &event));
            CL_CHECK(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL));
            CL_CHECK(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL));
            exec_time = static_cast<double>(time_end - time_start);
            total_time += exec_time;
            printf("Wiener execution time: %0.3f ms\n", exec_time/1000000.0);
#endif
        }
    }

#if ENABLE_PROFILING
    printf("Total execution time: %0.3f ms\n", total_time/1000000.0);
#endif

    cout << "Reading basic image..." << endl;

    CL_CHECK(clEnqueueReadImage(queue, basic_image_buffer, CL_TRUE, image_origin, image_region, 0, 0, basic.image, 0, NULL, NULL));

    cout << "Reading wiener image..." << endl;

    CL_CHECK(clEnqueueReadImage(queue, wiener_image_buffer, CL_TRUE, image_origin, image_region, 0, 0, wiener.image, 0, NULL, NULL));

    cout << "Wiener output:" << endl;
    for (int j = 0; j < 16; j++) {
        for (int i = 0; i < 16; i++) {
            printf("%u ", wiener.image[j*wiener.width + i]);
        }
        cout << endl;
    }

    if (original) {
        cout << "Reference:" << endl;
        for (int j = 0; j < 16; j++) {
            for (int i = 0; i < 16; i++) {
                printf("%u ", original.image[j*original.width + i]);
            }
            cout << endl;
        }
    }

#if 1
    f = fopen("distances1.dat", "wb");

    for (int i = 0; i < static_cast<int>(similar_coords_size); i += 2) {
        fputc(*((char*)distances + i + 1), f);
        fputc(*((char*)distances + i), f);
    }

    fclose(f);
#endif

#if 1
    f = fopen("distances2.dat", "wb");

    for (int i = 0; i < static_cast<int>(similar_coords_size); i += 2) {
        fputc(*((char*)distances2 + i + 1), f);
        fputc(*((char*)distances2 + i), f);
    }

    fclose(f);
#endif

    cout << "Writing: basic.pgm" << endl;
    basic.save("basic.pgm");
    cout << "Writing: wiener.pgm" << endl;
    wiener.save("wiener.pgm");

    if (original) {
        std::cout << "PSNR (noisy) : " << psnr(original.image, noisy.image, original.width*original.height) << " dB" << std::endl;
        std::cout << "PSNR (basic) : " << psnr(original.image, basic.image, original.width*original.height) << " dB" << std::endl;
        std::cout << "PSNR (wiener): " << psnr(original.image, wiener.image, original.width*original.height) << " dB" << std::endl;
    }


    // Create collage
    if (original) {
        PGM collage(original.width*2, original.height*2);
        for (int j = 0; j < original.height; j++) {
            for (int i = 0; i < original.width; i++) {
                collage.image[j*collage.width + i] = original.image[j*original.width + i];
                collage.image[j*collage.width + i + original.width] = noisy.image[j*original.width + i];
                collage.image[(j + original.height)*collage.width + i] = basic.image[j*original.width + i];
                collage.image[(j + original.height)*collage.width + i + original.width] = wiener.image[j*original.width + i];
            }
        }

        collage.save("collage.pgm");
    }
    else {
        cout << "No original image provided for reference, so no PSNR calculations or collage" << endl;
    }

    system("md5sum distances1.dat");
    system("md5sum distances2.dat");
    system("md5sum basic.pgm");
    system("md5sum wiener.pgm");

    delete[] device_ids;
    delete[] platform_ids;
    delete[] distances;
    delete[] distances2;

    CL_CHECK(clReleaseMemObject(noisy_image_buffer));
    CL_CHECK(clReleaseMemObject(basic_image_buffer));
    CL_CHECK(clReleaseMemObject(wiener_image_buffer));
    CL_CHECK(clReleaseMemObject(similar_coords_buffer));
    CL_CHECK(clReleaseMemObject(block_counts_buffer));
#if USE_PLATFORM == PLATFORM_ATI
    CL_CHECK(clReleaseMemObject(accumulator_buffer));
    CL_CHECK(clReleaseMemObject(weight_map_buffer));
#endif
    CL_CHECK(clReleaseKernel(dist_kernel));
    CL_CHECK(clReleaseKernel(basic_kernel));
    CL_CHECK(clReleaseKernel(wiener_kernel));
    CL_CHECK(clReleaseCommandQueue(queue));
    CL_CHECK(clReleaseProgram(program));
    CL_CHECK(clReleaseContext(context));

    system("pause");
    return EXIT_SUCCESS;
}

