#ifndef CONFIG_H
#define CONFIG_H

#define PROFILE "modified"

#define PLATFORM_NVIDIA 0
#define PLATFORM_ATI    1

#define ENABLE_PROFILING 1
#define UNROLL 0

#define DISTANCE_IN_SPARSE 0

#define BLOCK_SIZE 8
#define BLOCK_SIZE_HALF 4
#define BLOCK_SIZE_SQ 64

#define WINDOW_SIZE 21
#define WINDOW_SIZE_HALF 10

#define STEP_SIZE 7
// Multiple of STEP_SIZE
#define SPLIT_SIZE_X (12*STEP_SIZE)
#define SPLIT_SIZE_Y (12*STEP_SIZE)

#define WINDOW_STEP_SIZE_1 1
#define WINDOW_STEP_SIZE_2 1

#define MAX_BLOCK_COUNT_1 8
// 32 causes crash on CPU
#define MAX_BLOCK_COUNT_2 8

#define USE_KAISER_WINDOW 0

#define DCT_1D 0
#define HAAR_1D 1
#define TRANSFORM_METHOD_1D HAAR_1D

#define D_THRESHOLD_1 (3 * 2500)
#define D_THRESHOLD_2 (3 * 400)

// Default sigma value to use
#ifndef SIGMA
#   define SIGMA 25
#endif

#define VARIANCE ((float)SIGMA*(float)SIGMA)

#if (SIGMA > 40)
#   define USE_2D_THRESHOLD 0
#   define TAU_1D (2.7f * (float)SIGMA)
#else
#   define USE_2D_THRESHOLD 0
#   define TAU_1D (2.7f * (float)SIGMA)
#endif

#define TAU_2D (2.0f * (float)SIGMA)

#endif

