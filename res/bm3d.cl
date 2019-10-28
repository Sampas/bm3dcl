// vim: ft=c

// Note: changes in included files won't trigger rebuilding
#include "config.h"
#include "dct.h"
#include "haar.h"

__constant sampler_t sampler =
      CLK_NORMALIZED_COORDS_FALSE
//    | CLK_ADDRESS_CLAMP_TO_EDGE // Clamp to edge value
    | CLK_ADDRESS_CLAMP // Clamp to zeros
    | CLK_FILTER_NEAREST;

#if USE_KAISER_WINDOW
#   define KAISER(x, y) kaiser_b_2[y*8 + x]
__constant float kaiser_b_2[] = {
    0.1924f, 0.2989f, 0.3846f, 0.4325f, 0.4325f, 0.3846f, 0.2989f, 0.1924f,
    0.2989f, 0.4642f, 0.5974f, 0.6717f, 0.6717f, 0.5974f, 0.4642f, 0.2989f,
    0.3846f, 0.5974f, 0.7688f, 0.8644f, 0.8644f, 0.7688f, 0.5974f, 0.3846f,
    0.4325f, 0.6717f, 0.8644f, 0.9718f, 0.9718f, 0.8644f, 0.6717f, 0.4325f,
    0.4325f, 0.6717f, 0.8644f, 0.9718f, 0.9718f, 0.8644f, 0.6717f, 0.4325f,
    0.3846f, 0.5974f, 0.7688f, 0.8644f, 0.8644f, 0.7688f, 0.5974f, 0.3846f,
    0.2989f, 0.4642f, 0.5974f, 0.6717f, 0.6717f, 0.5974f, 0.4642f, 0.2989f,
    0.1924f, 0.2989f, 0.3846f, 0.4325f, 0.4325f, 0.3846f, 0.2989f, 0.1924f
};
#endif

inline void threshold_2d(float in[8][8]) {
    int i, j;
    for (j = 0; j < BLOCK_SIZE; j++) {
        for (i = 0; i < BLOCK_SIZE; i++) {
            if (fabs(in[j][i]) <= TAU_2D) in[j][i] = 0.0f;
        }
    }
}

inline void threshold_1d(float in[8], int* weight_count, int block_count) {
    int i;
    for (i = 0; i < BLOCK_SIZE; i++) {
        //if (fabs(in[i]) <= TAU_1D * sqrt((float)block_count)) in[i] = 0.0f;
        if (fabs(in[i]) <= TAU_1D) in[i] = 0.0f;
        else (*weight_count)++;
    }
}

__kernel void calc_distances(
    __read_only image2d_t input,
    __global short* similar_coords,
    __global uchar* block_counts,
    const int threshold,
    const int max_block_count,
    const int window_step_size
) {

    const short2 gid = {get_global_id(0) * STEP_SIZE, get_global_id(1) * STEP_SIZE};
    const size_t tot_globals = get_global_size(0) * get_global_size(1);
    const size_t global_id = get_global_id(1) * get_global_size(0) + get_global_id(0);
    // Use this if you want to crash AMD's clBuildProgram()
    //int distances[MAX_BLOCK_COUNT] = { [0 ... MAX_BLOCK_COUNT-1] = INT_MAX };
    int distances[MAX_BLOCK_COUNT_2];
    short2 positions[MAX_BLOCK_COUNT_2];
    short block_count = 0;

    for (int n = 0; n < max_block_count; n++)
        distances[n] = INT_MAX;

    // Start block matching from window
    for (int j = WINDOW_SIZE_HALF % window_step_size; j < WINDOW_SIZE; j += window_step_size) {
        for (int i = WINDOW_SIZE_HALF % window_step_size; i < WINDOW_SIZE; i += window_step_size) {

            // Create reference block
            uchar ref[BLOCK_SIZE][BLOCK_SIZE];

            // Interestingly moving this outside outer for loops reduces performance significantly
            for (int y = 0; y < BLOCK_SIZE; y++) {
                for (int x = 0; x < BLOCK_SIZE; x++) {
                    const int2 ref_pos = {gid.x + x, gid.y + y};
                    ref[y][x] = (uchar)read_imageui(input, sampler, ref_pos).s0;
                }
            }

            int d = 0;

            // Create matching block
            for (int y = 0; y < BLOCK_SIZE; y++) {
                for (int x = 0; x < BLOCK_SIZE; x++) {
                    const int2 pos = {gid.x + i + x - WINDOW_SIZE_HALF,
                                      gid.y + j + y - WINDOW_SIZE_HALF};
                    const uchar b = (uchar)read_imageui(input, sampler, pos).s0;
                    d += (ref[y][x]-b) * (ref[y][x]-b);
                }
            }

            // Only count block if it's distance is under threshold and
            // is smaller than any already found block.
            if (d <= threshold) {
                for (int n = 0; n < max_block_count; n++) {
                    if (d < distances[n]) {
                        for (int k = max_block_count-1; k > n; k--) {
                            distances[k] = distances[k-1];
                            positions[k] = positions[k-1];
                        }
                        block_count++;
                        distances[n] = d;
                        positions[n].x = i;
                        positions[n].y = j;
                        break;
                    }
                }
            }
        }
    }

    if (block_count > max_block_count) block_count = max_block_count;
#if 0
    else {
        uchar n = 1;
        while ((n << 1) <= block_count) n <<= 1;
        block_count = n;
    }
#endif

    block_counts[global_id] = (uchar)block_count;

    /* Nearest block coordinates are written as such into memory:
     *  most similar              second most similar      - sorted order of similarity
     * | x0 | y0 | x0 | y0 | ... | x1 | y1 | x1 | y1 | ... - coords for similar block
     *  t0        t1        tn    t0        t1        tn   - thread id
     *  R00       R01       Rmn   R00       R01       Rmn  - reference block id
     */
    for (int n = 0; n < block_count; n++) {
        // SoA or AoS?
        const int ind = 2 * (n * tot_globals + global_id);
        similar_coords[ind] = positions[n].x;
        similar_coords[ind+1] = positions[n].y;
    }
}

#if USE_PLATFORM == PLATFORM_ATI
#   define ACCU(x, y) accumulator[(y*SPLIT_SIZE_X + x) * tot_globals + global_id]
#   define WM(x, y) weight_map[(y*SPLIT_SIZE_X + x) * tot_globals + global_id]
#else
#   define ACCU(x, y) accumulator[y][x]
#   define WM(x, y) weight_map[y][x]
#endif

__kernel void bm3d_basic_filter(
    __read_only image2d_t input,
    __write_only image2d_t output,
    __global short* similar_coords,
    __global uchar* block_counts,
    const int global_size_x_d,
    const int tot_globals_d
#if USE_PLATFORM == PLATFORM_ATI
  , __global float* accumulator
  , __global float* weight_map
#endif
) {
#if 1
    const int2 gid = {get_global_id(0) * SPLIT_SIZE_X, get_global_id(1) * SPLIT_SIZE_Y};
    if (gid.x > WIDTH-1 || gid.y > HEIGHT-1) return;
    //const size_t tot_globals = get_global_size(0) * get_global_size(1);
    //const size_t global_id = get_global_id(1) * get_global_size(0) + get_global_id(0);

#if 1
    const int2 back_limit = {max(gid.x - WINDOW_SIZE_HALF, 0),
                             max(gid.y - WINDOW_SIZE_HALF, 0)};
    const int2 front_limit = {min(gid.x + SPLIT_SIZE_X - 1 + WINDOW_SIZE_HALF, WIDTH-1),
                              min(gid.y + SPLIT_SIZE_Y - 1 + WINDOW_SIZE_HALF, HEIGHT-1)};
#else
    const int2 back_limit = gid;
    const int2 front_limit = gid;
#endif

#if USE_PLATFORM == PLATFORM_ATI
    for (int j = 0; j < SPLIT_SIZE_Y; j++) {
        for (int i = 0; i < SPLIT_SIZE_X; i++) {
            ACCU(i, j) = 0.0f;
            WM(i, j) = 0.0f;
        }
    }
#else
    float accumulator[SPLIT_SIZE_Y][SPLIT_SIZE_X] = {{0.0f}};
    float weight_map[SPLIT_SIZE_Y][SPLIT_SIZE_X] = {{0.0f}};
#endif

    int ri = gid.x;
    int rj = gid.y;

    while (ri - STEP_SIZE >= back_limit.x) ri -= STEP_SIZE;
    while (rj - STEP_SIZE >= back_limit.y) rj -= STEP_SIZE;

    const int ri_min = ri;
    const int rj_min = rj;

    // Loop through all reference blocks that can contribute to a split block.
    for (rj = rj_min; rj <= front_limit.y; rj += STEP_SIZE) {
        for (ri = ri_min; ri <= front_limit.x; ri += STEP_SIZE) {

            const int rgid = (rj/STEP_SIZE)*global_size_x_d + (ri/STEP_SIZE);

            float stack[MAX_BLOCK_COUNT_1][BLOCK_SIZE][BLOCK_SIZE];

            const uchar block_count = block_counts[rgid];
            int weight_count = 0;

            // Build stack of similar blocks
            for (int n = 0; n < block_count; n++) {
                float block[BLOCK_SIZE][BLOCK_SIZE];
                for (int j = 0; j < BLOCK_SIZE; j++) {
                    for (int i = 0; i < BLOCK_SIZE; i++) {
                        const int2 pos = {ri - WINDOW_SIZE_HALF + similar_coords[2*(n*tot_globals_d + rgid)] + i,
                                          rj - WINDOW_SIZE_HALF + similar_coords[2*(n*tot_globals_d + rgid)+1] + j};
                        block[j][i] = (float)read_imageui(input, sampler, pos).s0;
                    }
                }

                dct2(block, stack[n]);
#if USE_2D_THRESHOLD
                threshold_2d(stack[n]);
#endif
            }

            // Do collaborative filtering
            for (int j = 0; j < BLOCK_SIZE; j++) {
                for (int i = 0; i < BLOCK_SIZE; i++) {

                    int blocks_left = block_count;
                    int k = 0;

                    // Process only max 8 layers at the time because of 8-point DCT
                    while (blocks_left > 0) {

                        float pipe[8] = { 0.0f };
                        float tr_pipe[8];

                        for (int n = 0; n < min(blocks_left, 8); n++) {
                            pipe[n] = stack[k*8 + n][j][i];
                        }

#if TRANSFORM_METHOD_1D == DCT_1D
                        dct(pipe, tr_pipe, true);
                        threshold_1d(tr_pipe, &weight_count, block_count);
                        idct(tr_pipe, pipe, true);
#elif TRANSFORM_METHOD_1D == HAAR_1D
                        haar(pipe, tr_pipe);
                        threshold_1d(tr_pipe, &weight_count, block_count);
                        ihaar(tr_pipe, pipe);
#endif

                        for (int n = 0; n < min(blocks_left, 8); n++) {
                            stack[k*8 + n][j][i] = pipe[n];
                        }

                        k++;
                        blocks_left -= 8;
                    }
                }
            }

            // Convert weight count to weight multiplier
            const float wx = (weight_count >= 1) ? (1.0f / (VARIANCE * (float)weight_count)) : 1.0f;

            // Relocate stack blocks to their positions in split rectangle
            for (int n = 0; n < block_count; n++) {
                float block[BLOCK_SIZE][BLOCK_SIZE];
                idct2(stack[n], block);

                for (int j = 0; j < BLOCK_SIZE; j++) {
                    for (int i = 0; i < BLOCK_SIZE; i++) {
                        const int2 pixel_offset = {ri - gid.x, rj - gid.y};
                        const int2 pixel_pos = {similar_coords[2*(n*tot_globals_d + rgid)] - WINDOW_SIZE_HALF + i + pixel_offset.x,
                                                similar_coords[2*(n*tot_globals_d + rgid)+1] - WINDOW_SIZE_HALF + j + pixel_offset.y};

                        if (pixel_pos.x >= 0 && pixel_pos.y >= 0 && pixel_pos.x < SPLIT_SIZE_X && pixel_pos.y < SPLIT_SIZE_Y) {
#if USE_KAISER_WINDOW
                            const float pixel_wx = wx * KAISER(i, j);
#else
                            const float pixel_wx = wx;
#endif
                            ACCU(pixel_pos.x, pixel_pos.y) += block[j][i] * pixel_wx;
                            WM(pixel_pos.x, pixel_pos.y) += pixel_wx;
                        }
                    }
                }
            }
        }
    }

#if UNROLL
#   pragma unroll
#endif
    for (int j = 0; j < SPLIT_SIZE_Y; j++) {
#if UNROLL
#   pragma unroll
#endif
        for (int i = 0; i < SPLIT_SIZE_X; i++) {
            const int2 pos = {gid.x + i, gid.y + j};
            if (pos.x < WIDTH && pos.y < HEIGHT) {
                // Normalize aggregation output
                uchar pixel_value = convert_uchar_sat(
                    ACCU(i, j) / WM(i, j)
                );
                //if (WM(i, j) == 0) pixel_value = 255;
                //if (pixel_value == 0) pixel_value = 255;
                write_imageui(output, pos, pixel_value);
            }
        }
    }
#endif
}

__kernel void bm3d_wiener_filter(
    __read_only image2d_t input,
    __read_only image2d_t basic,
    __write_only image2d_t output,
    __global short* similar_coords,
    __global uchar* block_counts,
    const int global_size_x_d,
    const int tot_globals_d
#if USE_PLATFORM == PLATFORM_ATI
  , __global float* accumulator
  , __global float* weight_map
#endif
) {
#if 1
    const int2 gid = {get_global_id(0) * SPLIT_SIZE_X, get_global_id(1) * SPLIT_SIZE_Y};
    if (gid.x > WIDTH-1 || gid.y > HEIGHT-1) return;
    //const size_t tot_globals = get_global_size(0) * get_global_size(1);
    //const size_t global_id = get_global_id(1)*get_global_size(0) + get_global_id(0);

#if 1
    const int2 back_limit = {max(gid.x - WINDOW_SIZE_HALF, 0),
                             max(gid.y - WINDOW_SIZE_HALF, 0)};
    const int2 front_limit = {min(gid.x + SPLIT_SIZE_X - 1 + WINDOW_SIZE_HALF, WIDTH-1),
                              min(gid.y + SPLIT_SIZE_Y - 1 + WINDOW_SIZE_HALF, HEIGHT-1)};
#else
    const int2 back_limit = gid;
    const int2 front_limit = gid;
#endif

#if USE_PLATFORM == PLATFORM_ATI
    for (int j = 0; j < SPLIT_SIZE_Y; j++) {
        for (int i = 0; i < SPLIT_SIZE_X; i++) {
            ACCU(i, j) = 0.0f;
            WM(i, j) = 0.0f;
        }
    }
#else
    float accumulator[SPLIT_SIZE_Y][SPLIT_SIZE_X] = {{0.0f}};
    float weight_map[SPLIT_SIZE_Y][SPLIT_SIZE_X] = {{0.0f}};
#endif

    int ri = gid.x;
    int rj = gid.y;

    while (ri - STEP_SIZE >= back_limit.x) ri -= STEP_SIZE;
    while (rj - STEP_SIZE >= back_limit.y) rj -= STEP_SIZE;

    const int ri_min = ri;
    const int rj_min = rj;

    for (rj = rj_min; rj <= front_limit.y; rj += STEP_SIZE) {
        for (ri = ri_min; ri <= front_limit.x; ri += STEP_SIZE) {
            const int rgid = (rj/STEP_SIZE)*global_size_x_d + (ri/STEP_SIZE);

            float basic_stack[MAX_BLOCK_COUNT_2][BLOCK_SIZE][BLOCK_SIZE];
            float noise_stack[MAX_BLOCK_COUNT_2][BLOCK_SIZE][BLOCK_SIZE];

            const uchar block_count = block_counts[rgid];
            float sumsqr_weights = 0.0f;

            // Build stack
            for (int n = 0; n < block_count; n++) {
                float basic_block[BLOCK_SIZE][BLOCK_SIZE];
                float noise_block[BLOCK_SIZE][BLOCK_SIZE];

                for (int j = 0; j < BLOCK_SIZE; j++) {
                    for (int i = 0; i < BLOCK_SIZE; i++) {
                        const int2 pos = {ri - WINDOW_SIZE_HALF + similar_coords[2*(n*tot_globals_d + rgid)] + i,
                                          rj - WINDOW_SIZE_HALF + similar_coords[2*(n*tot_globals_d + rgid)+1] + j};
                        basic_block[j][i] = (float)read_imageui(basic, sampler, pos).s0;
                        noise_block[j][i] = (float)read_imageui(input, sampler, pos).s0;
                    }
                }

                dct2(basic_block, basic_stack[n]);
                dct2(noise_block, noise_stack[n]);
            }

            for (int j = 0; j < BLOCK_SIZE; j++) {
                for (int i = 0; i < BLOCK_SIZE; i++) {

                    int blocks_left = block_count;
                    int k = 0;

                    while (blocks_left > 0) {

                        float pipe[8] = {0.0f};
                        float pipe2[8] = {0.0f};
                        float tr_basic_pipe[8];
                        float tr_noise_pipe[8];

                        for (int n = 0; n < min(blocks_left, 8); n++) {
                            pipe[n] = basic_stack[k*8 + n][j][i];
                            pipe2[n] = noise_stack[k*8 + n][j][i];
                        }

#if TRANSFORM_METHOD_1D == DCT_1D
                        dct(pipe, tr_basic_pipe, true);
                        dct(pipe2, tr_noise_pipe, true);
#elif TRANSFORM_METHOD_1D == HAAR_1D
                        haar(pipe, tr_basic_pipe);
                        haar(pipe2, tr_noise_pipe);
#endif

                        for (int n = 0; n < 8; n++) {
                            // weights
                            pipe[n] = tr_basic_pipe[n]*tr_basic_pipe[n] / (tr_basic_pipe[n]*tr_basic_pipe[n] + VARIANCE);
                            sumsqr_weights += pipe[n]*pipe[n];
                            pipe2[n] = pipe[n]*tr_noise_pipe[n];
                        }

#if TRANSFORM_METHOD_1D == DCT_1D
                        idct(pipe2, pipe, true);
#elif TRANSFORM_METHOD_1D == HAAR_1D
                        ihaar(pipe2, pipe);
#endif

                        for (int n = 0; n < 8; n++) {
                            noise_stack[k*8 + n][j][i] = pipe[n];
                        }

                        k++;
                        blocks_left -= 8;
                    }
                }
            }

            const float wx = 1.0f / (VARIANCE * sumsqr_weights);

            for (int n = 0; n < block_count; n++) {
                float block[BLOCK_SIZE][BLOCK_SIZE];
                idct2(noise_stack[n], block);

                for (int j = 0; j < BLOCK_SIZE; j++) {
                    for (int i = 0; i < BLOCK_SIZE; i++) {
                        const int2 pixel_offset = {ri - gid.x, rj - gid.y};
                        const int2 pixel_pos = {similar_coords[2*(n*tot_globals_d + rgid)] - WINDOW_SIZE_HALF + i + pixel_offset.x,
                                                similar_coords[2*(n*tot_globals_d + rgid)+1] - WINDOW_SIZE_HALF + j + pixel_offset.y};

                        if (pixel_pos.x >= 0 && pixel_pos.y >= 0 && pixel_pos.x < SPLIT_SIZE_X && pixel_pos.y < SPLIT_SIZE_Y) {
#if USE_KAISER_WINDOW
                            const float pixel_wx = wx * KAISER(i, j);
#else
                            const float pixel_wx = wx;
#endif
                            ACCU(pixel_pos.x, pixel_pos.y) += block[j][i] * pixel_wx;
                            WM(pixel_pos.x, pixel_pos.y) += pixel_wx;
                        }
                    }
                }
            }
        }
    }

#if UNROLL
#   pragma unroll
#endif
    for (int j = 0; j < SPLIT_SIZE_Y; j++) {
#if UNROLL
#   pragma unroll
#endif
        for (int i = 0; i < SPLIT_SIZE_X; i++) {
            const int2 pos = {gid.x + i, gid.y + j};
            if (pos.x < WIDTH && pos.y < HEIGHT) {
                const uchar pixel_value = convert_uchar_sat(
                    ACCU(i, j) / WM(i, j)
                );
                write_imageui(output, pos, pixel_value);
            }
        }
    }
#endif
}

