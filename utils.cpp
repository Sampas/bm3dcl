#include "utils.h"

#include <limits>
#include <cmath>

void CL_CALLBACK pfn_notify(const char *errinfo, const void *private_info, size_t cb, void *user_data) {
    std::cerr << "OpenCL Error: " << errinfo << std::endl;
}

unsigned next_multiple(const unsigned x, const unsigned n) {
    return (x + n-1) & ~(n-1);
}

double psnr(const unsigned char * const original, const unsigned char * const result, const size_t size) {
    double mse = 0.0;

    for (size_t i = 0; i < size; i++) {
        double e = (double)original[i] - (double)result[i];
        mse += e * e;
    }

    mse /= (double)size;
    if (mse == 0.0) return std::numeric_limits<double>::infinity();
    return 10.0 * log10(255.0 * 255.0 / mse);
}

