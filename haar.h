#ifndef HAAR_H
#define HAAR_H

#define INV_SQRT2_F 0.70710678118654752440084436210485f

inline void haar(float x[8], float y[8]) {
  int i, j;
  int k = 8;

#pragma unroll
  for (j = 0; j < 3; j++) {
    int k2 = k;
    k >>= 1;

#pragma unroll
    for (i = 0; i < k; i++) {
      int i2 = i << 1;
      int i21 = i2 + 1;
      y[i]   = ( x[i2] + x[i21] ) * INV_SQRT2_F;
      y[i+k] = ( x[i2] - x[i21] ) * INV_SQRT2_F;
    }

#pragma unroll
    for (i = 0; i < k2; i++) {
      x[i] = y[i];
    }
  }
}

inline void ihaar(float x[8], float y[8]) {
  int i, j;
  int k = 1;

#pragma unroll
  for (j = 0; j < 3; j++) {

#pragma unroll
    for (i = 0; i < k; i++) {
      int i2 = i << 1;
      int ik = i + k;
      y[i2]   = ( x[i] + x[ik] ) * INV_SQRT2_F;
      y[i2+1] = ( x[i] - x[ik] ) * INV_SQRT2_F;
    }

    k <<= 1;

#pragma unroll
    for (i = 0; i < k; i++) {
      x[i] = y[i];
    }
  }
}

#endif

