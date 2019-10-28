#ifndef DCT_H
#define DCT_H

// cos(3*PI/16)
#define C3A 0.83146961230254523707878837761791f
// sin(3*PI/16)
#define C3B 0.55557023301960222474283081394853f
// cos(PI/16)
#define C1A 0.98078528040323044912618223613424f
// sin(PI/16)
#define C1B 0.19509032201612826784828486847702f
// sqrt(2) * cos(3*PI/8)
// NOT: sqrt(2) * cos(PI/16)
#define S2C3A 0.54119610014619698439972320536639f
// sqrt(2) * sin(3*PI/8)
// NOT: sqrt(2) * sin(PI/16)
#define S2C3B 1.3065629648763765278566431734272f
// 1 / sqrt(8)
#define C_NORM_1D 0.35355339059327376220042218105242f
// 1 / 8
#define C_NORM_2D 0.125f

#ifndef M_SQRT2_F
#   define M_SQRT2_F 1.4142135623730950488016887242097f
#endif

inline void dct(const float in[8], float out[8], bool normalize) {

    float st2[8], st3[2];
    float tmp;

    // Stage 1

    out[0] = in[0] + in[7];
    out[1] = in[1] + in[6];
    out[2] = in[2] + in[5];
    out[3] = in[3] + in[4];

    out[4] = in[3] - in[4];
    out[5] = in[2] - in[5];
    out[6] = in[1] - in[6];
    out[7] = in[0] - in[7];

    // Stage 2

    st2[0] = out[0] + out[3];
    st2[1] = out[1] + out[2];
    st2[2] = out[1] - out[2];
    st2[3] = out[0] - out[3];

    tmp = C3A * (out[4] + out[7]);

    st2[4] = tmp + (C3B-C3A)*out[7];
    st2[7] = tmp - (C3A+C3B)*out[4];

    tmp = C1A * (out[5] + out[6]);

    st2[5] = tmp + (C1B-C1A)*out[6];
    st2[6] = tmp - (C1A+C1B)*out[5];

    // Stage 3

    out[0] = st2[0] + st2[1];
    out[4] = st2[0] - st2[1];

    tmp = S2C3A * (st2[2] + st2[3]);

    out[2] = tmp + (S2C3B-S2C3A)*st2[3];
    out[6] = tmp - (S2C3A+S2C3B)*st2[2];

    st3[0] = st2[4] + st2[6];
    st3[1] = st2[5] + st2[7];
    out[3] = st2[7] - st2[5];
    out[5] = st2[4] - st2[6];

    // Stage 4

    out[7] = st3[1] - st3[0];
    out[3] *= M_SQRT2_F;
    out[5] *= M_SQRT2_F;
    out[1] = st3[0] + st3[1];

    if (normalize)
        for (int i = 0; i < 8; i++)
            out[i] *= C_NORM_1D;
}

inline void idct(const float in[8], float out[8], bool normalize) {

    float st1[8], st4[2];
    float tmp;

    // Stage 4

    st4[0] = in[1] - in[7];
    st1[5] = in[3] * M_SQRT2_F;
    st1[6] = in[5] * M_SQRT2_F;
    st4[1] = in[1] + in[7];

    // Stage 3

    out[0] = in[0] + in[4];
    out[1] = in[0] - in[4];

    tmp = S2C3A * (in[2] + in[6]);

    out[2] = tmp - (S2C3A+S2C3B)*in[6];
    out[3] = tmp + (S2C3B-S2C3A)*in[2];

    out[4] = st4[0] + st1[6];
    out[5] = st4[1] - st1[5];
    out[6] = st4[0] - st1[6];
    out[7] = st1[5] + st4[1];

    // Stage 2

    st1[0] = out[0] + out[3];
    st1[1] = out[1] + out[2];
    st1[2] = out[1] - out[2];
    st1[3] = out[0] - out[3];

    tmp = C3A * (out[4] + out[7]);

    st1[4] = tmp - (C3A+C3B)*out[7];
    st1[7] = tmp + (C3B-C3A)*out[4];

    tmp = C1A * (out[5] + out[6]);

    st1[5] = tmp - (C1A+C1B)*out[6];
    st1[6] = tmp + (C1B-C1A)*out[5];

    // Stage 1

    out[0] = st1[0] + st1[7];
    out[1] = st1[1] + st1[6];
    out[2] = st1[2] + st1[5];
    out[3] = st1[3] + st1[4];

    out[4] = st1[3] - st1[4];
    out[5] = st1[2] - st1[5];
    out[6] = st1[1] - st1[6];
    out[7] = st1[0] - st1[7];

    if (normalize)
        for (int i = 0; i < 8; i++)
            out[i] *= C_NORM_1D;
}

inline void transpose(float in[8][8], float out[8][8]) {
    int i, j;
    for (j = 0; j < 8; j++) {
        for (i = 0; i < 8; i++) {
            out[j][i] = in[i][j];
        }
    }
}

inline void dct2(float in[8][8], float out[8][8]) {
    int i, j;

    float res[8][8];

#if 1

    // Process rows
    for (j = 0; j < 8; j++) {

        float st2[8], st3[2];
        float tmp;

        // Stage 1

        res[j][0] = in[j][0] + in[j][7];
        res[j][1] = in[j][1] + in[j][6];
        res[j][2] = in[j][2] + in[j][5];
        res[j][3] = in[j][3] + in[j][4];

        res[j][4] = in[j][3] - in[j][4];
        res[j][5] = in[j][2] - in[j][5];
        res[j][6] = in[j][1] - in[j][6];
        res[j][7] = in[j][0] - in[j][7];

        // Stage 2

        st2[0] = res[j][0] + res[j][3];
        st2[1] = res[j][1] + res[j][2];
        st2[2] = res[j][1] - res[j][2];
        st2[3] = res[j][0] - res[j][3];

        tmp = C3A * (res[j][4] + res[j][7]);

        st2[4] = tmp + (C3B-C3A)*res[j][7];
        st2[7] = tmp - (C3A+C3B)*res[j][4];

        tmp = C1A * (res[j][5] + res[j][6]);

        st2[5] = tmp + (C1B-C1A)*res[j][6];
        st2[6] = tmp - (C1A+C1B)*res[j][5];

        // Stage 3

        res[j][0] = st2[0] + st2[1];
        res[j][4] = st2[0] - st2[1];

        tmp = S2C3A * (st2[2] + st2[3]);

        res[j][2] = tmp + (S2C3B-S2C3A)*st2[3];
        res[j][6] = tmp - (S2C3A+S2C3B)*st2[2];

        st3[0] = st2[4] + st2[6];
        st3[1] = st2[5] + st2[7];
        res[j][3] = st2[7] - st2[5];
        res[j][5] = st2[4] - st2[6];

        // Stage 4

        res[j][7] = st3[1] - st3[0];
        res[j][3] *= M_SQRT2_F;
        res[j][5] *= M_SQRT2_F;
        res[j][1] = st3[0] + st3[1];
    }

    // Process columns
    for (i = 0; i < 8; i++) {

        float st2[8], st3[2];
        float tmp;

        // Stage 1

        out[i][0] = res[0][i] + res[7][i];
        out[i][1] = res[1][i] + res[6][i];
        out[i][2] = res[2][i] + res[5][i];
        out[i][3] = res[3][i] + res[4][i];

        out[i][4] = res[3][i] - res[4][i];
        out[i][5] = res[2][i] - res[5][i];
        out[i][6] = res[1][i] - res[6][i];
        out[i][7] = res[0][i] - res[7][i];

        // Stage 2

        st2[0] = out[i][0] + out[i][3];
        st2[1] = out[i][1] + out[i][2];
        st2[2] = out[i][1] - out[i][2];
        st2[3] = out[i][0] - out[i][3];

        tmp = C3A * (out[i][4] + out[i][7]);

        st2[4] = tmp + (C3B-C3A)*out[i][7];
        st2[7] = tmp - (C3A+C3B)*out[i][4];

        tmp = C1A * (out[i][5] + out[i][6]);

        st2[5] = tmp + (C1B-C1A)*out[i][6];
        st2[6] = tmp - (C1A+C1B)*out[i][5];

        // Stage 3

        out[i][0] = st2[0] + st2[1];
        out[i][4] = st2[0] - st2[1];

        tmp = S2C3A * (st2[2] + st2[3]);

        out[i][2] = tmp + (S2C3B-S2C3A)*st2[3];
        out[i][6] = tmp - (S2C3A+S2C3B)*st2[2];

        st3[0] = st2[4] + st2[6];
        st3[1] = st2[5] + st2[7];
        out[i][3] = st2[7] - st2[5];
        out[i][5] = st2[4] - st2[6];

        // Stage 4

        out[i][7] = st3[1] - st3[0];
        out[i][3] *= M_SQRT2_F;
        out[i][5] *= M_SQRT2_F;
        out[i][1] = st3[0] + st3[1];

        // Normalize
        for (j = 0; j < 8; j++) {
            out[i][j] *= C_NORM_2D;
        }
    }

#else

    // Process rows
    for (j = 0; j < 8; j++) {
        dct(in[j], res[j], false);
    }

    transpose(res, out);

    // Process columns
    for (j = 0; j < 8; j++) {
        dct(out[j], res[j], false);
    }

    // Normalize
    for (j = 0; j < 8; j++) {
        for (i = 0; i < 8; i++) {
            out[j][i] = res[j][i] * C_NORM_2D;
        }
    }

#endif
}

inline void idct2(float in[8][8], float out[8][8]) {
    int i, j;

    float res[8][8];

#if 1

    // Process rows
    for (j = 0; j < 8; j++) {

        float st1[8], st4[2];
        float tmp;

        // Stage 4

        st4[0] = in[j][1] - in[j][7];
        st1[5] = in[j][3] * M_SQRT2_F;
        st1[6] = in[j][5] * M_SQRT2_F;
        st4[1] = in[j][1] + in[j][7];

        // Stage 3

        res[j][0] = in[j][0] + in[j][4];
        res[j][1] = in[j][0] - in[j][4];

        tmp = S2C3A * (in[j][2] + in[j][6]);

        res[j][2] = tmp - (S2C3A+S2C3B)*in[j][6];
        res[j][3] = tmp + (S2C3B-S2C3A)*in[j][2];

        res[j][4] = st4[0] + st1[6];
        res[j][5] = st4[1] - st1[5];
        res[j][6] = st4[0] - st1[6];
        res[j][7] = st1[5] + st4[1];

        // Stage 2

        st1[0] = res[j][0] + res[j][3];
        st1[1] = res[j][1] + res[j][2];
        st1[2] = res[j][1] - res[j][2];
        st1[3] = res[j][0] - res[j][3];

        tmp = C3A * (res[j][4] + res[j][7]);

        st1[4] = tmp - (C3A+C3B)*res[j][7];
        st1[7] = tmp + (C3B-C3A)*res[j][4];

        tmp = C1A * (res[j][5] + res[j][6]);

        st1[5] = tmp - (C1A+C1B)*res[j][6];
        st1[6] = tmp + (C1B-C1A)*res[j][5];

        // Stage 1

        res[j][0] = st1[0] + st1[7];
        res[j][1] = st1[1] + st1[6];
        res[j][2] = st1[2] + st1[5];
        res[j][3] = st1[3] + st1[4];

        res[j][4] = st1[3] - st1[4];
        res[j][5] = st1[2] - st1[5];
        res[j][6] = st1[1] - st1[6];
        res[j][7] = st1[0] - st1[7];
    }

    // Process columns
    for (i = 0; i < 8; i++) {

        float st1[8], st4[2];
        float tmp;

        // Stage 4

        st4[0] = res[1][i] - res[7][i];
        st1[5] = res[3][i] * M_SQRT2_F;
        st1[6] = res[5][i] * M_SQRT2_F;
        st4[1] = res[1][i] + res[7][i];

        // Stage 3

        out[i][0] = res[0][i] + res[4][i];
        out[i][1] = res[0][i] - res[4][i];

        tmp = S2C3A * (res[2][i] + res[6][i]);

        out[i][2] = tmp - (S2C3A+S2C3B)*res[6][i];
        out[i][3] = tmp + (S2C3B-S2C3A)*res[2][i];

        out[i][4] = st4[0] + st1[6];
        out[i][5] = st4[1] - st1[5];
        out[i][6] = st4[0] - st1[6];
        out[i][7] = st1[5] + st4[1];

        // Stage 2

        st1[0] = out[i][0] + out[i][3];
        st1[1] = out[i][1] + out[i][2];
        st1[2] = out[i][1] - out[i][2];
        st1[3] = out[i][0] - out[i][3];

        tmp = C3A * (out[i][4] + out[i][7]);

        st1[4] = tmp - (C3A+C3B)*out[i][7];
        st1[7] = tmp + (C3B-C3A)*out[i][4];

        tmp = C1A * (out[i][5] + out[i][6]);

        st1[5] = tmp - (C1A+C1B)*out[i][6];
        st1[6] = tmp + (C1B-C1A)*out[i][5];

        // Stage 1

        out[i][0] = st1[0] + st1[7];
        out[i][1] = st1[1] + st1[6];
        out[i][2] = st1[2] + st1[5];
        out[i][3] = st1[3] + st1[4];

        out[i][4] = st1[3] - st1[4];
        out[i][5] = st1[2] - st1[5];
        out[i][6] = st1[1] - st1[6];
        out[i][7] = st1[0] - st1[7];

        // Normalize
        for (j = 0; j < 8; j++) {
            out[i][j] *= C_NORM_2D;
        }
    }

#else

    // Process rows
    for (j = 0; j < 8; j++) {
        idct(in[j], res[j], false);
    }

    transpose(res, out);

    // Process columns
    for (j = 0; j < 8; j++) {
        idct(out[j], res[j], false);
    }

    // Normalize
    for (j = 0; j < 8; j++) {
        for (i = 0; i < 8; i++) {
            out[j][i] = res[j][i] * C_NORM_2D;
        }
    }

#endif
}

#endif

