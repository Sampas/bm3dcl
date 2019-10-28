#define _CRT_SECURE_NO_WARNINGS

#include <algorithm>
#include <iostream>
#include <string>
#include <cstdio>
#include <cassert>

#include "PGM.h"

PGM::PGM()
    : filename("")
    , type(0)
    , width(0)
    , height(0)
    , scale(0)
    , image(NULL)
{
    unload();
}

PGM::PGM(const std::string& _filename)
    : filename("")
    , type(0)
    , width(0)
    , height(0)
    , scale(0)
    , image(NULL)
{
    unload();
    load(_filename);
}

PGM::PGM(const int _width, const int _height, const int _type, const int _scale)
    : filename("")
    , type(_type)
    , width(_width)
    , height(_height)
    , scale(_scale)
    , image(new unsigned char[width*height])
{

}

PGM::operator bool() const {
    return !filename.empty() && image != NULL && width != 0 && height != 0 && type != 0 && scale != 0;
}

bool PGM::load(const std::string& _filename) {
    if (image) {
        unload();
    }

    FILE* f = fopen(_filename.c_str(), "rb");
    if (f == NULL) {
        return false;
    }

    filename = _filename;
    fscanf(f, "P%d\n%d %d\n%d\n", &type, &width, &height, &scale);
    image = new unsigned char[width*height];
    size_t n = fread(image, sizeof(unsigned char), width*height, f);
    fclose(f);

    assert(n == width*height);
    return true;
}

void PGM::unload() {
    if (image) {
        delete[] image;
    }
    filename = "";
    image = NULL;
    type = 0;
    width = 0;
    height = 0;
    scale = 0;
}

PGM::~PGM() {
    unload();
}

bool PGM::save(const std::string& _filename) const {
    std::string fn;

    if (_filename.empty()) {
        fn = filename;
    }
    else {
        fn = _filename;
    }

    if (fn.empty()) return false;

    FILE* f = fopen(fn.c_str(), "wb");
    if (f == NULL) return false;

    fprintf(f, "P%d\n%d %d\n%d\n", type, width, height, scale);

    for (int i = 0; i < width*height; i++) {
        fputc(image[i], f);
    }

    fclose(f);
    return true;
}

void PGM::debug() const {
    std::cout << "Filename: " << filename << std::endl;
    std::cout << "Type: " << type << std::endl;
    std::cout << "Scale: " << scale << std::endl;
    std::cout << "Width: " << width << std::endl;
    std::cout << "Height: " << height << std::endl;
}

void PGM::debug_content(const int w, const int h, const int ox, const int oy) const {
    if (image) {
        for (int j = oy; j < std::min(oy + h, height); j++) {
            for (int i = ox; i < std::min(ox + w, width); i++) {
                printf("%u ", image[j*width + i]);
            }
            std::cout << std::endl;
        }
    }
}

