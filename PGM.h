#ifndef PGM_H
#define PGM_H

#include <string>

class PGM {
public:
    PGM();
    PGM(const std::string& _filename);
    PGM(const int _width, const int _height, const int _type=5, const int _scale=255);
    virtual ~PGM();
    /*explicit*/ operator bool() const;

    bool load(const std::string& _filename);
    void unload();
    bool save(const std::string& _filename) const;
    void debug() const;
    void debug_content(const int w=16, const int h=16, const int ox=0, const int oy=0) const;

    std::string filename;
    int type;
    int width;
    int height;
    int scale;
    unsigned char* image;
};

#endif

