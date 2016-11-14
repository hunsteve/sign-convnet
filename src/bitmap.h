/*
 * bitmap.h
 *
 *  Created on: Nov 12, 2016
 *      Author: steve
 */

#ifndef BITMAP_H_
#define BITMAP_H_
#include <stdio.h>
#include <vector>

std::vector<unsigned char> readBMP(const char* filename) {
    FILE* f = fopen(filename, "rb");
    unsigned char header[0x8a];
    fread(header, sizeof(unsigned char), 0x8a, f);  // 0x8a byte header

    int w = *(int*)&header[18];
    int h = *(int*)&header[22];
    int size = 3 * w * h;

    std::vector<unsigned char> rgb;
    rgb.resize(3 * w * h);
    fread(rgb.data(), sizeof(unsigned char), size,
          f);  // read the rest of the data at once
    fclose(f);

    return rgb;
}

#endif /* BITMAP_H_ */
