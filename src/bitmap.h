/*
 * bitmap.h
 *
 *  Created on: Nov 12, 2016
 *      Author: steve
 */

#ifndef BITMAP_H_
#define BITMAP_H_
#include <stdio.h>

unsigned char* readBMP(char* filename)
{
	FILE* f = fopen(filename, "rb");
	unsigned char header[54];
	fread(header, sizeof(unsigned char), 54, f); //54 byte header

	int w = *(int*)&header[18];
	int h = *(int*)&header[22];
	int size = 3 * w * h;

	unsigned char* rgb = new unsigned char[size]; // allocate 3 bytes per pixel

	fread(rgb, sizeof(unsigned char), size, f); // read the rest of the data at once
	fclose(f);

	return rgb;
}


#endif /* BITMAP_H_ */
