#include <stdio.h>
#include <stdlib.h>

#include "bitmap.cuh"
#include "utils.cuh"

// Based on http://stackoverflow.com/questions/14279242/read-bitmap-file-into-structure
// Load bitmap file, create a header object and pass it into access information
unsigned char *loadBitmapFile(char* filename, BitmapInfoHeader* bitmapInfoHeader)
{
    FILE* filePtr;
    BitmapFileHeader bitmapFileHeader;
    unsigned char* bitmapImage = 0;
    int imageIdx = 0;

    // Open filename in read binary mode
    filePtr = fopen(filename, "rb");
    if (filePtr == NULL) {
		LOG_ERROR("Failed to open file: '%s'", filename);
        return NULL;
	}

    // Read the bitmap file header
    fread(&bitmapFileHeader, sizeof(BitmapFileHeader), 1, filePtr);

    // Verify that this is a bmp file by checking bitmap id
    if (bitmapFileHeader.type !=0x4D42) {
		LOG_ERROR("Invalid file '%s'", filename);
        fclose(filePtr);
        return NULL;
    }

    // Read the bitmap info header
    fread(bitmapInfoHeader, sizeof(BitmapInfoHeader), 1, filePtr);

    // Move file point to the beginning of bitmap data
    int failed = fseek(filePtr, bitmapFileHeader.offBits, SEEK_SET);

	if(failed != 0)	{
		LOG_ERROR("Seeking bitmap failed");
        free(bitmapImage);
        fclose(filePtr);
        return NULL;
	}

    // Read bitmap image data
    bitmapImage = (unsigned char*)malloc(bitmapInfoHeader->sizeImage);

    if (!bitmapImage) {
		LOG_ERROR("Malloc failed");
        free(bitmapImage);
        fclose(filePtr);
        return NULL;
    }

    fread(bitmapImage, bitmapInfoHeader->sizeImage, 1, filePtr);

    // Make sure bitmap image data was read
    if (bitmapImage == NULL) {
		LOG_ERROR("Unable to read bitmap data");
        fclose(filePtr);
        return NULL;
    }

    fclose(filePtr);
    return bitmapImage;
}

int overwriteBitmapData(char* filename, unsigned char* data)
{
	FILE* filePtr;
    BitmapFileHeader bitmapFileHeader;
	BitmapInfoHeader bitmapInfoHeader;

    // Open filename in read write binary mode
    filePtr = fopen(filename, "rb+");
    if (filePtr == NULL) {
		LOG_ERROR("Failed to open file: '%s'", filename);
        return FALSE;
	}

    // Read the bitmap file header
    fread(&bitmapFileHeader, sizeof(BitmapFileHeader), 1, filePtr);

    // Verify that this is a bmp file by checking bitmap id
    if (bitmapFileHeader.type !=0x4D42) {
		LOG_ERROR("Invalid file '%s'", filename);
        fclose(filePtr);
        return FALSE;
    }

    // Read the bitmap info header
    fread(&bitmapInfoHeader, sizeof(BitmapInfoHeader), 1, filePtr);

    // Move file point to the beginning of bitmap data
    int failed = fseek(filePtr, bitmapFileHeader.offBits, SEEK_SET);
	if(failed != 0)	{
		LOG_ERROR("Seeking bitmap failed");
        fclose(filePtr);
        return NULL;
	}
	// Overwrite data
	fwrite((void*)data, bitmapInfoHeader.sizeImage, 1, filePtr); 

    fclose(filePtr);
    return TRUE;
}