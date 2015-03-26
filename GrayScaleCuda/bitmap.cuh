#ifndef BITMAP_CUH
#define BITMAP_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Structure packing
#pragma pack(push, 1)

// Define custom types if not defined before
#ifndef WORD
#define WORD short
#endif

#ifndef DWORD
#define DWORD int
#endif

#ifndef LONG
#define LONG unsigned
#endif

// Based on http://stackoverflow.com/questions/14279242/read-bitmap-file-into-structure
// Bitmap structures
typedef struct
{
    WORD type;
    DWORD size;
    WORD reserved1;
    WORD reserved2;
    DWORD offBits;
} BitmapFileHeader;

typedef struct
{
    DWORD size;
    LONG width;
    LONG height;
    WORD planes;
    WORD bitCount;
    DWORD compression;
    DWORD sizeImage;
    LONG xPelsPerMeter;
    LONG yPelsPerMeter;
    DWORD clrUsed;
    DWORD clrImportant;
} BitmapInfoHeader;

#pragma pack(pop)

unsigned char* loadBitmapFile(char* filename, BitmapInfoHeader* bitmapInfoHeader);
int overwriteBitmapData(char* filename, unsigned char* data);

#endif // BITMAP_CUH