/*
By downloading, copying, installing or using the software you agree to this license.
If you do not agree to this license, do not download, install,
copy or use the software.


                  License Agreement For libfacedetection
                     (3-clause BSD License)

Copyright (c) 2018-2021, Shiqi Yu, all rights reserved.
shiqi.yu@gmail.com

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are disclaimed.
In no event shall copyright holders or contributors be liable for any direct,
indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
*/

#pragma once

#include "facedetection_export.h"

//#define _ENABLE_AVX512 //Please enable it if X64 CPU
//#define _ENABLE_AVX2 //Please enable it if X64 CPU
//#define _ENABLE_NEON //Please enable it if ARM CPU

#ifdef __cplusplus
extern "C" {
#endif

FACEDETECTION_EXPORT int * facedetect_cnn(unsigned char * result_buffer, //buffer memory for storing face detection results, !!its size must be 0x20000 Bytes!!
                    unsigned char * rgb_image_data, int width, int height, int step, float thresh); //input image, it must be BGR (three channels) insteed of RGB image!

FACEDETECTION_EXPORT void release_resources();

#ifdef __cplusplus
} /* extern "C" */
#endif
/*
DO NOT EDIT the following code if you don't really understand it.
*/
#if defined(_ENABLE_AVX512) || defined(_ENABLE_AVX2)
#include <immintrin.h>
#endif


#if defined(_ENABLE_NEON)
#include "arm_neon.h"
//NEON does not support UINT8*INT8 dot product
//to conver the input data to range [0, 127],
//and then use INT8*INT8 dot product
#define _MAX_UINT8_VALUE 127
#else
#define _MAX_UINT8_VALUE 255
#endif

#if defined(_ENABLE_AVX512) 
#define _MALLOC_ALIGN 512
#elif defined(_ENABLE_AVX2) 
#define _MALLOC_ALIGN 256
#else
#define _MALLOC_ALIGN 128
#endif

#if defined(_ENABLE_AVX512)&& defined(_ENABLE_NEON)
#error Cannot enable the two of AVX512 and NEON at the same time.
#endif
#if defined(_ENABLE_AVX2)&& defined(_ENABLE_NEON)
#error Cannot enable the two of AVX and NEON at the same time.
#endif
#if defined(_ENABLE_AVX512)&& defined(_ENABLE_AVX2)
#error Cannot enable the two of AVX512 and AVX2 at the same time.
#endif


#if defined(_OPENMP)
#include <omp.h>
#endif

#include <string.h>

void* myAlloc(size_t size);
void myFree_(void* ptr);
#define myFree(ptr) (myFree_(*(ptr)), *(ptr)=0);

#ifndef MIN
#  define MIN(a,b)  ((a) > (b) ? (b) : (a))
#endif

#ifndef MAX
#  define MAX(a,b)  ((a) < (b) ? (b) : (a))
#endif

typedef struct FaceRect_
{
    float score;
    int x;
    int y;
    int w;
    int h;
    int lm[10];
}FaceRect;

typedef struct ConvInfoStruct_ {
    int channels;
    int num_filters;
    int is_depthwise; // int
    int is_pointwise; // int
    int with_relu;    // int
    float* pWeights;
    float* pBiases;
}ConvInfoStruct;

typedef struct CDataBlob_ {
	int rows;
	int cols;
	int channels; //in element
    int channelStep; //in byte
    int totalCapacity; //in byte
    int flag;
    float* data;
    char* name;
}CDataBlob;

void CDataBlob_create(CDataBlob* blob, int r, int c, int ch);
void CDataBlob_setNULL(CDataBlob* blob);
void CDataBlob_setZero(CDataBlob* blob);
int CDataBlob_isEmpty(const CDataBlob* blob);
float* CDataBlob_ptr(const CDataBlob* blob, int r, int c);
float CDataBlob_getElement(const CDataBlob* blob, int r, int c, int ch);

typedef struct Filters_ {
    int channels;
    int num_filters;
    int is_depthwise;
    int is_pointwise;
    int with_relu;
    CDataBlob weights;
    CDataBlob biases;
}Filters;

void Filters_create(Filters* filters, const ConvInfoStruct* convinfo);
void Filters_release(Filters* filters);

void objectdetect_cnn(const unsigned char* rgbImageData, int with, int height, int step, float thresh,
                      FaceRect** faceRects, int* num_faces);

//CDataBlob<float> setDataFrom3x3S2P1to1x1S1P0FromImage(const unsigned char* inputData, int imgWidth, int imgHeight, int imgChannels, int imgWidthStep, int padDivisor=32);
void setDataFrom3x3S2P1to1x1S1P0FromImage(const unsigned char* inputData, int imgWidth, int imgHeight, int imgChannels, int imgWidthStep, int padDivisor,
                                          CDataBlob* outBlob);

//CDataBlob<float> convolution(const CDataBlob* inputData, const Filters* filters, int do_relu = true);
void convolution(const CDataBlob* inputData, const Filters* filters, int do_relu,
                CDataBlob* outBlob);

//CDataBlob<float> convolutionDP(const CDataBlob* inputData,
//                const Filters* filtersP, const Filters* filtersD, int do_relu = true);
void convolutionDP(const CDataBlob* inputData,
                const Filters* filtersP, const Filters* filtersD, int do_relu,
                CDataBlob* outBlob);

//CDataBlob<float> convolution4layerUnit(const CDataBlob* inputData,
//                const Filters* filtersP1, const Filters* filtersD1,
//                const Filters* filtersP2, const Filters* filtersD2, int do_relu = true);
void convolution4layerUnit(const CDataBlob* inputData,
                const Filters* filtersP1, const Filters* filtersD1,
                const Filters* filtersP2, const Filters* filtersD2, int do_relu,
                CDataBlob* outBlob);

void maxpooling2x2S2(const CDataBlob* inputData,
                    CDataBlob* outBlob);

//CDataBlob<float> elementAdd(const CDataBlob* inputData1, const CDataBlob* inputData2);
void elementAdd(const CDataBlob* inputData1, const CDataBlob* inputData2,
                CDataBlob* outBlob);

//CDataBlob<float> upsampleX2(const CDataBlob* inputData);
void upsampleX2(const CDataBlob* inputData,
                CDataBlob* outBlob);

//CDataBlob<float> meshgrid(int feature_width, int feature_height, int stride, float offset=0.0f);
void meshgrid(int feature_width, int feature_height, int stride, float offset,
            CDataBlob* outBlob);

// TODO implement in SIMD
void bbox_decode(CDataBlob* bbox_pred, const CDataBlob* priors, int stride);
void kps_decode(CDataBlob* bbox_pred, const CDataBlob* priors, int stride);

void blob2vector(const CDataBlob* inputData,
                CDataBlob* outBlob);

void concat3(const CDataBlob* inputData1, const CDataBlob* inputData2, const CDataBlob* inputData3,
                CDataBlob* outBlob);

// TODO implement in SIMD
void sigmoid(CDataBlob* inputData);

void detection_output(const CDataBlob* cls,
                const CDataBlob* reg,
                const CDataBlob* kps,
                const CDataBlob* obj,
                float overlap_threshold, float confidence_threshold, int top_k, int keep_top_k,
                FaceRect** face_rects, int* num_faces);