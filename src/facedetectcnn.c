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
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "facedetectcnn.h"

typedef struct NormalizedBBox_
{
    float xmin;
    float ymin;
    float xmax;
    float ymax;
    float lm[10];
    float score;
} NormalizedBBox;

//#define __DEBUG_ALLOC_FREE_NUM_

#ifdef __DEBUG_ALLOC_FREE_NUM_
    static int g_myAlloc_num = 0;
#endif

void* myAlloc(size_t size)
{
    char *ptr, *ptr0;
	ptr0 = (char*)malloc(
		(size_t)(size + _MALLOC_ALIGN * ((size >= 4096) + 1L) + sizeof(char*)));

	if (!ptr0)
		return 0;

	// align the pointer
	ptr = (char*)(((size_t)(ptr0 + sizeof(char*) + 1) + _MALLOC_ALIGN - 1) & ~(size_t)(_MALLOC_ALIGN - 1));
	*(char**)(ptr - sizeof(char*)) = ptr0;

#ifdef __DEBUG_ALLOC_FREE_NUM_
    g_myAlloc_num++;
    printf("myAlloc: %d\n", g_myAlloc_num);
#endif
	return ptr;
}

void myFree_(void* ptr)
{
	// Pointer must be aligned by _MALLOC_ALIGN
	if (ptr)
	{
		if (((size_t)ptr & (_MALLOC_ALIGN - 1)) != 0)
			return;
		free(*((char**)ptr - 1));
#ifdef __DEBUG_ALLOC_FREE_NUM_
        g_myAlloc_num--;
        printf("myFree: %d\n", g_myAlloc_num);
#endif
	}
}

void CDataBlob_create(CDataBlob* blob, int r, int c, int ch)
{
    int channelStep = 0;
    //alloc space for int8 array
    int remBytes = (sizeof(float)* ch) % (_MALLOC_ALIGN / 8);
    if (remBytes == 0)
    {
        channelStep = ch * sizeof(float);
    }
    else
    {
        channelStep = (ch * sizeof(float)) + (_MALLOC_ALIGN / 8) - remBytes;
    }
    int totalCapacity = r * c * channelStep;
    if((0 != blob->flag) && (blob->data) && (blob->totalCapacity >= totalCapacity))
    {
        blob->rows = r;
        blob->cols = c;
        blob->channels = ch;
        blob->channelStep = channelStep;
        return;
    }

    CDataBlob_release(blob);
    blob->data = (float *)myAlloc(totalCapacity);
    if (blob->data == NULL)
    {
        fprintf(stderr, "%s : Failed to alloc memeory for uint8 data blob: %d, %d %d\n",__FUNCTION__ , blob->rows, blob->cols, blob->channels);
    }
    blob->rows = r;
    blob->cols = c;
    blob->channels = ch;
    blob->channelStep = channelStep;
    blob->totalCapacity = totalCapacity;
    blob->flag = 1;
    CDataBlob_setZero(blob);
}

void CDataBlob_setZero(CDataBlob* blob)
{
    if(blob->data)
    {
        memset(blob->data, 0, blob->channelStep * blob->rows * blob->cols);
    }
}

void CDataBlob_release(CDataBlob* blob) //CDataBlob_setNULL
{
    if (blob->data)
    {
        myFree(&blob->data);
    }
    blob->rows = 0;
    blob->cols = 0;
    blob->channels = 0;
    blob->channelStep = 0;
    blob->totalCapacity = 0;
    blob->flag = 0;
    blob->data = NULL;
    blob->name = NULL;
}

int CDataBlob_isEmpty(const CDataBlob* blob)
{
    return (blob->rows <= 0 || blob->cols <= 0 || blob->channels == 0 || blob->data == NULL) ? 1 : 0;
}

float* CDataBlob_ptr(const CDataBlob* blob, int r, int c)
{
    if( r < 0 || r >= blob->rows || c < 0 || c >= blob->cols )
        return NULL;

    return (blob->data + (r * blob->cols + c) * blob->channelStep /sizeof(float));
}

float CDataBlob_getElement(const CDataBlob* blob, int r, int c, int ch)
{
    if (blob->data)
    {
        if (r >= 0 && r < blob->rows &&
            c >= 0 && c < blob->cols &&
            ch >= 0 && ch < blob->channels)
        {
            const float* p = CDataBlob_ptr(blob, r, c);
            return (p[ch]);
        }
    }

    return 0.0f;
}

void Filters_create(Filters* filters, const ConvInfoStruct* convinfo)
{
    filters->channels = convinfo->channels;
    filters->num_filters =  convinfo->num_filters;
    filters->is_depthwise = convinfo->is_depthwise;
    filters->is_pointwise = convinfo->is_pointwise;
    filters->with_relu = convinfo->with_relu;

    if(!filters->is_depthwise && filters->is_pointwise) //1x1 point wise
    {
        CDataBlob_create(&filters->weights, 1, filters->num_filters, filters->channels);
    }
    else if(filters->is_depthwise && !filters->is_pointwise) //3x3 depth wise
    {
        CDataBlob_create(&filters->weights, 1, 9, filters->channels);
    }
    else
    {
        fprintf(stderr, "Unsupported filter type. Only 1x1 point-wise and 3x3 depth-wise are supported.");
        return;
    }
    CDataBlob_create(&filters->biases, 1, 1, filters->num_filters);
    //the format of convinfo.pWeights/biases must meet the format in this->weigths/biases
    for(int fidx = 0; fidx < filters->weights.cols; fidx++)
    {
        memcpy(CDataBlob_ptr(&filters->weights, 0, fidx),
                convinfo->pWeights + filters->channels * fidx ,
                filters->channels * sizeof(float));
    }

    memcpy(CDataBlob_ptr(&filters->biases, 0, 0), convinfo->pBiases, sizeof(float) * filters->num_filters);
}

void Filters_release(Filters* filters)
{
    CDataBlob_release(&filters->weights);
    CDataBlob_release(&filters->biases);
}

void setDataFrom3x3S2P1to1x1S1P0FromImage(const unsigned char* inputData, int imgWidth, int imgHeight, int imgChannels, int imgWidthStep, int is_rgb, int padDivisor,
                                          CDataBlob* outBlob)
{
    if (imgChannels != 3) {
        fprintf(stderr, "%s : The input image must be a 3-channel RGB image.\n", __FUNCTION__);
        exit(1);
    }
    if (padDivisor != 32) {
        fprintf(stderr, "%s : This version need pad of 32.\n", __FUNCTION__);
        exit(1);
    }
    int rows = ((imgHeight - 1) / padDivisor + 1) * padDivisor / 2;
    int cols = ((imgWidth - 1) / padDivisor + 1 ) * padDivisor / 2;
    int channels = 32;
    CDataBlob_create(outBlob, rows, cols, channels);

    if(0 != is_rgb)
    {
        #if defined(_OPENMP)
        #pragma omp parallel for
        #endif
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {;
                float* pData = CDataBlob_ptr(outBlob, r, c);
                for (int fy = -1; fy <= 1; fy++) {
                    int srcy = r * 2 + fy;

                    if (srcy < 0 || srcy >= imgHeight) //out of the range of the image
                        continue;

                    for (int fx = -1; fx <= 1; fx++) {
                        int srcx = c * 2 + fx;

                        if (srcx < 0 || srcx >= imgWidth) //out of the range of the image
                            continue;

                        const unsigned char * pImgData = inputData + imgWidthStep * srcy + imgChannels * srcx;

                        int output_channel_offset = ((fy + 1) * 3 + fx + 1) ; //3x3 filters, 3-channel image
                        pData[output_channel_offset * imgChannels] = pImgData[2];
                        pData[output_channel_offset * imgChannels + 1] = pImgData[1];
                        pData[output_channel_offset * imgChannels + 2] = pImgData[0];
                    }
                }
            }
        }
    }
    else
    {
        #if defined(_OPENMP)
        #pragma omp parallel for
        #endif
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {;
                float* pData = CDataBlob_ptr(outBlob, r, c);
                for (int fy = -1; fy <= 1; fy++) {
                    int srcy = r * 2 + fy;

                    if (srcy < 0 || srcy >= imgHeight) //out of the range of the image
                        continue;

                    for (int fx = -1; fx <= 1; fx++) {
                        int srcx = c * 2 + fx;

                        if (srcx < 0 || srcx >= imgWidth) //out of the range of the image
                            continue;

                        const unsigned char * pImgData = inputData + imgWidthStep * srcy + imgChannels * srcx;

                        int output_channel_offset = ((fy + 1) * 3 + fx + 1) ; //3x3 filters, 3-channel image
                        pData[output_channel_offset * imgChannels] = pImgData[0];
                        pData[output_channel_offset * imgChannels + 1] = pImgData[1];
                        pData[output_channel_offset * imgChannels + 2] = pImgData[2];
                    }
                }
            }
        }
    }
}

//p1 and p2 must be 512-bit aligned (16 float numbers)
float dotProduct(const float * p1, const float * p2, int num)
{
    float sum = 0.f;

#if defined(_ENABLE_AVX512)
    __m512 a_float_x16, b_float_x16;
    __m512 sum_float_x16 = _mm512_setzero_ps();
    for (int i = 0; i < num; i += 16)
    {
        a_float_x16 = _mm512_load_ps(p1 + i);
        b_float_x16 = _mm512_load_ps(p2 + i);
        sum_float_x16 = _mm512_add_ps(sum_float_x16, _mm512_mul_ps(a_float_x16, b_float_x16));
    }
   sum = _mm512_reduce_add_ps(sum_float_x16);
#elif defined(_ENABLE_AVX2)
    __m256 a_float_x8, b_float_x8;
    __m256 sum_float_x8 = _mm256_setzero_ps();
    for (int i = 0; i < num; i += 8)
    {
        a_float_x8 = _mm256_load_ps(p1 + i);
        b_float_x8 = _mm256_load_ps(p2 + i);
        sum_float_x8 = _mm256_add_ps(sum_float_x8, _mm256_mul_ps(a_float_x8, b_float_x8));
    }
   sum_float_x8 = _mm256_hadd_ps(sum_float_x8, sum_float_x8);
   sum_float_x8 = _mm256_hadd_ps(sum_float_x8, sum_float_x8);
   sum = ((float*)&sum_float_x8)[0] + ((float*)&sum_float_x8)[4];
#elif defined(_ENABLE_NEON)
    float32x4_t a_float_x4, b_float_x4;
    float32x4_t sum_float_x4;
    sum_float_x4 = vdupq_n_f32(0);
    for (int i = 0; i < num; i+=4)
    {
        a_float_x4 = vld1q_f32(p1 + i);
        b_float_x4 = vld1q_f32(p2 + i);
        sum_float_x4 = vaddq_f32(sum_float_x4, vmulq_f32(a_float_x4, b_float_x4));
    }
    sum += vgetq_lane_f32(sum_float_x4, 0);
    sum += vgetq_lane_f32(sum_float_x4, 1);
    sum += vgetq_lane_f32(sum_float_x4, 2);
    sum += vgetq_lane_f32(sum_float_x4, 3);
#else
    for(int i = 0; i < num; i++)
    {
        sum += (p1[i] * p2[i]);
    }
#endif

    return sum;
}

int vecMulAdd(const float * p1, const float * p2, float * p3, int num)
{
#if defined(_ENABLE_AVX512)
    __m512 a_float_x16, b_float_x16, c_float_x16;
    for (int i = 0; i < num; i += 16)
    {
        a_float_x16 = _mm512_load_ps(p1 + i);
        b_float_x16 = _mm512_load_ps(p2 + i);
        c_float_x16 = _mm512_load_ps(p3 + i);
        c_float_x16 = _mm512_add_ps(c_float_x16, _mm512_mul_ps(a_float_x16, b_float_x16));
        _mm512_store_ps(p3 + i, c_float_x16);
    }
#elif defined(_ENABLE_AVX2)
    __m256 a_float_x8, b_float_x8, c_float_x8;
    for (int i = 0; i < num; i += 8)
    {
        a_float_x8 = _mm256_load_ps(p1 + i);
        b_float_x8 = _mm256_load_ps(p2 + i);
        c_float_x8 = _mm256_load_ps(p3 + i);
        c_float_x8 = _mm256_add_ps(c_float_x8, _mm256_mul_ps(a_float_x8, b_float_x8));
        _mm256_store_ps(p3 + i, c_float_x8);
    }
#elif defined(_ENABLE_NEON)
    float32x4_t a_float_x4, b_float_x4, c_float_x4;
    for (int i = 0; i < num; i+=4)
    {
        a_float_x4 = vld1q_f32(p1 + i);
        b_float_x4 = vld1q_f32(p2 + i);
        c_float_x4 = vld1q_f32(p3 + i);
        c_float_x4 = vaddq_f32(c_float_x4, vmulq_f32(a_float_x4, b_float_x4));
        vst1q_f32(p3 + i, c_float_x4);
    }
#else
    for(int i = 0; i < num; i++)
        p3[i] += (p1[i] * p2[i]);
#endif

    return 1;
}

int vecAdd2(const float * p1, float * p2, int num)
{
#if defined(_ENABLE_AVX512)
    __m512 a_float_x16, b_float_x16;
    for (int i = 0; i < num; i += 16)
    {
        a_float_x16 = _mm512_load_ps(p1 + i);
        b_float_x16 = _mm512_load_ps(p2 + i);
        b_float_x16 = _mm512_add_ps(a_float_x16, b_float_x16);
        _mm512_store_ps(p2 + i, b_float_x16);
    }
#elif defined(_ENABLE_AVX2)
    __m256 a_float_x8, b_float_x8;
    for (int i = 0; i < num; i += 8)
    {
        a_float_x8 = _mm256_load_ps(p1 + i);
        b_float_x8 = _mm256_load_ps(p2 + i);
        b_float_x8 = _mm256_add_ps(a_float_x8, b_float_x8);
        _mm256_store_ps(p2 + i, b_float_x8);
    }
#elif defined(_ENABLE_NEON)
    float32x4_t a_float_x4, b_float_x4, c_float_x4;
    for (int i = 0; i < num; i+=4)
    {
        a_float_x4 = vld1q_f32(p1 + i);
        b_float_x4 = vld1q_f32(p2 + i);
        c_float_x4 = vaddq_f32(a_float_x4, b_float_x4);
        vst1q_f32(p2 + i, c_float_x4);
    }
#else
    for(int i = 0; i < num; i++)
    {
        p2[i] += p1[i];
    }
#endif
    return 1;
}

int vecAdd3(const float * p1, const float * p2, float* p3, int num)
{
#if defined(_ENABLE_AVX512)
    __m512 a_float_x16, b_float_x16;
    for (int i = 0; i < num; i += 16)
    {
        a_float_x16 = _mm512_load_ps(p1 + i);
        b_float_x16 = _mm512_load_ps(p2 + i);
        b_float_x16 = _mm512_add_ps(a_float_x16, b_float_x16);
        _mm512_store_ps(p3 + i, b_float_x16);
    }
#elif defined(_ENABLE_AVX2)
    __m256 a_float_x8, b_float_x8;
    for (int i = 0; i < num; i += 8)
    {
        a_float_x8 = _mm256_load_ps(p1 + i);
        b_float_x8 = _mm256_load_ps(p2 + i);
        b_float_x8 = _mm256_add_ps(a_float_x8, b_float_x8);
        _mm256_store_ps(p3 + i, b_float_x8);
    }
#elif defined(_ENABLE_NEON)
    float32x4_t a_float_x4, b_float_x4, c_float_x4;
    for (int i = 0; i < num; i+=4)
    {
        a_float_x4 = vld1q_f32(p1 + i);
        b_float_x4 = vld1q_f32(p2 + i);
        c_float_x4 = vaddq_f32(a_float_x4, b_float_x4);
        vst1q_f32(p3 + i, c_float_x4);
    }
#else
    for(int i = 0; i < num; i++)
    {
        p3[i] = p1[i] + p2[i];
    }
#endif
    return 1;
}

int convolution_1x1pointwise(const CDataBlob* inputData, const Filters* filters, CDataBlob* outputData)
{
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (int row = 0; row < outputData->rows; row++)
    {
        for (int col = 0; col < outputData->cols; col++)
        {;
            float * pOut = CDataBlob_ptr(outputData, row, col);
            const float * pIn = CDataBlob_ptr(inputData, row, col);
            for (int ch = 0; ch < outputData->channels; ch++)
            {
                const float * pF = CDataBlob_ptr(&filters->weights, 0, ch);
                pOut[ch] = dotProduct(pIn, pF, inputData->channels);
                pOut[ch] += filters->biases.data[ch];
            }
        }
    }
    return 1;
}

int convolution_3x3depthwise(const CDataBlob* inputData, const Filters* filters, CDataBlob* outputData)
{
    //set all elements in outputData to zeros
    CDataBlob_setZero(outputData);
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (int row = 0; row < outputData->rows; row++)
    {  
        int srcy_start = row - 1;
        int srcy_end = srcy_start + 3;
        srcy_start = MAX(0, srcy_start);
        srcy_end = MIN(srcy_end, inputData->rows);

        for (int col = 0; col < outputData->cols; col++)
        {
            float * pOut = CDataBlob_ptr(outputData, row, col);
            int srcx_start = col - 1;
            int srcx_end = srcx_start + 3;
            srcx_start = MAX(0, srcx_start);
            srcx_end = MIN(srcx_end, inputData->cols);

            for ( int r = srcy_start; r < srcy_end; r++)
                for( int c = srcx_start; c < srcx_end; c++)
                {
                    int filter_r = r - row + 1;
                    int filter_c = c - col + 1;
                    int filter_idx = filter_r * 3 + filter_c;
                    vecMulAdd(CDataBlob_ptr(inputData, r, c),
                              CDataBlob_ptr(&filters->weights, 0, filter_idx),
                              pOut,filters->num_filters);
                }
            vecAdd2(CDataBlob_ptr(&filters->biases, 0, 0), pOut, filters->num_filters);
        }
    }
     return 1;
}

int relu(CDataBlob* inputoutputData)
{
    if(1 == CDataBlob_isEmpty(inputoutputData))
    {
        fprintf(stderr,"%s : The input data is empty.\n", __FUNCTION__);
        return 0;
    }
    
    int len = inputoutputData->cols * inputoutputData->rows * inputoutputData->channelStep / sizeof(float);


#if defined(_ENABLE_AVX512)
    __m512 a, bzeros;
    bzeros = _mm512_setzero_ps(); //zeros
    for( int i = 0; i < len; i+=16)
    {
        a = _mm512_load_ps(inputoutputData->data + i);
        a = _mm512_max_ps(a, bzeros);
        _mm512_store_ps(inputoutputData->data + i, a);
    }
#elif defined(_ENABLE_AVX2)
    __m256 a, bzeros;
    bzeros = _mm256_setzero_ps(); //zeros
    for( int i = 0; i < len; i+=8)
    {
        a = _mm256_load_ps(inputoutputData->data + i);
        a = _mm256_max_ps(a, bzeros);
        _mm256_store_ps(inputoutputData->data + i, a);
    }
#else    
    for( int i = 0; i < len; i++)
        inputoutputData->data[i] *= (inputoutputData->data[i] >0);
#endif

    return 1;
}

void IntersectBBox(const NormalizedBBox* bbox1, const NormalizedBBox* bbox2,
                   NormalizedBBox* intersect_bbox) 
{
    if (bbox2->xmin > bbox1->xmax || bbox2->xmax < bbox1->xmin ||
        bbox2->ymin > bbox1->ymax || bbox2->ymax < bbox1->ymin)
    {
        // Return [0, 0, 0, 0] if there is no intersection.
        intersect_bbox->xmin = 0;
        intersect_bbox->ymin = 0;
        intersect_bbox->xmax = 0;
        intersect_bbox->ymax = 0;
    }
    else
    {
        intersect_bbox->xmin = (MAX(bbox1->xmin, bbox2->xmin));
        intersect_bbox->ymin = (MAX(bbox1->ymin, bbox2->ymin));
        intersect_bbox->xmax = (MIN(bbox1->xmax, bbox2->xmax));
        intersect_bbox->ymax = (MIN(bbox1->ymax, bbox2->ymax));
    }
}

float JaccardOverlap(const NormalizedBBox* bbox1, const NormalizedBBox* bbox2)
{
    NormalizedBBox intersect_bbox;
    IntersectBBox(bbox1, bbox2, &intersect_bbox);
    float intersect_width, intersect_height;
    intersect_width = intersect_bbox.xmax - intersect_bbox.xmin;
    intersect_height = intersect_bbox.ymax - intersect_bbox.ymin;

    if (intersect_width > 0 && intersect_height > 0) 
    {
        float intersect_size = intersect_width * intersect_height;
        float bsize1 = (bbox1->xmax - bbox1->xmin)*(bbox1->ymax - bbox1->ymin);
        float bsize2 = (bbox2->xmax - bbox2->xmin)*(bbox2->ymax - bbox2->ymin);
        return intersect_size / ( bsize1 + bsize2 - intersect_size);
    }
    else 
    {
        return 0.f;
    }
}

int SortScoreBBoxPairDescend(const void* pair1,   const void* pair2)
{
    const NormalizedBBox* arg1 = (const NormalizedBBox*)pair1;
    const NormalizedBBox* arg2 = (const NormalizedBBox*)pair2;

    return (arg1->score > arg2->score) ? -1 : 1;
}


void upsampleX2(const CDataBlob* inputData, CDataBlob* outData) {
    if (1 == CDataBlob_isEmpty((CDataBlob*)(inputData))) {
        fprintf(stderr,"%s : The input data is empty.\n", __FUNCTION__);
        exit(1);
    }
    CDataBlob_create(outData, inputData->rows * 2, inputData->cols * 2, inputData->channels);
    for (int r = 0; r < inputData->rows; r++) {
        for (int c = 0; c < inputData->cols; c++) {
            const float * pIn = CDataBlob_ptr(inputData, r, c);
            int outr = r * 2;
            int outc = c * 2;
            for (int ch = 0; ch < inputData->channels; ++ch) {
                CDataBlob_ptr(outData, outr, outc)[ch] = pIn[ch];
                CDataBlob_ptr(outData,outr, outc + 1)[ch] = pIn[ch];
                CDataBlob_ptr(outData,outr + 1, outc)[ch] = pIn[ch];
                CDataBlob_ptr(outData,outr + 1, outc + 1)[ch] = pIn[ch];
            }
        }
    }
}

void elementAdd(const CDataBlob* inputData1, const CDataBlob* inputData2,
                CDataBlob* outData) {
    if (inputData1->rows != inputData2->rows || inputData1->cols != inputData2->cols || inputData1->channels != inputData2->channels) {
        fprintf(stderr, "%s : The two input datas must be in the same shape.\n", __FUNCTION__);
        exit(1);
    }
    CDataBlob_create(outData, inputData1->rows, inputData1->cols, inputData1->channels);
    for (int r = 0; r < inputData1->rows; r++) {
        for (int c = 0; c < inputData1->cols; c++) {
            const float * pIn1 = CDataBlob_ptr(inputData1, r, c);
            const float * pIn2 = CDataBlob_ptr(inputData2, r, c);
            float* pOut = CDataBlob_ptr(outData, r, c);
            vecAdd3(pIn1, pIn2, pOut, inputData1->channels);
        }
    }
}

void convolution(const CDataBlob* inputData, const Filters* filters, int do_relu,
                 CDataBlob* outputData)
{
    if(1 == CDataBlob_isEmpty(inputData) || 1 == CDataBlob_isEmpty(&filters->weights) || 1 == CDataBlob_isEmpty(&filters->biases))
    {
        fprintf(stderr, "%s : The input data or filter data is empty\n", __FUNCTION__);
        exit(1);
    }
    if(inputData->channels != filters->channels)
    {
        fprintf(stderr, "%s : The input data dimension cannot meet filters: %d vs %d\n", __FUNCTION__, inputData->channels, filters->channels);
        exit(1);
    }

    CDataBlob_create(outputData, inputData->rows, inputData->cols, filters->num_filters);
    if(filters->is_pointwise && !filters->is_depthwise)
    {
        convolution_1x1pointwise(inputData, filters, outputData);
    }
    else if(!filters->is_pointwise && filters->is_depthwise)
    {
        convolution_3x3depthwise(inputData, filters, outputData);
    }
    else
    {
        fprintf(stderr, "%s :  Unsupported filter type.\n",__FUNCTION__);
        exit(1);
    }

    if(do_relu)
        relu(outputData);
}

static CDataBlob __g_blob_in_convolutionDP__ = {0, 0, 0, 0, 0, 0, NULL, NULL};
void convolutionDP(const CDataBlob* inputData, const Filters* filtersP, const Filters* filtersD, int do_relu,
                   CDataBlob* outputData)
{

//    printf("00000000000000 : %p, %d\n", tmp.data, tmp.totalCapacity);
    convolution(inputData, filtersP, 0, &__g_blob_in_convolutionDP__);
    convolution(&__g_blob_in_convolutionDP__, filtersD, do_relu, outputData);
}

static CDataBlob __g_blob_in_convolution4layerUnit__ = {0, 0, 0, 0, 0, 0, NULL, NULL};
void convolution4layerUnit(const CDataBlob* inputData,
                const Filters* filtersP1, const Filters* filtersD1,
                const Filters* filtersP2, const Filters* filtersD2, int do_relu,
                CDataBlob* outputData)
{
    convolutionDP(inputData, filtersP1, filtersD1, 1, &__g_blob_in_convolution4layerUnit__);
    convolutionDP(&__g_blob_in_convolution4layerUnit__, filtersP2, filtersD2, do_relu, outputData);
}

//only 2X2 S2 is supported
void maxpooling2x2S2(const CDataBlob* inputData,
                     CDataBlob* outputData)
{
    if (1 == CDataBlob_isEmpty(inputData))
    {
        fprintf(stderr,"%s : The input data is empty.\n", __FUNCTION__);
        exit(1);
    }
    int outputR = (int)(ceil((inputData->rows - 3.0f) / 2)) + 1;
    int outputC = (int)(ceil((inputData->cols - 3.0f) / 2)) + 1;
    int outputCH = inputData->channels;

    if (outputR < 1 || outputC < 1)
    {
        fprintf(stderr,"%s : The size of the output is not correct. (%d, %d)\n", __FUNCTION__, outputR, outputC);
        exit(1);        
    }
    CDataBlob_create(outputData, outputR, outputC, outputCH);

    for (int row = 0; row < outputData->rows; row++)
    {
        for (int col = 0; col < outputData->cols; col++)
        {
            size_t inputMatOffsetsInElement[4];
            int elementCount = 0;

            int rstart = row * 2;
            int cstart = col * 2;
            int rend = MIN(rstart + 2, inputData->rows);
            int cend = MIN(cstart + 2, inputData->cols);

            for (int fr = rstart; fr < rend; fr++)
            {
                for (int fc = cstart; fc < cend; fc++)
                {
                    inputMatOffsetsInElement[elementCount++] = (fr * inputData->cols + fc) * inputData->channelStep / sizeof(float);
                }
            }

            float * pOut = CDataBlob_ptr(outputData, row, col);
            float * pIn = inputData->data;

#if defined(_ENABLE_NEON)
            for (int ch = 0; ch < outputData->channels; ch += 4)
            {
                float32x4_t tmp;
                float32x4_t maxVal = vld1q_f32(pIn + ch + inputMatOffsetsInElement[0]);
                for (int ec = 1; ec < elementCount; ec++)
                {
                    tmp = vld1q_f32(pIn + ch + inputMatOffsetsInElement[ec]);
                    maxVal = vmaxq_f32(maxVal, tmp);
                }
                vst1q_f32(pOut + ch, maxVal);
            }
#elif defined(_ENABLE_AVX512)
            for (int ch = 0; ch < outputData->channels; ch += 16)
            {
                __m512 tmp;
                __m512 maxVal = _mm512_load_ps((__m512 const*)(pIn + ch + inputMatOffsetsInElement[0]));
                for (int ec = 1; ec < elementCount; ec++)
                {
                    tmp = _mm512_load_ps((__m512 const*)(pIn + ch + inputMatOffsetsInElement[ec]));
                    maxVal = _mm512_max_ps(maxVal, tmp);
                }
                _mm512_store_ps((__m512*)(pOut + ch), maxVal);
            }
#elif defined(_ENABLE_AVX2)
            for (int ch = 0; ch < outputData->channels; ch += 8)
            {
                __m256 tmp;
                __m256 maxVal = _mm256_load_ps((float const*)(pIn + ch + inputMatOffsetsInElement[0]));
                for (int ec = 1; ec < elementCount; ec++)
                {
                    tmp = _mm256_load_ps((float const*)(pIn + ch + inputMatOffsetsInElement[ec]));
                    maxVal = _mm256_max_ps(maxVal, tmp);
                }
                _mm256_store_ps(pOut + ch, maxVal);
            }
#else
            for (int ch = 0; ch < outputData->channels; ch++)
            {
                float maxVal = pIn[ch + inputMatOffsetsInElement[0]];
                for (int ec = 1; ec < elementCount; ec++)
                {
                    maxVal = MAX(maxVal, pIn[ch + inputMatOffsetsInElement[ec]]);
                }
                pOut[ch] = maxVal;
            }
#endif
        }
    }
}

void meshgrid(int feature_width, int feature_height, int stride, float offset,
                CDataBlob* out) {
    CDataBlob_create(out, feature_height, feature_width, 2);
    for(int r = 0; r < feature_height; ++r) {
        float rx = (float)(r * stride) + offset;
        for(int c = 0; c < feature_width; ++c) {
            float* p = CDataBlob_ptr(out, r, c);
            p[0] = (float)(c * stride) + offset;
            p[1] = rx;
        }
    }
}

void bbox_decode(CDataBlob* bbox_pred, const CDataBlob* priors, int stride) {
    if(bbox_pred->cols != priors->cols || bbox_pred->rows != priors->rows) {
        fprintf(stderr, "%s : Mismatch between feature map and anchor size. (%d, %d) vs (%d, %d)\n", __FUNCTION__,
                bbox_pred->rows, bbox_pred->cols,
                priors->rows, priors->cols);
    }
    if(4 != bbox_pred->channels) {
        fprintf(stderr, "%s : The bbox dim must be 4.\n", __FUNCTION__);
    }
    float fstride = (float)stride;
    for(int r = 0; r < bbox_pred->rows; ++r) {
        for(int c = 0; c < bbox_pred->cols; ++c) {
            float* pb = CDataBlob_ptr(bbox_pred, r, c);
            const float* pp = CDataBlob_ptr(priors, r, c);
            float cx = pb[0] * fstride + pp[0];
            float cy = pb[1] * fstride + pp[1];
            float w = expf(pb[2]) * fstride;
            float h = expf(pb[3]) * fstride;
            pb[0] = cx - w / 2.f;
            pb[1] = cy - h / 2.f;
            pb[2] = cx + w / 2.f;
            pb[3] = cy + h / 2.f;
        }
    }
}

void kps_decode(CDataBlob* kps_pred, const CDataBlob* priors, int stride) {
    if(kps_pred->cols != priors->cols || kps_pred->rows != priors->rows) {
        fprintf(stderr, "%s : Mismatch between feature map and anchor size.\n", __FUNCTION__ );
        exit(1);
    }
    if(kps_pred->channels & 1) {
        fprintf(stderr, "%s : The kps dim must be even.\n", __FUNCTION__ );
        exit(1);
    }
    float fstride = (float)stride;
    int num_points = kps_pred->channels >> 1;

    for(int r = 0; r < kps_pred->rows; ++r) {
        for(int c = 0; c < kps_pred->cols; ++c) {
            float* pb = CDataBlob_ptr(kps_pred, r, c);
            const float* pp = CDataBlob_ptr(priors, r, c);
            for(int n = 0; n < num_points; ++n) {
                pb[2 * n] = pb[2 * n] * fstride + pp[0];
                pb[2 * n + 1] = pb[2 * n + 1] * fstride + pp[1];           
            }
        }
    }
}

void concat3(const CDataBlob* inputData1, const CDataBlob* inputData2, const CDataBlob* inputData3,
             CDataBlob* outputData)
{
    if ((1 == CDataBlob_isEmpty(inputData1)) || (1 == CDataBlob_isEmpty(inputData2)) || (1 == CDataBlob_isEmpty(inputData3)))
    {
        fprintf(stderr, "%s : The input data is empty.\n", __FUNCTION__ );
        exit(1);
    }

    if ((inputData1->cols != inputData2->cols) ||
        (inputData1->rows != inputData2->rows) ||
        (inputData1->cols != inputData3->cols) ||
        (inputData1->rows != inputData3->rows))
    {
        fprintf(stderr, "%s : The three inputs must have the same size.\n", __FUNCTION__ );
        exit(1);
    }
    int outputR = inputData1->rows;
    int outputC = inputData1->cols;
    int outputCH = inputData1->channels + inputData2->channels + inputData3->channels;

    if (outputR < 1 || outputC < 1 || outputCH < 1)
    {
        fprintf(stderr, "%s : The size of the output is not correct. (%d, %d, %d)\n", __FUNCTION__ ,outputR, outputC, outputCH);
        exit(1);
    }

    CDataBlob_create(outputData, outputR, outputC, outputCH);

    for (int row = 0; row < outputData->rows; row++)
    {
        for (int col = 0; col < outputData->cols; col++)
        {
            float * pOut = CDataBlob_ptr(outputData, row, col);
            const float* pIn1 = CDataBlob_ptr(inputData1, row, col);
            const float* pIn2 = CDataBlob_ptr(inputData2, row, col);
            const float* pIn3 = CDataBlob_ptr(inputData3, row, col);

            memcpy(pOut, pIn1, sizeof(float) * inputData1->channels);
            memcpy(pOut + inputData1->channels, pIn2, sizeof(float)* inputData2->channels);
            memcpy(pOut + inputData1->channels + inputData2->channels, pIn3, sizeof(float)* inputData3->channels);
        }
    }
}

void blob2vector(const CDataBlob* inputData,
                 CDataBlob* outputData)
{
    if (1 == CDataBlob_isEmpty(inputData))
    {
        fprintf(stderr, "%s : The input data is empty.\n", __FUNCTION__ );
        exit(1);
    }

    CDataBlob_create(outputData, 1, 1, inputData->rows * inputData->cols * inputData->channels);

    int bytesOfAChannel = inputData->channels * sizeof(float);
    float* pOut = CDataBlob_ptr(outputData, 0, 0);
    for (int row = 0; row < inputData->rows; row++)
    {
        for (int col = 0; col < inputData->cols; col++)
        {
            const float* pIn = CDataBlob_ptr(inputData, row, col);
            memcpy(pOut, pIn, bytesOfAChannel);
            pOut += inputData->channels;
        }
    }
}

void sigmoid(CDataBlob* inputData) {
    for(int r = 0; r < inputData->rows; ++r) {
        for(int c = 0; c < inputData->cols; ++c) {
            float* pIn = CDataBlob_ptr(inputData, r,c);
            for(int ch = 0; ch < inputData->channels; ++ch) {
                float v = pIn[ch];
                v = MIN(v, 88.3762626647949f);
                v = MAX(v, -88.3762626647949f);
                pIn[ch] = (float)(1.f / (1.f + expf(-v)));
            }
        }
    }
}


void detection_output(const CDataBlob* cls,
                      const CDataBlob* reg,
                      const CDataBlob* kps,
                      const CDataBlob* obj,
                      float overlap_threshold,
                      float confidence_threshold,
                      int top_k,
                      int keep_top_k,
                      CDataBlob* face_blob, int* num_faces)
{
    if (CDataBlob_isEmpty(reg) || CDataBlob_isEmpty(cls) || CDataBlob_isEmpty(kps) ||
        CDataBlob_isEmpty(obj))//|| iou.isEmpty())
    {
        fprintf(stderr, "%s : The input data is null.\n", __FUNCTION__);
        exit(1);
    }
    if (reg->cols != 1 || reg->rows != 1 || cls->cols != 1 || cls->rows != 1 || kps->cols != 1 || kps->rows != 1 ||
        obj->cols != 1 || obj->rows != 1)
    {
        fprintf(stderr, "%s : Only support vector format.\n", __FUNCTION__);
        exit(1);
    }

    if ((int) (kps->channels / obj->channels) != 10)
    {
        fprintf(stderr, "%s : Only support 5 keypoints. (%d)\n", __FUNCTION__, kps->channels);
        exit(1);
    }

    const float *pCls = CDataBlob_ptr(cls, 0, 0);
    const float *pReg = CDataBlob_ptr(reg, 0, 0);
    const float *pObj = CDataBlob_ptr(obj, 0, 0);
    const float *pKps = CDataBlob_ptr(kps, 0, 0);

    int valid_count = 0;
    //get the candidates those are > confidence_threshold
    for (int i = 0; i < cls->channels; ++i)
    {
        float conf = sqrtf(pCls[i] * pObj[i]);
        // float conf = pCls[i] * pObj[i];

        if (conf >= confidence_threshold)
        {
            valid_count++;
        }
    }
    NormalizedBBox *score_bbox_vec = NULL;
    size_t score_bbox_vec_alloc_size = valid_count * sizeof(NormalizedBBox);
    if(__g_blob_in_convolutionDP__.totalCapacity < score_bbox_vec_alloc_size)
    {
        score_bbox_vec = (NormalizedBBox *) myAlloc(score_bbox_vec_alloc_size);
    }
    else
    {
        score_bbox_vec = (NormalizedBBox*)(__g_blob_in_convolutionDP__.data);
    }

    valid_count = 0;
    //get the candidates those are > confidence_threshold
    for (int i = 0; i < cls->channels; ++i)
    {
        float conf = sqrtf(pCls[i] * pObj[i]);
        // float conf = pCls[i] * pObj[i];

        if (conf >= confidence_threshold)
        {
            NormalizedBBox bb;
            bb.xmin = pReg[4 * i];
            bb.ymin = pReg[4 * i + 1];
            bb.xmax = pReg[4 * i + 2];
            bb.ymax = pReg[4 * i + 3];
            bb.score = conf;

            //store the five landmarks
            memcpy(bb.lm, pKps + 10 * i, 10 * sizeof(float));
            score_bbox_vec[valid_count++] = bb;
        }
    }

    //Sort the score pair according to the scores in descending order
    qsort(score_bbox_vec, valid_count, sizeof(NormalizedBBox), SortScoreBBoxPairDescend);

    // Keep top_k scores if needed.
    if (top_k > -1 && top_k < valid_count)
    {
//        score_bbox_vec.resize(top_k);
        valid_count = top_k;
    }

    //Do NMS
    NormalizedBBox *final_score_bbox_vec = NULL;
    size_t final_score_bbox_vec_alloc_size = keep_top_k * sizeof(NormalizedBBox);
    if(__g_blob_in_convolution4layerUnit__.totalCapacity < final_score_bbox_vec_alloc_size)
    {
        final_score_bbox_vec = (NormalizedBBox *) myAlloc(final_score_bbox_vec_alloc_size);
    }
    else
    {
        final_score_bbox_vec = (NormalizedBBox*)(__g_blob_in_convolution4layerUnit__.data);
    }
    int final_count = 0;
    for (int idx = 0; idx < valid_count; idx++)
    {
        const NormalizedBBox *bb1 = &score_bbox_vec[idx];
        int keep = 1;
        for (size_t k = 0; k < final_count && k < keep_top_k; k++)
        {
            if (keep)
            {
                const NormalizedBBox *bb2 = &final_score_bbox_vec[k];
                float overlap = JaccardOverlap(bb1, bb2);
                keep = (overlap <= overlap_threshold);
            } else
            {
                break;
            }
        }
        if (keep)
        {
            final_score_bbox_vec[final_count++] = *bb1;
        }
    }

    FaceRect *face_rects = (FaceRect*)(face_blob->data);
    size_t face_rects_alloc_size = final_count * sizeof(FaceRect);
    if(face_blob->totalCapacity < face_rects_alloc_size)
    {
        final_count = (int)(face_blob->totalCapacity / sizeof(FaceRect));
    }
    *num_faces = final_count;
    for (int fi = 0; fi < final_count; fi++)
    {
        const NormalizedBBox *pp = &final_score_bbox_vec[fi];

        FaceRect r;
        r.score = pp->score;
        r.x = (int) (pp->xmin);
        r.y = (int) (pp->ymin);
        r.w = (int) (pp->xmax - pp->xmin);
        r.h = (int) (pp->ymax - pp->ymin);
        //copy landmark data
        for (int i = 0; i < 10; ++i)
        {
            r.lm[i] = (int) (pp->lm[i]);
        }
        face_rects[fi] = r;
    }
    if(__g_blob_in_convolutionDP__.totalCapacity < score_bbox_vec_alloc_size)
    {
        myFree(&score_bbox_vec);
    }
    if(__g_blob_in_convolution4layerUnit__.totalCapacity < final_score_bbox_vec_alloc_size)
    {
        myFree(&final_score_bbox_vec);
    }
}

void deinit_middle_blobs()
{
    CDataBlob_release(&__g_blob_in_convolutionDP__);
    CDataBlob_release(&__g_blob_in_convolution4layerUnit__);
}