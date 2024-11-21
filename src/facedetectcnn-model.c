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
#include "facedetectcnn.h"

#if 0
#include <opencv2/opencv.hpp>
cv::TickMeter cvtm;
#define TIME_START cvtm.reset();cvtm.start();
#define TIME_END(FUNCNAME) cvtm.stop(); printf(FUNCNAME);printf("=%g\n", cvtm.getTimeMilli());
#else
#define TIME_START
#define TIME_END(FUNCNAME)
#endif


#define NUM_CONV_LAYER 53
extern ConvInfoStruct param_pConvInfo[NUM_CONV_LAYER];
static Filters g_pFilters[NUM_CONV_LAYER];
static int g_param_initialized = 0;
void init_parameters()
{
    for(int idx = 0; idx < NUM_CONV_LAYER; idx++)
    {
        Filters_create(&g_pFilters[idx], &param_pConvInfo[idx]);
    }
}

void deinit_parameters()
{
    for(int idx = 0; idx < NUM_CONV_LAYER; idx++)
    {
        Filters_release(&g_pFilters[idx]);
    }
    g_param_initialized = 0;
}

#define NUM_MIDDLE_TMP_BLOB 50
static CDataBlob g_pBlob[NUM_MIDDLE_TMP_BLOB];
void init_blob()
{
    for(int idx = 0; idx < NUM_MIDDLE_TMP_BLOB; idx++)
    {
        g_pBlob[idx].rows = 0;
        g_pBlob[idx].cols = 0;
        g_pBlob[idx].channels = 0;
        g_pBlob[idx].channelStep = 0;
        g_pBlob[idx].totalCapacity = 0;
        g_pBlob[idx].flag = 0;
        g_pBlob[idx].data = NULL;

        char blob_name[256] = {'\0'};
        sprintf(blob_name, "g_pBlob_%d", idx);
        g_pBlob[idx].name = blob_name;
    }
}

void deinit_blob()
{
    for(int idx = 0; idx < NUM_MIDDLE_TMP_BLOB; idx++)
    {
        CDataBlob_release(&g_pBlob[idx]);// CDataBlob_release
    }
}

void init_facedetect_resources()
{
    init_parameters();
    init_blob();
}

void release_facedetect_resources()
{
    deinit_parameters();
    deinit_blob();
    deinit_middle_blobs();
}
void objectdetect_cnn(const unsigned char * rgbImageData, int width, int height, int step, int is_rgb, float thresh,
                     CDataBlob* face_blob , int* num_faces)
{
    TIME_START;
    /*auto fx = */setDataFrom3x3S2P1to1x1S1P0FromImage(rgbImageData, width, height, 3, step, is_rgb, 32, &g_pBlob[0]);
    TIME_END("convert data");

    /***************CONV0*********************/
    TIME_START;
    /*fx = */convolution(&g_pBlob[0], &g_pFilters[0], 1, &g_pBlob[1]);
    CDataBlob_release(&g_pBlob[0]);
    TIME_END("conv_head");

    TIME_START;
    /*fx = */convolutionDP(&g_pBlob[1], &g_pFilters[1], &g_pFilters[2], 1, &g_pBlob[2]);
    CDataBlob_release(&g_pBlob[1]);
    TIME_END("conv0");

    TIME_START;
    /*fx = */maxpooling2x2S2(&g_pBlob[2], &g_pBlob[3]);
    CDataBlob_release(&g_pBlob[2]);
    TIME_END("pool0");

    /***************CONV1*********************/
    TIME_START;
    /*fx = */convolution4layerUnit(&g_pBlob[3], &g_pFilters[3], &g_pFilters[4], &g_pFilters[5], &g_pFilters[6], 1, &g_pBlob[4]);
    CDataBlob_release(&g_pBlob[3]);
    TIME_END("conv1");

    /***************CONV2*********************/
    TIME_START;
    /*fx = */convolution4layerUnit(&g_pBlob[4], &g_pFilters[7], &g_pFilters[8], &g_pFilters[9], &g_pFilters[10], 1, &g_pBlob[5]);
    CDataBlob_release(&g_pBlob[4]);
    TIME_END("conv2");

    /***************CONV3*********************/
    TIME_START;
    /*fx = */maxpooling2x2S2(&g_pBlob[5], &g_pBlob[6]);
    CDataBlob_release(&g_pBlob[5]);
    TIME_END("pool3");

    TIME_START;
    /*auto fb1 = */convolution4layerUnit(&g_pBlob[6], &g_pFilters[11], &g_pFilters[12], &g_pFilters[13], &g_pFilters[14], 1, &g_pBlob[7]);
    CDataBlob_release(&g_pBlob[6]);
    TIME_END("conv3");

    /***************CONV4*********************/
    TIME_START;
    /*fx = */maxpooling2x2S2(&g_pBlob[7], &g_pBlob[8]);
    TIME_END("pool4");

    TIME_START;
    /*auto fb2 = */convolution4layerUnit(&g_pBlob[8], &g_pFilters[15], &g_pFilters[16], &g_pFilters[17], &g_pFilters[18], 1, &g_pBlob[9]);
    CDataBlob_release(&g_pBlob[8]);
    TIME_END("conv4");

    /***************CONV5*********************/
    TIME_START;
    /* fx = */maxpooling2x2S2(&g_pBlob[9], &g_pBlob[10]);
    TIME_END("pool5");

    TIME_START;
    /*auto fb3 = */convolution4layerUnit(&g_pBlob[10], &g_pFilters[19], &g_pFilters[20], &g_pFilters[21], &g_pFilters[22], 1, &g_pBlob[11]);
    CDataBlob_release(&g_pBlob[10]);
    TIME_END("conv5");

    /***************branch5*********************/
    TIME_START;
    /*fb3 = */convolutionDP(&g_pBlob[11], &g_pFilters[27], &g_pFilters[28], 1, &g_pBlob[24]);
    CDataBlob_release(&g_pBlob[11]);
    /*pred_cls[2] = */convolutionDP(&g_pBlob[24], &g_pFilters[33], &g_pFilters[34], 0, &g_pBlob[17]);
    /*pred_reg[2] = */convolutionDP(&g_pBlob[24], &g_pFilters[39], &g_pFilters[40], 0, &g_pBlob[14]);
    /*pred_kps[2] = */convolutionDP(&g_pBlob[24], &g_pFilters[51], &g_pFilters[52], 0, &g_pBlob[20]);
    /*pred_obj[2] = */convolutionDP(&g_pBlob[24], &g_pFilters[45], &g_pFilters[46], 0, &g_pBlob[23]);
    TIME_END("branch5");

    /*****************add5*********************/    
    TIME_START;
    upsampleX2(&g_pBlob[24], &g_pBlob[25]);
    /*fb2 = */elementAdd(&g_pBlob[25], &g_pBlob[9], &g_pBlob[26]);
    CDataBlob_release(&g_pBlob[25]);
    CDataBlob_release(&g_pBlob[9]);
    TIME_END("add5");

    /*****************add6*********************/    
    TIME_START;
    /*fb2 = */convolutionDP(&g_pBlob[26], &g_pFilters[25], &g_pFilters[26], 1, &g_pBlob[27]);
    CDataBlob_release(&g_pBlob[26]);
    /*pred_cls[1] = */convolutionDP(&g_pBlob[27], &g_pFilters[31], &g_pFilters[32], 0, &g_pBlob[16]);
    /*pred_reg[1] = */convolutionDP(&g_pBlob[27], &g_pFilters[37], &g_pFilters[38], 0, &g_pBlob[13]);
    /*pred_kps[1] = */convolutionDP(&g_pBlob[27], &g_pFilters[49], &g_pFilters[50], 0, &g_pBlob[19]);
    /*pred_obj[1] = */convolutionDP(&g_pBlob[27], &g_pFilters[43], &g_pFilters[44], 0, &g_pBlob[22]);
    TIME_END("branch4");

    /*****************add4*********************/
    TIME_START;
    upsampleX2(&g_pBlob[27], &g_pBlob[28]);
    /*fb1 = */elementAdd(&g_pBlob[28], &g_pBlob[7], &g_pBlob[29]);
    CDataBlob_release(&g_pBlob[28]);
    CDataBlob_release(&g_pBlob[7]);
    TIME_END("add4");

    /***************branch3*********************/
    TIME_START;
    /*fb1 = */convolutionDP(&g_pBlob[29], &g_pFilters[23], &g_pFilters[24], 1, &g_pBlob[30]);
    CDataBlob_release(&g_pBlob[29]);
    /*pred_cls[0] = */convolutionDP(&g_pBlob[30], &g_pFilters[29], &g_pFilters[30], 0, &g_pBlob[15]);
    /*pred_reg[0] = */convolutionDP(&g_pBlob[30], &g_pFilters[35], &g_pFilters[36], 0, &g_pBlob[12]);
    /*pred_kps[0] = */convolutionDP(&g_pBlob[30], &g_pFilters[47], &g_pFilters[48], 0, &g_pBlob[18]);
    /*pred_obj[0] = */convolutionDP(&g_pBlob[30], &g_pFilters[41], &g_pFilters[42], 0, &g_pBlob[21]);
    TIME_END("branch3");

    /***************PRIORBOX*********************/
    TIME_START;
    /*auto prior3 = */meshgrid(g_pBlob[30].cols, g_pBlob[30].rows, 8, 0.0f, &g_pBlob[31]);
    /*auto prior4 = */meshgrid(g_pBlob[27].cols, g_pBlob[27].rows, 16, 0.0f,  &g_pBlob[32]);
    /*auto prior5 = */meshgrid(g_pBlob[24].cols, g_pBlob[24].rows, 32, 0.0f, &g_pBlob[33]);
    CDataBlob_release(&g_pBlob[30]);
    CDataBlob_release(&g_pBlob[27]);
    CDataBlob_release(&g_pBlob[24]);
    TIME_END("prior");
    /***************PRIORBOX*********************/
    // release 12-23, 31-48
    TIME_START;
    bbox_decode(&g_pBlob[12], &g_pBlob[31], 8);
    bbox_decode(&g_pBlob[13], &g_pBlob[32], 16);
    bbox_decode(&g_pBlob[14], &g_pBlob[33], 32);

    kps_decode(&g_pBlob[18], &g_pBlob[31], 8);
    kps_decode(&g_pBlob[19], &g_pBlob[32], 16);
    kps_decode(&g_pBlob[20], &g_pBlob[33], 32);
    CDataBlob_release(&g_pBlob[31]);
    CDataBlob_release(&g_pBlob[32]);
    CDataBlob_release(&g_pBlob[33]);

    blob2vector(&g_pBlob[15], &g_pBlob[34]);
    blob2vector(&g_pBlob[16], &g_pBlob[35]);
    blob2vector(&g_pBlob[17], &g_pBlob[36]);
    CDataBlob_release(&g_pBlob[15]);
    CDataBlob_release(&g_pBlob[16]);
    CDataBlob_release(&g_pBlob[17]);

    /*auto cls = */concat3(&g_pBlob[34], &g_pBlob[35], &g_pBlob[36], &g_pBlob[46]);
    CDataBlob_release(&g_pBlob[34]);
    CDataBlob_release(&g_pBlob[35]);
    CDataBlob_release(&g_pBlob[36]);

    blob2vector(&g_pBlob[12], &g_pBlob[37]);
    blob2vector(&g_pBlob[13], &g_pBlob[38]);
    blob2vector(&g_pBlob[14], &g_pBlob[39]);
    CDataBlob_release(&g_pBlob[12]);
    CDataBlob_release(&g_pBlob[13]);
    CDataBlob_release(&g_pBlob[14]);

    /*auto reg = */concat3(&g_pBlob[37], &g_pBlob[38], &g_pBlob[39], &g_pBlob[47]);
    CDataBlob_release(&g_pBlob[37]);
    CDataBlob_release(&g_pBlob[38]);
    CDataBlob_release(&g_pBlob[39]);

    blob2vector(&g_pBlob[18], &g_pBlob[40]);
    blob2vector(&g_pBlob[19], &g_pBlob[41]);
    blob2vector(&g_pBlob[20], &g_pBlob[42]);
    CDataBlob_release(&g_pBlob[18]);
    CDataBlob_release(&g_pBlob[19]);
    CDataBlob_release(&g_pBlob[20]);

    /*auto kps = */concat3(&g_pBlob[40], &g_pBlob[41], &g_pBlob[42], &g_pBlob[48]);
    CDataBlob_release(&g_pBlob[40]);
    CDataBlob_release(&g_pBlob[41]);
    CDataBlob_release(&g_pBlob[42]);

    blob2vector(&g_pBlob[21], &g_pBlob[43]);
    blob2vector(&g_pBlob[22], &g_pBlob[44]);
    blob2vector(&g_pBlob[23], &g_pBlob[45]);
    CDataBlob_release(&g_pBlob[21]);
    CDataBlob_release(&g_pBlob[22]);
    CDataBlob_release(&g_pBlob[23]);

    /*auto obj = */concat3(&g_pBlob[43], &g_pBlob[44], &g_pBlob[45], &g_pBlob[49]);
    CDataBlob_release(&g_pBlob[43]);
    CDataBlob_release(&g_pBlob[44]);
    CDataBlob_release(&g_pBlob[45]);

    sigmoid(&g_pBlob[46]); // cls
    sigmoid(&g_pBlob[49]); // obj
    TIME_END("decode")

    TIME_START;
    detection_output(&g_pBlob[46], &g_pBlob[47], &g_pBlob[48], &g_pBlob[49], 0.45f, 0.5f, 1000, 512,
                                                           face_blob, num_faces);
    TIME_END("detection output")
    CDataBlob_release(&g_pBlob[46]);
    CDataBlob_release(&g_pBlob[47]);
    CDataBlob_release(&g_pBlob[48]);
}

int* facedetect_cnn(unsigned char * result_buffer, //buffer memory for storing face detection results, !!its size must be 0x9000 Bytes!!
    unsigned char * rgb_image_data, int width, int height, int step, int is_rgb, float thresh) //input image, it must be BGR (three-channel) image!
{

    if (!result_buffer)
    {
        fprintf(stderr, "%s: null buffer memory.\n", __FUNCTION__);
        return NULL;
    }
    //clear memory
    result_buffer[0] = 0;
    result_buffer[1] = 0;
    result_buffer[2] = 0;
    result_buffer[3] = 0;

    int num_faces = 0;
    objectdetect_cnn(rgb_image_data, width, height, step, is_rgb, thresh, &g_pBlob[49], &num_faces);
    FaceRect* faces = (FaceRect*)(g_pBlob[49].data);
    num_faces = MIN(num_faces, 1024); //1024 = 0x9000 / (16 * 2 + 4)

    int * pCount = (int *)result_buffer;
    pCount[0] = num_faces;

    for (int i = 0; i < num_faces; i++)
    {
        //copy data
        short * p = ((short*)(result_buffer + 4)) + 16 * i;
        p[0] = (short)(faces[i].score * 100);
        p[1] = (short)faces[i].x;
        p[2] = (short)faces[i].y;
        p[3] = (short)faces[i].w;
        p[4] = (short)faces[i].h;
        //copy landmarks
        for (int lmidx = 0; lmidx < 10; lmidx++)
        {
            p[5 + lmidx] = (short)faces[i].lm[lmidx];
        }
    }
    return pCount;
}
