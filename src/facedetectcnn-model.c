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
Filters g_pFilters[NUM_CONV_LAYER];
int param_initialized = 0;
void init_parameters()
{
    for(int i = 0; i < NUM_CONV_LAYER; i++)
    {
        Filters_create(&g_pFilters[i], &param_pConvInfo[i]);
    }
}

void deinit_parameters()
{
    for(int i = 0; i < NUM_CONV_LAYER; i++)
    {
        Filters_release(&g_pFilters[i]);
    }
    param_initialized = 0;
}

#define NUM_MIDDLE_TMP_BLOB 50

CDataBlob* g_pBlob[NUM_MIDDLE_TMP_BLOB];
void deinit_blob()
{
    for(int i = 0; i < NUM_MIDDLE_TMP_BLOB; i++)
    {
        CDataBlob_setNULL(g_pBlob[i]);
    }
}

void release_resources()
{
    deinit_parameters();
}
void objectdetect_cnn(const unsigned char * rgbImageData, int width, int height, int step, float thresh,
                     FaceRect** faceRects, int* num_faces)
{

    TIME_START;
    if (!param_initialized)
    {
        init_parameters();
        param_initialized = 1;
    }
    TIME_END("init");

    TIME_START;
    CDataBlob fx;
    g_pBlob[0] = &fx;
    /*auto fx = */setDataFrom3x3S2P1to1x1S1P0FromImage(rgbImageData, width, height, 3, step, 32, &fx);
    TIME_END("convert data");

    /***************CONV0*********************/
    TIME_START;
    CDataBlob fx1;
    g_pBlob[1] = &fx1;
    /*fx = */convolution(&fx, &g_pFilters[0], 1, &fx1);
    TIME_END("conv_head");

    TIME_START;
    CDataBlob fx2;
    g_pBlob[2] = &fx2;
    /*fx = */convolutionDP(&fx1, &g_pFilters[1], &g_pFilters[2], 1, &fx2);
    TIME_END("conv0");

    TIME_START;
    CDataBlob fx3;
    g_pBlob[3] = &fx3;
    /*fx = */maxpooling2x2S2(&fx2, &fx3);
    TIME_END("pool0");

    /***************CONV1*********************/
    TIME_START;
    CDataBlob fx4;
    g_pBlob[4] = &fx4;
    /*fx = */convolution4layerUnit(&fx3, &g_pFilters[3], &g_pFilters[4], &g_pFilters[5], &g_pFilters[6], 1, &fx4);
    TIME_END("conv1");

    /***************CONV2*********************/
    TIME_START;
    CDataBlob fx5;
    g_pBlob[5] = &fx5;
    /*fx = */convolution4layerUnit(&fx4, &g_pFilters[7], &g_pFilters[8], &g_pFilters[9], &g_pFilters[10], 1, &fx5);
    TIME_END("conv2");

    /***************CONV3*********************/
    TIME_START;
    CDataBlob fx6;
    g_pBlob[6] = &fx6;
    /*fx = */maxpooling2x2S2(&fx5, &fx6);
    TIME_END("pool3");

    TIME_START;
    CDataBlob fb1;
    g_pBlob[7] = &fb1;
    /*auto fb1 = */convolution4layerUnit(&fx6, &g_pFilters[11], &g_pFilters[12], &g_pFilters[13], &g_pFilters[14], 1, &fb1);
    TIME_END("conv3");

    /***************CONV4*********************/
    TIME_START;
    CDataBlob fxx;
    g_pBlob[8] = &fxx;
    /*fx = */maxpooling2x2S2(&fb1, &fxx);
    TIME_END("pool4");

    TIME_START;
    CDataBlob fb2;
    g_pBlob[9] = &fb2;
    /*auto fb2 = */convolution4layerUnit(&fxx, &g_pFilters[15], &g_pFilters[16], &g_pFilters[17], &g_pFilters[18], 1, &fb2);
    TIME_END("conv4");

    /***************CONV5*********************/
    TIME_START;
    CDataBlob fx_conv5;
    g_pBlob[10] = &fx_conv5;
    /* fx = */maxpooling2x2S2(&fb2, &fx_conv5);
    TIME_END("pool5");

    TIME_START;
    CDataBlob fb3;
    g_pBlob[11] = &fb3;
    /*auto fb3 = */convolution4layerUnit(&fx_conv5, &g_pFilters[19], &g_pFilters[20], &g_pFilters[21], &g_pFilters[22], 1, &fb3);
    TIME_END("conv5");

    CDataBlob pred_reg[3], pred_cls[3], pred_kps[3], pred_obj[3];
    g_pBlob[12] = &pred_reg[0];
    g_pBlob[13] = &pred_reg[1];
    g_pBlob[14] = &pred_reg[2];

    g_pBlob[15] = &pred_cls[0];
    g_pBlob[16] = &pred_cls[1];
    g_pBlob[17] = &pred_cls[2];

    g_pBlob[18] = &pred_kps[0];
    g_pBlob[19] = &pred_kps[1];
    g_pBlob[20] = &pred_kps[2];

    g_pBlob[21] = &pred_obj[0];
    g_pBlob[22] = &pred_obj[1];
    g_pBlob[23] = &pred_obj[2];
    /***************branch5*********************/
    TIME_START;
    CDataBlob fb3_3;
    g_pBlob[24] = &fb3_3;
    /*fb3 = */convolutionDP(&fb3, &g_pFilters[27], &g_pFilters[28], 1, &fb3_3);
    /*pred_cls[2] = */convolutionDP(&fb3_3, &g_pFilters[33], &g_pFilters[34], 0, &pred_cls[2]);
    /*pred_reg[2] = */convolutionDP(&fb3_3, &g_pFilters[39], &g_pFilters[40], 0, &pred_reg[2]);
    /*pred_kps[2] = */convolutionDP(&fb3_3, &g_pFilters[51], &g_pFilters[52], 0, &pred_kps[2]);
    /*pred_obj[2] = */convolutionDP(&fb3_3, &g_pFilters[45], &g_pFilters[46], 0, &pred_obj[2]);
    TIME_END("branch5");

    /*****************add5*********************/    
    TIME_START;
    CDataBlob fb_up;
    g_pBlob[25] = &fb_up;
    upsampleX2(&fb3_3, &fb_up);
    CDataBlob fb_add5;
    g_pBlob[26] = &fb_add5;
    /*fb2 = */elementAdd(&fb_up, &fb2, &fb_add5);
    TIME_END("add5");

    /*****************add6*********************/    
    TIME_START;
    CDataBlob fb_add6;
    g_pBlob[27] = &fb_add6;
    /*fb2 = */convolutionDP(&fb_add5, &g_pFilters[25], &g_pFilters[26], 1, &fb_add6);
    /*pred_cls[1] = */convolutionDP(&fb_add6, &g_pFilters[31], &g_pFilters[32], 0, &pred_cls[1]);
    /*pred_reg[1] = */convolutionDP(&fb_add6, &g_pFilters[37], &g_pFilters[38], 0, &pred_reg[1]);
    /*pred_kps[1] = */convolutionDP(&fb_add6, &g_pFilters[49], &g_pFilters[50], 0, &pred_kps[1]);
    /*pred_obj[1] = */convolutionDP(&fb_add6, &g_pFilters[43], &g_pFilters[44], 0, &pred_obj[1]);
    TIME_END("branch4");

    /*****************add4*********************/
    TIME_START;
    CDataBlob fb_up1;
    g_pBlob[28] = &fb_up1;
    upsampleX2(&fb_add6, &fb_up1);
    CDataBlob fb_add4;
    g_pBlob[29] = &fb_add4;
    /*fb1 = */elementAdd(&fb_up1, &fb1, &fb_add4);
    TIME_END("add4");

    /***************branch3*********************/
    TIME_START;
    CDataBlob fb_branch3;
    g_pBlob[30] = &fb_branch3;
    /*fb1 = */convolutionDP(&fb_add4, &g_pFilters[23], &g_pFilters[24], 1, &fb_branch3);
    /*pred_cls[0] = */convolutionDP(&fb_branch3, &g_pFilters[29], &g_pFilters[30], 0, &pred_cls[0]);
    /*pred_reg[0] = */convolutionDP(&fb_branch3, &g_pFilters[35], &g_pFilters[36], 0, &pred_reg[0]);
    /*pred_kps[0] = */convolutionDP(&fb_branch3, &g_pFilters[47], &g_pFilters[48], 0, &pred_kps[0]);
    /*pred_obj[0] = */convolutionDP(&fb_branch3, &g_pFilters[41], &g_pFilters[42], 0, &pred_obj[0]);
    TIME_END("branch3");

    /***************PRIORBOX*********************/
    TIME_START;
    CDataBlob prior3, prior4, prior5;
    g_pBlob[31] = &prior3;
    g_pBlob[32] = &prior4;
    g_pBlob[33] = &prior5;
    /*auto prior3 = */meshgrid(fb_branch3.cols, fb_branch3.rows, 8, 0.0f, &prior3);
    /*auto prior4 = */meshgrid(fb_add6.cols, fb_add6.rows, 16, 0.0f,  &prior4);
    /*auto prior5 = */meshgrid(fb3_3.cols, fb3_3.rows, 32, 0.0f, &prior5);
    TIME_END("prior");
    /***************PRIORBOX*********************/

    TIME_START;
    bbox_decode(&pred_reg[0], &prior3, 8);
    bbox_decode(&pred_reg[1], &prior4, 16);
    bbox_decode(&pred_reg[2], &prior5, 32);

    kps_decode(&pred_kps[0], &prior3, 8);
    kps_decode(&pred_kps[1], &prior4, 16);
    kps_decode(&pred_kps[2], &prior5, 32);

    CDataBlob pred_cls_0, pred_cls_1, pred_cls_2, cls;
    g_pBlob[34] = &pred_cls_0;
    g_pBlob[35] = &pred_cls_1;
    g_pBlob[36] = &pred_cls_2;
    blob2vector(&pred_cls[0], &pred_cls_0);
    blob2vector(&pred_cls[1], &pred_cls_1);
    blob2vector(&pred_cls[2], &pred_cls_2);
    /*auto cls = */concat3(&pred_cls_0, &pred_cls_1, &pred_cls_2, &cls);

    CDataBlob pred_reg_0, pred_reg_1, pred_reg_2, reg;
    g_pBlob[37] = &pred_reg_0;
    g_pBlob[38] = &pred_reg_1;
    g_pBlob[39] = &pred_reg_2;
    blob2vector(&pred_reg[0], &pred_reg_0);
    blob2vector(&pred_reg[1], &pred_reg_1);
    blob2vector(&pred_reg[2], &pred_reg_2);
    /*auto reg = */concat3(&pred_reg_0, &pred_reg_1, &pred_reg_2, &reg);

    CDataBlob pred_kps_0, pred_kps_1, pred_kps_2, kps;
    g_pBlob[40] = &pred_kps_0;
    g_pBlob[41] = &pred_kps_1;
    g_pBlob[42] = &pred_kps_2;
    blob2vector(&pred_kps[0], &pred_kps_0);
    blob2vector(&pred_kps[1], &pred_kps_1);
    blob2vector(&pred_kps[2], &pred_kps_2);
    /*auto kps = */concat3(&pred_kps_0, &pred_kps_1, &pred_kps_2, &kps);

    CDataBlob pred_obj_0, pred_obj_1, pred_obj_2, obj;
    g_pBlob[43] = &pred_obj_0;
    g_pBlob[44] = &pred_obj_1;
    g_pBlob[45] = &pred_obj_2;
    blob2vector(&pred_obj[0], &pred_obj_0);
    blob2vector(&pred_obj[1], &pred_obj_1);
    blob2vector(&pred_obj[2], &pred_obj_2);
    /*auto obj = */concat3(&pred_obj_0, &pred_obj_1, &pred_obj_2, &obj);

    sigmoid(&cls);
    sigmoid(&obj);
    TIME_END("decode")

    g_pBlob[46] = &cls;
    g_pBlob[47] = &reg;
    g_pBlob[48] = &kps;
    g_pBlob[49] = &obj;
    TIME_START;
    /*std::vector<FaceRect> facesInfo = */detection_output(&cls, &reg, &kps, &obj, 0.45f, 0.5f, 1000, 512,
                                                           faceRects, num_faces);
    TIME_END("detection output")
    deinit_blob();
}

int* facedetect_cnn(unsigned char * result_buffer, //buffer memory for storing face detection results, !!its size must be 0x9000 Bytes!!
    unsigned char * rgb_image_data, int width, int height, int step, float thresh) //input image, it must be BGR (three-channel) image!
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

    FaceRect* faces;
    int num_faces = 0;
    /*std::vector<FaceRect> faces = */objectdetect_cnn(rgb_image_data, width, height, step, thresh, &faces, &num_faces);

//    int num_faces =(int)faces.size();
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
    myFree(&faces);
//    deinit_parameters();
    return pCount;
}
