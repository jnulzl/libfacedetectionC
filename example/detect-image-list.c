/*
By downloading, copying, installing or using the software you agree to this license.
If you do not agree to this license, do not download, install,
copy or use the software.


                  License Agreement For libfacedetection
                     (3-clause BSD License)

Copyright (c) 2018-2020, Shiqi Yu, all rights reserved.
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

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#if __GNUC__
#include <sys/time.h>
long get_current_time()
{
    struct timeval stop, start;
    gettimeofday(&start, NULL);
    return start.tv_sec * 1000  + start.tv_usec / 1000; // ms
}
#endif


//define the buffer size. Do not change the size!
//0x9000 = 1024 * (16 * 2 + 4), detect 1024 face at most
#define DETECT_BUFFER_SIZE 0x9000


int main(int argc, char* argv[])
{
    if(argc != 3)
    {
        printf("Usage: %s <img_list> thresh\n", argv[0]);
        return -1;
    }

	int * pResults = NULL;
    //pBuffer is used in the detection functions.
    //If you call functions in multiple threads, please create one buffer for each thread!
    unsigned char * pBuffer = (unsigned char *)malloc(DETECT_BUFFER_SIZE);
    if(!pBuffer)
    {
        fprintf(stderr, "Can not alloc buffer.\n");
        return -1;
    }
    float thresh = atof(argv[2]);
    init_facedetect_resources();

    int sleep_time = 0;
     //read any text file from currect directory
    FILE* file = fopen(argv[1], "r");
    if(!file)
    {
        fprintf(stderr, "%s :  Unable to open : %s ", __FUNCTION__ , argv[1]);
        return -1;
    }
    char* img_path = NULL;
    size_t len = 0;
    ssize_t read;
    while ((read = getline(&img_path, &len, file)) != -1)
    {
        if (img_path[read - 1] == '\n')
        {
            img_path[read - 1] = '\0';
        }
        printf("Detecting image %s ", img_path);
        //Load an image from the disk.
        int img_width = 0;
        int img_height = 0;
        int img_channels = 0;
        unsigned char* img_color = stbi_load(img_path, &img_width, &img_height, &img_channels, 3);
        if(!img_color)
        {
            fprintf(stderr, "Can not load the image file %s.\n", img_path);
            break;
        }

        printf("width : %d, height : %d\n", img_width, img_height);

        ///////////////////////////////////////////
        // CNN face detection
        // Best detection rate
        //////////////////////////////////////////
        //!!! The input image must be a BGR one (three-channel) instead of RGB
        //!!! DO NOT RELEASE pResults !!!
#if __GNUC__
        long start_time = get_current_time();
#endif
        int is_rgb = 1; // The image format from stbi_load is RGB
        pResults = facedetect_cnn(pBuffer, img_color, img_width, img_height, img_width * 3, is_rgb, thresh);
#if __GNUC__
        printf("Time : %ld ms\n", get_current_time() - start_time);
#endif
        printf("%d faces detected.\n", (pResults ? *pResults : 0));
        //print the detection results
        for(int i = 0; i < (pResults ? *pResults : 0); i++)
        {
            short * p = ((short*)(pResults + 1)) + 16*i;
            int confidence = p[0];
            int x = p[1];
            int y = p[2];
            int w = p[3];
            int h = p[4];
            //print the result
            printf("face %d: confidence=%d, [%d, %d, %d, %d] (%d,%d) (%d,%d) (%d,%d) (%d,%d) (%d,%d)\n",
                    i, confidence, x, y, w, h,
                    p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[12], p[13],p[14]);

        }
        stbi_image_free(img_color);
    }
    // Free the allocated memory for the line
    free(img_path);

    // Close the file
    fclose(file);

    //release the buffer
    free(pBuffer);
    release_facedetect_resources();
	return 0;
}
