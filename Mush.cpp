#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/video/tracking.hpp"
#include <opencv2/core/core.hpp>
 
#include <iostream>
#include <stdio.h>
 
using namespace std;
using namespace cv;
 
double min_face_size=80;
double max_face_size=300;
Mat mask;


int main( )
{
    VideoCapture cap(0);
    namedWindow( "window1", 1 ); 
    //namedWindow( "window2", 1 );
    mask = imread("Mus2.png"); 
    
 
    while(1)
    {
        Mat image;
        
        //cap >> image; 

        bool suc = cap.read(image);
        // Load Face cascade (.xml file)
        CascadeClassifier face_cascade( "haarcascade_frontalface_alt2.xml" );
        // Load mouth cascase xml file
        CascadeClassifier mouth_cascade("Mouth.xml");
 
        // Detect faces
        std::vector<Rect> faces;
        
 
        face_cascade.detectMultiScale( image, faces, 1.2, 2, 0|CV_HAAR_SCALE_IMAGE, Size(min_face_size, min_face_size),Size(max_face_size, max_face_size) ); 
 
        for(unsigned int i = 0; i < faces.size(); i++)
        {        

        //Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 ); 
        //circle(image,center, 3, Scalar(255, 255, 255), -1, 8);
        Rect face = faces[i];      
        
        //rectangle(image, Point(face.x, face.y), Point(face.x+face.width, face.y+face.height),Scalar(255, 0, 0), 1, 4);

        // Create an ROI in the lower half of the detected face to detect mouth efficiently // ROI_mouth & Rect mouth ROI are same but diff data structures i.e for convenience
        Mat ROI_mouth=image(Rect(face.x,face.y+face.height*0.6,face.width,face.height*0.4));
        Rect mouthROI(face.x,face.y+face.height*0.6,face.width,face.height*0.4);
        //rectangle(image,mouthROI,Scalar(255,255,0),1,8); 

        // Mouth ROI points
        /*circle(image, Point(face.x,face.y+face.height*0.6), 3, Scalar(0, 255, 0), -1, 8);// Left top mush
        circle(image, Point(face.x+face.width,face.y+face.height*0.6), 3, Scalar(0, 255, 0), -1, 8);// Right top mush
        circle(image, Point(face.x,face.y+face.height), 3, Scalar(0, 255, 0), -1, 8);// Left bottom mush
        circle(image, Point(face.x+face.width,face.y+face.height), 3, Scalar(0, 255, 0), -1, 8);// Right bottom mush*/

        // Detect mouths
        vector<Rect>mouth;
        mouth_cascade.detectMultiScale(ROI_mouth, mouth, 1.20, 5, 0|CASCADE_SCALE_IMAGE, Size(30, 30));
         
          
                for (unsigned int j = 0; j<mouth.size();j++)
                 {
                 Rect mt = mouth[j];
                 //rectangle(ROI_mouth,mt,Scalar(255,255,255),1,8);
                 //rectangle(ROI_mouth, Point(m.x, m.y), Point(m.x+m.width, m.y+m.height), Scalar(0, 255, 0), 1, 4);

                 // Create ROI to place mustache above mouth with following points
                 /*circle(ROI_mouth, Point(mt.x-mt.width*0.18,mt.y+mt.height*0.25), 3, Scalar(255, 0, 0), -1, 8);// Bottom left
                 circle(ROI_mouth, Point(mt.x+mt.width+mt.width*0.28,mt.y+mt.height*0.25), 3, Scalar(255, 0, 0), -1, 8);// Bottom right
                 circle(ROI_mouth, Point(mt.x-mt.width*0.18,mt.y-mt.height*0.31), 3, Scalar(255, 0, 0), -1, 8);// Top Left
                 circle(ROI_mouth, Point(mt.x+mt.width+mt.width*0.28,mt.y-mt.height*0.31), 3, Scalar(255, 0, 0), -1, 8);// Top right*/

                 // Rec is the ROI to be replaced with the moustache image
                 Rect Rec(mt.x-mt.width*0.18,mt.y-mt.height*0.31,mt.x+mt.width+mt.width*0.28-(mt.x-mt.width*0.18),mt.y+mt.height*0.25-(mt.y-mt.height*0.31));
                 //rectangle(ROI_mouth,Rec,Scalar(255,255,0),1,8);

                 // Replace ROI with mustache image i.e mask.png
                 Mat mask1,src1;
                 Size face_size(Rec.width,Rec.height);
                 resize(mask,mask1,face_size);
                 // ROI selection
                 Rect roi(Rec.x,Rec.y, Rec.width, Rec.height);
                //rectangle(image,roi,Scalar(255,255,255),1,8);
                ROI_mouth(roi).copyTo(src1);
                // to make the white region transparent
                Mat mask2,m,m1;
                cvtColor(mask1,mask2,CV_BGR2GRAY);
                threshold(mask2,mask2,230,255,CV_THRESH_BINARY_INV);

                vector<Mat> maskChannels(3),result_mask(3);
                split(mask1, maskChannels);
                bitwise_and(maskChannels[0],mask2,result_mask[0]);
                bitwise_and(maskChannels[1],mask2,result_mask[1]);
                bitwise_and(maskChannels[2],mask2,result_mask[2]);
                merge(result_mask,m );

                mask2 = 255 - mask2;
                vector<Mat> srcChannels(3);
                split(src1, srcChannels);
 
                bitwise_and(srcChannels[0],mask2,result_mask[0]);
                bitwise_and(srcChannels[1],mask2,result_mask[1]);
                bitwise_and(srcChannels[2],mask2,result_mask[2]);
                merge(result_mask,m1 ); 

                addWeighted(m,1,m1,1,0,m1);
                m1.copyTo(ROI_mouth(roi));
                }


        }
         

        imshow( "window1",image );

        
        // Press 'c' to escape
        if(waitKey(1) == 'c') break;        
    }
 
    waitKey(0);                  
    return 0;
}


