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
    mask = imread("2.jpg"); 
    
 
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

        Mat ROI = image(Rect(face.x, face.y, face.width, face.height));

        // ROI above the eyes to place a glass

        /*circle(image,Point(face.x+face.width*0.1,face.y+face.height*0.24),3,Scalar(0,255,0),-1,8); // Top left
        circle(image,Point(face.x+face.width*0.85,face.y+face.height*0.24),3,Scalar(0,255,0),-1,8); // Top Right
        circle(image,Point(face.x+face.width*0.1,face.y+face.height*0.46),3,Scalar(0,255,0),-1,8); // Bottom Left
        circle(image,Point(face.x+face.width*0.85,face.y+face.height*0.46),3,Scalar(0,255,0),-1,8); // Bottom Right*/

        // Mat ROI_shades and Rect Rec have same dimensions but different format
        Mat ROI_shades=image(Rect(face.x+face.width*0.1,face.y+face.height*0.24,face.x+face.width*0.85-(face.x+face.width*0.1),face.y+face.height*0.46-(face.y+face.height*0.24)));
        Rect Rec(face.x+face.width*0.1,face.y+face.height*0.24,face.x+face.width*0.85-(face.x+face.width*0.1),face.y+face.height*0.46-(face.y+face.height*0.24));
       // rectangle(image,Rec,Scalar(255,255,0),1,8);

        // Replace ROI_shades with the mask i.e shades image
                 Mat mask1,src1;
                 Size face_size(Rec.width,Rec.height);
                 resize(mask,mask1,face_size);
                 // ROI selection
                 Rect roi(Rec.x,Rec.y, Rec.width, Rec.height);
                //rectangle(image,roi,Scalar(255,255,255),1,8);
                image(roi).copyTo(src1);
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
                m1.copyTo(image(roi));

        }
         

        imshow( "window1",image );

        
        // Press 'c' to escape
        if(waitKey(1) == 'c') break;        
    }
 
    waitKey(0);                  
    return 0;
}
