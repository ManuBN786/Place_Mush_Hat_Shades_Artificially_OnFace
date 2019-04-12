#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/video/tracking.hpp"
#include <opencv2/core/core.hpp>
 
#include <iostream>
#include <stdio.h>
#include <sys/timeb.h>
 
using namespace std;
using namespace cv;
 
double min_face_size=80;
double max_face_size=300;
Mat mask;
float w,h;


#if defined(_MSC_VER) || defined(WIN32)  || defined(_WIN32) || defined(__WIN32__) \
    || defined(WIN64)    || defined(_WIN64) || defined(__WIN64__) 
int CLOCK()
{
    return clock();
}
#endif

#if defined(unix)        || defined(__unix)      || defined(__unix__) \
    || defined(linux)       || defined(__linux)     || defined(__linux__) \
    || defined(sun)         || defined(__sun) \
    || defined(BSD)         || defined(__OpenBSD__) || defined(__NetBSD__) \
    || defined(__FreeBSD__) || defined __DragonFly__ \
    || defined(sgi)         || defined(__sgi) \
    || defined(__MACOSX__)  || defined(__APPLE__) \
    || defined(__CYGWIN__) 
int CLOCK()
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC,  &t);
    return (t.tv_sec * 1000)+(t.tv_nsec*1e-6);
}
#endif

double _avgdur=0;
int _fpsstart=0;
double _avgfps=0;
double _fps1sec=0;

double avgdur(double newdur)
{
    _avgdur=0.98*_avgdur+0.02*newdur;
    return _avgdur;
}

double avgfps()
{
    if(CLOCK()-_fpsstart>1000)      
    {
        _fpsstart=CLOCK();
        _avgfps=0.7*_avgfps+0.3*_fps1sec;
        _fps1sec=0;
    }

    _fps1sec++;
    return _avgfps;
}


 
int main( )
{
    VideoCapture cap(0);
    namedWindow( "window1", 1 ); 
    //namedWindow( "window2", 1 );
    mask = imread("Hat1.jpg"); 

   double fps = cap.get(CV_CAP_PROP_FPS);
   int frameno=0;

   //cout<<" Frame Rate: "<<fps<<endl;
    
  /* int num_frames = 120;
   time_t start, end;
   VideoCapture video(0);
   
   Mat frame;

   time(&start);
   for(int k =0; k<num_frames;k++){
        video>>frame;
    }
    time(&end);
   double seconds = difftime(end,start);
   fps = num_frames/seconds;
   cout<<" Frame Rate: "<<fps<<endl;*/
      
 
    while(1)
    {
        Mat image;
        
        cap >> image;
 
        clock_t start=CLOCK(); 
        double dur = CLOCK()-start;
        printf("avg time per frame %f ms. fps %f. frameno = %d\n",avgdur(dur),avgfps(),frameno++ );

        cout<<" Camera Frame Rate: "<<fps<<endl;
        //bool suc = cap.read(image);
        // Load Face cascade (.xml file)
        CascadeClassifier face_cascade( "haarcascade_frontalface_alt2.xml" );
 
        // Detect faces
        std::vector<Rect> faces;
        vector<Rect>ROIs;
 
        face_cascade.detectMultiScale( image, faces, 1.2, 2, 0|CV_HAAR_SCALE_IMAGE, Size(min_face_size, min_face_size),Size(max_face_size, max_face_size) ); 
 
        for(unsigned int i = 0; i < faces.size(); ++i)
        {        

        //Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 ); 
        //circle(image,center, 3, Scalar(255, 255, 255), -1, 8);
        Rect face = faces[i];      
        
        //rectangle(image, Point(face.x, face.y), Point(face.x+face.width, face.y+face.height),Scalar(255, 0, 0), 1, 4);

        // ROI points for placing the hat on top of head


        /*circle(image, Point(face.x-face.width*0.08,face.y), 3, Scalar(0, 255, 0), -1, 8);// Left Bottom
        circle(image, Point(face.x-face.width*0.08,face.y-face.height*0.29), 3, Scalar(255, 255, 0), -1, 8);// Left top
        circle(image, Point(face.x+face.width+face.width*0.08,face.y), 3, Scalar(0, 255, 255), -1, 8); // Right bottom
        circle(image, Point(face.x+face.width+face.width*0.08,face.y-face.height*0.29), 3, Scalar(255, 255, 0), -1, 8);// Right top*/

        // Create an ROI above the head. Rec is the co ordinates of the rectangle created above the head

        Rect Rec(abs(face.x-face.width*0.08),abs(face.y-face.height*0.21),abs(face.width*0.08+face.width+face.width*0.08),abs(face.height*0.29));
        //rectangle(image,Rec,Scalar(255,255,0),1,8);

        w = Rec.width;  // the width of ROI
        h = Rec.height; // Height of ROI
        Size r(w,h);
        Mat ROI_hat = image(Rect(face.x-face.width*0.08,face.y-face.height*0.29,face.width*0.08+face.width+face.width*0.08,face.height*0.29));
          

        // To Place the hat first find the center
        Point center(Rec.x +Rec.width*0.5,Rec.y +Rec.height*0.5 );
        //circle(image,center, 3, Scalar(255, 255,0), -1, 8);

        cout<<Rec.width<<endl;
        cout<<Rec.height<<endl;
        

        // Now replace the ROI with the hat image i.e mask
       // image = putMask(image,center,Size(Rec.width,Rec.height));

              Mat mask1,src1;
              Size face_size(Rec.width,Rec.height);
              resize(mask,mask1,face_size);
              // ROI selection
              
              Rect roi(Rec.x,Rec.y, face_size.width, face_size.height);
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

/*Mat putMask(Mat src,Point center,Size face_size)
{
    Mat mask1,src1;
    resize(mask,mask1,face_size);    
 
    // ROI selection
     Rect roi(center.x - face_size.width/2, center.y - face_size.width/2, face_size.width, face_size.width);
    src(roi).copyTo(src1);
 
    // to make the white region transparent
    Mat mask2,m,m1;
    cvtColor(mask1,mask2,CV_BGR2GRAY);
    threshold(mask2,mask2,230,255,CV_THRESH_BINARY_INV); 
 
    vector<Mat> maskChannels(3),result_mask(3);
    split(mask1, maskChannels);
    bitwise_and(maskChannels[0],mask2,result_mask[0]);
    bitwise_and(maskChannels[1],mask2,result_mask[1]);
    bitwise_and(maskChannels[2],mask2,result_mask[2]);
    merge(result_mask,m );         //    imshow("m",m);
 
    mask2 = 255 - mask2;
    vector<Mat> srcChannels(3);
    split(src1, srcChannels);
    bitwise_and(srcChannels[0],mask2,result_mask[0]);
    bitwise_and(srcChannels[1],mask2,result_mask[1]);
    bitwise_and(srcChannels[2],mask2,result_mask[2]);
    merge(result_mask,m1 );        //    imshow("m1",m1);
 
    addWeighted(m,1,m1,1,0,m1);    //    imshow("m2",m1);
     
    m1.copyTo(src(roi));
 
    return src;
}*/
 

