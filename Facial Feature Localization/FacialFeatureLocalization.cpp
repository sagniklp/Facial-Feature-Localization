// FacialFeatureLocalization.cpp : Defines the entry point for the console application.
//
#pragma once
#include "stdafx.h"
#include "FaceDetection.h"
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/dir_nav.h>

string face_classifier_dir = "C:\\openCV\\Build\\install\\etc\\haarcascades\\haarcascade_frontalface_alt.xml";
string prof_face_classifier_dir = "C:\\openCV\\Build\\install\\etc\\haarcascades\\haarcascade_profileface.xml";
int main() {
	cv::VideoCapture capture;
	Mat frame, out_frame;
	FaceDetection *fd;
	capture.open(0);
	if (!capture.isOpened()) {
		cout << "video not opened";
		return -1;
	}
	fd=new FaceDetection(face_classifier_dir, prof_face_classifier_dir);
	//Reading the video stream 
	while (true) {
		capture.read(frame);
		cv::namedWindow("Display Window 1", CV_WINDOW_AUTOSIZE);
		out_frame= fd->FaceDetectionCaller(frame);
		if (cv::waitKey(1) == 27)
			break;
		imshow("Display Window 1", out_frame);
	}
	capture.release();
	cv::destroyWindow("Display Window 1");
	return 0;
}

