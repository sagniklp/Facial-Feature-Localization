// FacialFeatureLocalization.cpp : Defines the entry point for the console application.
//
#pragma once
#include "stdafx.h"
#include "FaceDetection.h"
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/dir_nav.h>
#include <dlib/opencv.h>
#include <fstream>

using dlib::rectangle;
string face_classifier_dir = "C:\\openCV\\Build\\install\\etc\\haarcascades\\haarcascade_frontalface_alt.xml";
string prof_face_classifier_dir = "C:\\openCV\\Build\\install\\etc\\haarcascades\\haarcascade_profileface.xml";
string landmark_detector_dir = "C:\\dlib-19.7\\examples\\shape_predictor_68_face_landmarks.dat";
int scalef = 2;


rectangle opencv_rect_2_dlib_rect(Rect cvrect) {
	rectangle drect;
	drect.set_left (cvrect.x/scalef);
	drect.set_top (cvrect.y/scalef);
	drect.set_bottom ((cvrect.y + cvrect.height)/scalef);
	drect.set_right ((cvrect.x + cvrect.width)/scalef);
	return drect;
}

void show_landmark_points(dlib::full_object_detection shape, Mat frame){
	for (size_t i = 0; i < shape.num_parts(); i++)
	{
		circle(frame,Point(shape.part(i).x()*scalef,shape.part(i).y()*scalef),1,Scalar(0,255,255),1);
	}
}


int main() {
	cv::VideoCapture capture;
	Mat frame, small_frame;
	dlib::array2d<dlib::bgr_pixel> dlib_frame;
	dlib::array2d<dlib::bgr_pixel> small_dlib_frame;
	dlib::shape_predictor keypoint_detector;
	vector<vector<Rect>> out_bboxes;
	FaceDetection *fd;
	//std::ifstream inpf;
	//inpf.open(landmark_detector_dir);
	//cout << std::string((std::istreambuf_iterator<char>(inpf)),(std::istreambuf_iterator<char>()));
	dlib::deserialize(landmark_detector_dir) >> keypoint_detector;
	capture.open(0);
	if (!capture.isOpened()) {
		cout << "video not opened";
		return -1;
	}
	fd = new FaceDetection(face_classifier_dir, prof_face_classifier_dir);
	//Reading the video stream 
	while (true) {
		capture.read(frame);
		resize(frame, small_frame,Size(),0.5,0.5);
		dlib::assign_image(small_dlib_frame, dlib::cv_image<dlib::bgr_pixel>(small_frame));
		cv::namedWindow("Display Window 1", CV_WINDOW_AUTOSIZE);
		out_bboxes= fd->FaceDetectionCaller(frame);
		//vector<dlib::full_object_detection> shapes;
		for (size_t i = 0; i < out_bboxes.size(); i++)
		{
			for (size_t j = 0; j < out_bboxes.at(i).size(); j++)
			{
				cv::rectangle(frame, out_bboxes[i][j], Scalar(0, i*128, 255), 1);
				rectangle dlib_bbox;
				dlib_bbox=opencv_rect_2_dlib_rect(out_bboxes[i][j]);
				dlib::full_object_detection shape = keypoint_detector(small_dlib_frame, dlib_bbox);
				//shapes.push_back(shape);
				show_landmark_points(shape,frame);
				//cout << shape.part(0).x();
				//cout << "\n" << shape.part(0);
			}
		}
		
		if (cv::waitKey(1) == 27)
			break;
		imshow("Display Window 1", frame);
	}
	capture.release();
	cv::destroyWindow("Display Window 1");
	return 0;
}

