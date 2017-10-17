
#pragma once
#include "stdafx.h"
#include <iostream>
#include <thread>
#include <future>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\objdetect.hpp>
#include <opencv2\videoio.hpp>
#include <opencv2\imgproc\imgproc.hpp> 

using cv::CascadeClassifier;
using cv::Mat;
using cv::Rect;
using cv::Scalar;
using cv::Point;
using cv::Size;
using std::promise;
using std::future;
using std::string;
using std::vector;
using std::thread;
using std::cout;

CascadeClassifier face_classf;
CascadeClassifier rprofile_face_classf;
CascadeClassifier lprofile_face_classf;

class FaceDetection
{
public:
	string face_classifier_dir;
	string prof_face_classifier_dir;
	Mat frame;
	static void facecf(Mat img, promise<vector<Rect>> face_rect, double scalefactor, int min_neighbors, int flags, cv::Size minsize, cv::Size maxsize) {
		vector<Rect> faces;
		face_classf.detectMultiScale(img, faces, scalefactor, min_neighbors, flags, minsize, maxsize);
		//cout << "\n fromf" << faces.size();
		face_rect.set_value(faces);
	}
	static void rpfacecf(Mat img, promise<vector<Rect>> face_rect, double scalefactor, int min_neighbors, int flags, cv::Size minsize, cv::Size maxsize) {
		vector<Rect> rfaces;
		rprofile_face_classf.detectMultiScale(img, rfaces, scalefactor, min_neighbors, flags, minsize, maxsize);
		//cout << "\n fromrpf" << rfaces.size();
		face_rect.set_value(rfaces);
	}
	static void lpfacecf(Mat img, promise<vector<Rect>> face_rect, double scalefactor, int min_neighbors, int flags, cv::Size minsize, cv::Size maxsize) {
		vector<Rect> lfaces;
		lprofile_face_classf.detectMultiScale(img, lfaces, scalefactor, min_neighbors, flags, minsize, maxsize);
		//cout << "\n fromlpf" << lfaces.size();
		face_rect.set_value(lfaces);
	}
	static Mat facedetect_boundingbox(Mat frame) {
		//vector<Rect> faces;
		//vector<Rect> right_profile_faces;
		//vector<Rect> left_profile_faces;
		promise<vector<Rect>> faces;
		promise<vector<Rect>> right_profile_faces;
		promise<vector<Rect>> left_profile_faces;
		future<vector<Rect>> ftr_faces = faces.get_future();
		future<vector<Rect>> ftr_right_profile_faces = right_profile_faces.get_future();
		future<vector<Rect>> ftr_left_profile_faces = left_profile_faces.get_future();
		Mat gray_frame, small_gray_frame;
		Mat flipped_gray_frame, small_flipped_gray_frame;
		cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);
		flip(gray_frame, flipped_gray_frame, 1);
		resize(gray_frame, small_gray_frame, Size(), 0.33, 0.33);
		resize(flipped_gray_frame, small_flipped_gray_frame, Size(), 0.33, 0.33);
		thread t1(&FaceDetection::facecf, small_gray_frame, move(faces), 1.1, 3, 0 | cv::CASCADE_SCALE_IMAGE, Size(33, 33), Size(80, 80));
		thread t2(&FaceDetection::rpfacecf, small_flipped_gray_frame, move(right_profile_faces), 1.1, 3, 0 | cv::CASCADE_SCALE_IMAGE, Size(33, 33), Size(80, 80));
		thread t3(&FaceDetection::lpfacecf, small_gray_frame, move(left_profile_faces), 1.1, 3, 0 | cv::CASCADE_SCALE_IMAGE, Size(33, 33), Size(80, 80));
		vector<Rect> f = ftr_faces.get();
		t1.detach();
		vector<Rect> lf = ftr_left_profile_faces.get();
		t2.detach();
		vector<Rect> rf = ftr_right_profile_faces.get();
		t3.detach();

		int scalef = 3;

		if (f.size()) {
			for (size_t i = 0; i < f.size(); i++)
			{
				cv::rectangle(frame, Point(scalef * f[i].x, scalef * f[i].y), Point(scalef * (f[i].x + f[i].width), scalef * (f[i].y + f[i].height)), Scalar(255, 0, 255), 3);
			}
		}
		else if (lf.size()) {
			for (size_t i = 0; i < lf.size(); i++)
			{
				cv::rectangle(frame, Point(scalef * lf[i].x, scalef * lf[i].y), Point(scalef * (lf[i].x + lf[i].width), scalef * (lf[i].y + lf[i].height)), Scalar(0, 255, 255), 3);
			}
		}
		else if (rf.size()) {
			for (size_t i = 0; i < rf.size(); i++)
			{
				int curr_x = scalef * rf[i].x;
				int curr_y = scalef * rf[i].y;
				int flippd_x = frame.cols - curr_x - (rf[i].width*scalef);
				int flippd_y = curr_y;
				cv::rectangle(frame, Point(flippd_x, flippd_y), Point(frame.cols - curr_x, curr_y + (scalef * rf[i].height)), Scalar(255, 255, 0), 3);
			}
		}
		return frame;
	}

	Mat FaceDetectionCaller (Mat frame) {
		//Reading the video stream 
		Mat f_detected_frame=facedetect_boundingbox(frame);
		return f_detected_frame;
	}

	FaceDetection(string fc, string pc) {
		face_classifier_dir = fc;
		prof_face_classifier_dir = pc;
		//Loading the cascades
		if (!face_classf.load(face_classifier_dir)) {
			cout << "--Error loading face cascade classifier--\n";
		}
		if (!lprofile_face_classf.load(prof_face_classifier_dir)) {
			cout << "--Error loading profile face cascade classifier--\n";
		}
		if (!rprofile_face_classf.load(prof_face_classifier_dir)) {
			cout << "--Error loading profile face cascade classifier--\n";
		}
	};
	//~FaceDetection();
};

