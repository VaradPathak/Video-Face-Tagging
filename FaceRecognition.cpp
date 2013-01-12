/*
 * FaceRecognition.cpp
 *
 *  Created on: Sep 29, 2012
 *      Author: vspathak
 */
/*
 * Copyright (c) 2011. Philipp Wagner <bytefish[at]gmx[dot]de>.
 * Released to public domain under terms of the BSD Simplified license.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the name of the organization nor the names of its contributors
 *     may be used to endorse or promote products derived from this software
 *     without specific prior written permission.
 *
 *   See <http://www.opensource.org/licenses/bsd-license>
 */

#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv/cv.h"

#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace std;

static void read_csv(const string& filename, vector<Mat>& images,
		vector<int>& labels, char separator = ';') {
	std::ifstream file(filename.c_str(), ifstream::in);
	if (!file) {
		string error_message =
				"No valid input file was given, please check the given filename.";
		CV_Error(CV_StsBadArg, error_message);
	}
	string line, path, classlabel;
	while (getline(file, line)) {
		stringstream liness(line);
		getline(liness, path, separator);
		getline(liness, classlabel);
		if (!path.empty() && !classlabel.empty()) {
			images.push_back(imread(path, 0));
			labels.push_back(atoi(classlabel.c_str()));
		}
	}
}

int main(int argc, const char *argv[]) {
	// Check for valid command line arguments, print usage
	// if no arguments were given.
	if (argc != 3) {
		cout << "usage: " << argv[0]
				<< " </path/to/haar_cascade> </path/to/csv.ext> </path/to/device id>"
				<< endl;
		cout
				<< "\t </path/to/haar_cascade> -- Path to the Haar Cascade for face detection."
				<< endl;
		cout
				<< "\t </path/to/csv.ext> -- Path to the CSV file with the face database."
				<< endl;
		cout << "\t <device id> -- The webcam device id to grab frames from."
				<< endl;
		exit(1);
	}
	// Get the path to your CSV:
	string fn_haar = string(argv[1]);
	string fn_csv = string(argv[2]);

	// These vectors hold the images and corresponding labels:
	vector<Mat> images;
	vector<int> labels;
	// Read in the data (fails if no valid input filename is given, but you'll get an error message):
	try {
		read_csv(fn_csv, images, labels);
	} catch (cv::Exception& e) {
		cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg
				<< endl;
		// nothing more we can do
		exit(1);
	}
	// Get the height from the first image. We'll need this
	// later in code to reshape the images to their original
	// size AND we need to reshape incoming faces to this size:

	int im_width = images[0].cols;
	int im_height = images[0].rows;
	// Create a FaceRecognizer and train it on the given images:
	Ptr<FaceRecognizer> model = createFisherFaceRecognizer();
	model->train(images, labels);
	// That's it for learning the Face Recognition model. You now
	// need to create the classifier for the task of Face Detection.
	// We are going to use the haar cascade you have specified in the
	// command line arguments:
	//
	CascadeClassifier haar_cascade;
	haar_cascade.load(fn_haar);
	// Get a handle to the Video device:
	//CvCapture *cap = cvCaptureFromAVI("new.avi");
	//VideoCapture cap("new.avi");
	//VideoCapture cap(0);
	// Check if we can use this device at all:
//	if (!cap.isOpened()) {
	//	cerr << "Capture file myVideo.avi cannot be opened." << endl;
	//	return -1;
	//}
	// Holds the current frame from the Video device:
	Mat frame0;
	Mat frame1;
	Mat frame2;
	//for (;;) {
	//	frame = cvRetrieveFrame(cap);
	//	cap >> frame;
	frame0 = images[0];
	frame1 = images[1];
	frame2 = images[2];
	// Clone the current frame:
	Mat original0 = frame0.clone();
	Mat original1 = frame1.clone();
	Mat original2 = frame2.clone();
	// Convert the current frame to grayscale:
	/*Mat original0;
	 Mat original1;
	 Mat original2;
	 cvtColor(original0, original0, CV_BGR2GRAY);
	 cvtColor(original1, original1, CV_BGR2GRAY);
	 cvtColor(original2, original2, CV_BGR2GRAY);*/
	// Find the faces in the frame:
	vector<Rect_<int> > faces0;
	vector<Rect_<int> > faces1;
	vector<Rect_<int> > faces2;
	haar_cascade.detectMultiScale(original0, faces0);
	haar_cascade.detectMultiScale(original1, faces1);
	haar_cascade.detectMultiScale(original2, faces2);
	// At this point you have the position of the faces in
	// faces. Now we'll get the faces, make a prediction and
	// annotate it in the video. Cool or what?
	for (unsigned int i = 0; i < faces0.size(); i++) {
		// Process face by face:
		Rect face_i = faces0[i];
		// Crop the face from the image. So simple with OpenCV C++:
		Mat face = original0(face_i);
		// Resizing the face is necessary for Eigenfaces and Fisherfaces. You can easily
		// verify this, by reading through the face recognition tutorial coming with OpenCV.
		// Resizing IS NOT NEEDED for Local Binary Patterns Histograms, so preparing the
		// input data really depends on the algorithm used.
		//
		// I strongly encourage you to play around with the algorithms. See which work best
		// in your scenario, LBPH should always be a contender for robust face recognition.
		//
		// Since I am showing the Fisherfaces algorithm here, I also show how to resize the
		// face you have just found:
		Mat face_resized;
		cv::resize(face, face_resized, Size(im_width, im_height), 1.0, 1.0,
				INTER_CUBIC);
		// Now perform the prediction, see how easy that is:
		int prediction = model->predict(face_resized);
		// And finally write all we've found out to the original image!
		// First of all draw a green rectangle around the detected face:
		rectangle(original0, face_i, CV_RGB(0, 255, 0), 1);
		// Create the text we will annotate the box with:
		string box_text = format("Prediction = %d", prediction);
		// Calculate the position for annotated text (make sure we don't
		// put illegal values in there):
		int pos_x = std::max(face_i.tl().x - 10, 0);
		int pos_y = std::max(face_i.tl().y - 10, 0);
		// And now put it into the image:
		putText(original0, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN,
				1.0, CV_RGB(0, 255, 0), 2.0);
		// Show the result:
		namedWindow("Display Image", 1024);
		imshow("Display Image", original0);
		waitKey(0);
	}
	for (unsigned int i = 0; i < faces1.size(); i++) {
		// Process face by face:
		Rect face_i = faces1[i];
		// Crop the face from the image. So simple with OpenCV C++:
		Mat face = original1(face_i);
		// Resizing the face is necessary for Eigenfaces and Fisherfaces. You can easily
		// verify this, by reading through the face recognition tutorial coming with OpenCV.
		// Resizing IS NOT NEEDED for Local Binary Patterns Histograms, so preparing the
		// input data really depends on the algorithm used.
		//
		// I strongly encourage you to play around with the algorithms. See which work best
		// in your scenario, LBPH should always be a contender for robust face recognition.
		//
		// Since I am showing the Fisherfaces algorithm here, I also show how to resize the
		// face you have just found:
		Mat face_resized;
		cv::resize(face, face_resized, Size(im_width, im_height), 1.0, 1.0,
				INTER_CUBIC);
		// Now perform the prediction, see how easy that is:
		int prediction = model->predict(face_resized);
		// And finally write all we've found out to the original image!
		// First of all draw a green rectangle around the detected face:
		rectangle(original1, face_i, CV_RGB(0, 255, 0), 1);
		// Create the text we will annotate the box with:
		string box_text = format("Prediction = %d", prediction);
		// Calculate the position for annotated text (make sure we don't
		// put illegal values in there):
		int pos_x = std::max(face_i.tl().x - 10, 0);
		int pos_y = std::max(face_i.tl().y - 10, 0);
		// And now put it into the image:
		putText(original1, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN,
				1.0, CV_RGB(0, 255, 0), 2.0);
		// Show the result:
		namedWindow("Display Image", 1024);
		imshow("Display Image", original1);
		waitKey(0);
	}
	for (unsigned int i = 0; i < faces2.size(); i++) {
		// Process face by face:
		Rect face_i = faces2[i];
		// Crop the face from the image. So simple with OpenCV C++:
		Mat face = original2(face_i);
		// Resizing the face is necessary for Eigenfaces and Fisherfaces. You can easily
		// verify this, by reading through the face recognition tutorial coming with OpenCV.
		// Resizing IS NOT NEEDED for Local Binary Patterns Histograms, so preparing the
		// input data really depends on the algorithm used.
		//
		// I strongly encourage you to play around with the algorithms. See which work best
		// in your scenario, LBPH should always be a contender for robust face recognition.
		//
		// Since I am showing the Fisherfaces algorithm here, I also show how to resize the
		// face you have just found:
		Mat face_resized;
		cv::resize(face, face_resized, Size(im_width, im_height), 1.0, 1.0,
				INTER_CUBIC);
		// Now perform the prediction, see how easy that is:
		int prediction = model->predict(face_resized);
		// And finally write all we've found out to the original image!
		// First of all draw a green rectangle around the detected face:
		rectangle(original2, face_i, CV_RGB(0, 255, 0), 1);
		// Create the text we will annotate the box with:
		string box_text = format("Prediction = %d", prediction);
		// Calculate the position for annotated text (make sure we don't
		// put illegal values in there):
		int pos_x = std::max(face_i.tl().x - 10, 0);
		int pos_y = std::max(face_i.tl().y - 10, 0);
		// And now put it into the image:
		putText(original2, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN,
				1.0, CV_RGB(0, 255, 0), 2.0);
		// Show the result:
		namedWindow("Display Image", 1024);
		imshow("Display Image", original2);
		waitKey(0);
	}

	/*// And display it:
	 char key = (char) waitKey(20);
	 // Exit this loop on escape:
	 if (key == 27)
	 break;*/
	/*}
	 return 0;*/
}

