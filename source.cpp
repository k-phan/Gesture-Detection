#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

/**
Function returns Max of 3 Integers
*/
int myMax(int a, int b, int c);

/**
Function returns Min of 3 Integers
*/
int myMin(int a, int b, int c);

/**
Function detects whether a pixel is skin color based on RGB values
*/
void mySkinDetect(Mat& src, Mat& dst);

/** 
Function Creates Contours based on input
*/
void myHandDetect(Mat& src, Mat& dst);

int main(){
	/* Use Camera */
	VideoCapture cap(0);

	/* If Camera Fails, EXIT */
	if (!cap.isOpened()) {
		cout << "Failed to open video camera." << endl;
		return -1;
	}

	/* Windows */
	namedWindow("ControlVideo", WINDOW_AUTOSIZE);
	namedWindow("SkinDetect", WINDOW_AUTOSIZE);
	namedWindow("Contours", WINDOW_AUTOSIZE);

	/* Test Frame Reading from Camera */
	Mat testFrame;
	bool frameSuccess = cap.read(testFrame);

	if (!frameSuccess) {
		cout << "Failed to read test frame from video stream." << endl;
	}

	/* Camera Loop */
	while (1) {
		/* Read New Frame */
		Mat frame;
		bool fSuccess = cap.read(frame);

		/* If reading a frame ever fails, BREAK */
		if (!fSuccess) {
			cout << "Failed to read frame from video loop." << endl;
			break;
		}

		/* Create Output Frame */
		Mat frameDest = Mat::zeros(frame.rows, frame.cols, CV_8UC1);	// create empty frame
		Mat skinFrame = frameDest.clone();

		/* Image Processing */
		mySkinDetect(frame, skinFrame);
		Mat contourFrame = skinFrame.clone();
		myHandDetect(skinFrame, contourFrame);

		/* Output Frame */
		imshow("ControlVideo", frame);
		imshow("SkinDetect", skinFrame);
		imshow("Contours", contourFrame);

		/* Wait for ESC Key */
		if (waitKey(30) == 27) {
			cout << "ESC -- Exiting Now" << endl;
			break;
		}
	}

	cap.release();
	return 0;
}

/* Maximum of 3 Integers */
int myMax(int a, int b, int c) {
	int m = a;
	(void)((m < b) && (m = b));
	(void)((m < c) && (m = c));
	return m;
}

/* Minimum of 3 Integers */
int myMin(int a, int b, int c) {
	int m = a;
	(void)((m > b) && (m = b));
	(void)((m > c) && (m = c));
	return m;
}

/* Skin detection function */
void mySkinDetect(Mat& src, Mat& dst) {
/*	Surveys of skin color modeling and detection techniques:
	Vezhnevets, Vladimir, Vassili Sazonov, and Alla Andreeva. 
		"A survey on pixel-based skin color detection techniques." 
		Proc. Graphicon. Vol. 3. 2003.
	Kakumanu, Praveen, Sokratis Makrogiannis, and Nikolaos Bourbakis. 
		"A survey of skin-color modeling and detection methods." 
		Pattern recognition 40.3 (2007): 1106-1122.	*/
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			/* For each pixel, compute AVG intensity of 3 color channels */
			Vec3b intensity = src.at<Vec3b>(i, j);
			int B = intensity[0]; int G = intensity[1]; int R = intensity[2];
			/* From Research: IF the below is TRUE --> SKIN */
			if ((R > 95 && G > 40 && B > 20) &&
				(myMax(R, G, B) - myMin(R, G, B) > 15) &&
				(abs(R - G) > 15) && (R > G) && (R > B)) {
				/* Set SKIN pixels to WHITE */
				dst.at<uchar>(i, j) = 255;
			}
		}
	}
}

/* Draw Lines around hand */
void myHandDetect(Mat& src, Mat& dst) {
	// Mat threshold_output;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	// /* Detect Edges Using Threshold */
	// threshold(src, threshold_output, 100, 255, THRESH_BINARY);

	/* Find Contours from the Threshold Output */
	findContours(src, contours, hierarchy, CV_RETR_TREE, 
		CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	/* Find CONVEX HULL for each contour */
	vector<vector<Point> > hull(contours.size());
	for (int i = 0; i < contours.size(); i++) {
		convexHull(Mat(contours[i]), hull[i], false);
	}

	/* Draw Contours based on Hull Results */
	for (int i = 0; i < contours.size(); i ++) {
		Scalar color = Scalar(255, 0, 0);
		drawContours(dst, contours, i, color, 1, 8, vector<Vec4i>(), 0, Point());
		drawContours(dst, hull, i, color, 1, 8, vector<Vec4i>(), 0, Point());
	}
}