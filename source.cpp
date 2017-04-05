#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/background_segm.hpp"
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
Function Creates Contours based on input and draws largest
*/
int findFingers(Mat& src, Mat& dst);

/**
Function to calculate angle between 2 fingers,
Source: https://picoledelimao.github.io/blog/2015/11/15/fingertip-detection-on-opencv/
*/
float innerAngle(float px1, float py1, float px2, float py2, float cx1, float cy1);

int main(){
	/* Use Camera */
	VideoCapture cap(0);

	/* If Camera Fails, EXIT */
	if (!cap.isOpened()) {
		cout << "Failed to open video camera." << endl;
		return -1;
	}

	/* Windows */
	// namedWindow("ControlVideo", WINDOW_AUTOSIZE);
	// namedWindow("SkinDetect", WINDOW_AUTOSIZE);
	// namedWindow("BGSubtract", WINDOW_AUTOSIZE);
	// namedWindow("Color BGS", WINDOW_AUTOSIZE);
	// namedWindow("FingerTips", WINDOW_AUTOSIZE);
	namedWindow("Output", WINDOW_AUTOSIZE);

	/* Test Frame Reading from Camera */
	Mat testFrame;
	bool frameSuccess = cap.read(testFrame);

	/* Background Subtract Permanent Variables */
	Mat fgMaskMog;
	Ptr<BackgroundSubtractorKNN> pMOG;
	pMOG = createBackgroundSubtractorKNN();

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
			/* Background Subtract */
		Mat bgsFrame = frame.clone();
		pMOG->apply(bgsFrame, fgMaskMog, .0008);
		// pMOG->apply(bgsFrame, fgMaskMog);
		// Mat colorForeground = Mat::zeros(frame.size(), frame.type());
		// frame.copyTo(colorForeground, fgMaskMog);
			/* Skin Detect */
		mySkinDetect(frame, skinFrame);
		Mat newSkinFrame = frameDest.clone();
		skinFrame.copyTo(newSkinFrame, fgMaskMog);
			/* Blur Image */
		Mat blurFrame1 = frameDest.clone();
		Mat blurFrame2 = frameDest.clone();
		GaussianBlur(newSkinFrame, blurFrame1, Size(11, 55), 0, BORDER_DEFAULT);
		medianBlur(blurFrame1, blurFrame2, 13);
			/* Find Contours */
		Mat contourFrame = blurFrame2.clone();
		int numCircles = findFingers(blurFrame2, contourFrame);
			/* Final Frame */
		Mat finalFrame = contourFrame.clone();
		String outputString = "";
		if (numCircles == 8 || numCircles == 9) {
			outputString = "High Five!!";
		} else if (numCircles == 6 || numCircles == 7) {
			outputString = "High Four?!";
		} else if (numCircles == 3 || numCircles == 4) {
			outputString = "Peace!";
		} else if (numCircles == 1 || numCircles == 2) {
			outputString = "UP UP!";
		}
		putText(finalFrame, outputString, cvPoint(60, 60), FONT_HERSHEY_SIMPLEX, 
			2.0, cvScalar(255, 0, 0), 4, CV_AA);

		/* Output Frame */
		// imshow("ControlVideo", frame);
		// imshow("SkinDetect", skinFrame);
		// imshow("BGSubtract", fgMaskMog);
		// imshow("Color BGS", colorForeground);
		// imshow("FingerTips", contourFrame);
		imshow("Output", finalFrame);


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
int findFingers(Mat& src, Mat& dst) {
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	/* Find Contours from the Threshold Output */
	findContours(src, contours, hierarchy, CV_RETR_EXTERNAL, 
		CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	/* Find CONVEX HULL for each contour */
	vector<vector<Point> > hull(contours.size());
	// vector<vector<int> > hullIndices(contours.size());
	for (int i = 0; i < contours.size(); i++) {
		convexHull(Mat(contours[i]), hull[i], false);
	}

	/* Find Largest Contour & Its AREA */
	int maxIndex = -1;
	double maxArea = 0;
	for (int i = 0; i < contours.size(); i++) {
		double area = contourArea(contours[i], false);
		if (area > maxArea) {
			maxArea = area;
			maxIndex = i;
		}
	}

	Scalar color = Scalar(255, 0, 0);
	drawContours(dst, contours, maxIndex, color, 1, 8, vector<Vec4i>(), 0, Point());
	drawContours(dst, hull, maxIndex, color, 1, 8, vector<Vec4i>(), 0, Point());

	int returnValue = -1;

	/** Detect Defects 
		- Tutorial From: https://picoledelimao.github.io/blog/2015/11/15/fingertip-detection-on-opencv/
		*/
	if (!contours.empty()) {
		if(hull[maxIndex].size() > 2) {
			/* Calculate the Single Hull & its Indices */
			vector<int> hullIndices;
			convexHull(Mat(contours[maxIndex]), hullIndices, true);

			/* Calculate the Defects (Gaps) */
			vector<Vec4i> convexityDefectSet;
			convexityDefects(Mat(contours[maxIndex]), hullIndices, convexityDefectSet);

			/* Bound the Hand with a Box */
			Rect boundary = boundingRect(hull[maxIndex]);
			rectangle(dst, boundary, Scalar(255, 0, 0));
			Point midPoint = Point(boundary.x + boundary.width/2, boundary.y + boundary.height/2);
			vector<Point> validPoints;
			int boundArea = boundary.width * boundary.height;

			/* Find Finger Tips */
			for (int i = 0; i < convexityDefectSet.size(); i++) {
				Point p1 = contours[maxIndex][convexityDefectSet[i][0]];
				Point p2 = contours[maxIndex][convexityDefectSet[i][1]];
				Point p3 = contours[maxIndex][convexityDefectSet[i][2]];

				double angle = atan2(midPoint.y - p1.y, midPoint.x - p1.x) * 180 / CV_PI;
				double inAngle = innerAngle(p1.x, p1.y, p2.x, p2.y, p3.x, p3.y);
				double len = sqrt(pow(p1.x - p3.x, 2) + pow(p1.y - p3.y, 2));

				if (angle > -30 && angle < 180 && abs(inAngle) > 10 && boundArea > 42000
					&& abs(inAngle) < 120 && len > 0.1 * boundary.height) {
					if(p1.y < boundary.y + 0.4 * boundary.height) validPoints.push_back(p1);
					if(p2.y < boundary.y + 0.4 * boundary.height) validPoints.push_back(p2);
				}

				/* Draw Lines for Testing */
				// line(dst, p1, p3, Scalar(255, 0, 0), 2);
				// line(dst, p3, p2, Scalar(255, 0, 0), 2);
			}

			/* Draw Finger Tips */
			for (int i = 0; i < validPoints.size(); i++) {
				circle(dst, validPoints[i], 9, Scalar(255, 0, 0));
			}

			returnValue = validPoints.size();
		}
	}

	return returnValue;
}

/* Function to Calculate Inner Angle,
 	Source Mentioned in Prototype
*/
float innerAngle(float px1, float py1, float px2, float py2, float cx1, float cy1) {
	float dist1 = sqrt( (px1-cx1)*(px1-cx1) + (py1-cy1)*(py1-cy1) );
	float dist2 = sqrt( (px2-cx1)*(px2-cx1) + (py2-cy1)*(py2-cy1) );
	float Ax, Ay, Bx, By, Cx, Cy;

	Cx = cx1; Cy = cy1;
	if(dist1 < dist2) {
		Bx = px1;
		By = py1;
		Ax = px2;
		Ay = py2;
	} else {
		Bx = px2;
		By = py2;
		Ax = px1;
		Ay = py1;
	}
	
	float Q1 = Cx - Ax;
	float Q2 = Cy - Ay;
	float P1 = Bx - Ax;
	float P2 = By - Ay;

	float A = acos((P1*Q1 + P2*Q2) / (sqrt(P1*P1+P2*P2) * sqrt(Q1*Q1+Q2*Q2)));

	A = A*180/CV_PI;

	return A;
}