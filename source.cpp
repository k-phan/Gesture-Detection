#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

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
		frameDest = frame.clone();

		/* Output Frame */
		imshow("ControlVideo", frameDest);

		/* Wait for ESC Key */
		if (waitKey(30) == 27) {
			cout << "ESC -- Exiting Now" << endl;
			break;
		}
	}

	cap.release();
	return 0;
}