﻿
#include "Utilities.h"


// Test videos and ground truth
char* abandoned_removed_video_files[] = {
	"Video01.avi",
	"Video02.avi",
	"Video03.avi",
	"Video04.avi",
	"Video05.avi",
	"Video06.avi",
	"Video07.avi",
	"Video08.avi",
	"Video09.avi",
	"Video10.avi" };
#define ABANDONED 1
#define REMOVED 2
#define OTHER_CHANGE 3
#define IMAGE_NUMBER_INDEX 0
#define FRAME_NUMBER_INDEX 1
#define CHANGE_TYPE_INDEX 2
#define TOP_ROW_INDEX 3
#define LEFT_COLUMN_INDEX 4
#define BOTTOM_ROW_INDEX 5
#define RIGHT_COLUMN_INDEX 6
// The items in the table below are
//   (1) Video number, (2) Frame number, (3) Type of change, (4) Top row,
//   (5) Left column, (6) Bottom row, (7) Right column
int object_locations[][7] = {
	{1, 115, REMOVED, 105,249,148,311}, // Laptop removed
	{2, 87, ABANDONED, 130,250,148,302}, // Laptop abandoned
	{3, 87, REMOVED, 164,186,223,234}, // Bag removed
	{4, 91, ABANDONED, 208,356,238,388}, // Bag abandoned
	{4, 255, REMOVED, 208,356,238,388}, // Bag removed
	{5, 137, ABANDONED, 118,68,126,76}, // Bag abandoned
	{5, 357, REMOVED, 118,68,126,76}, // Bag removed
	{6, 127, ABANDONED, 28,210,40,227}, // Bag abandoned
	{6, 379, REMOVED, 28,210,40,227}, // Bag removed
	{7, 555, ABANDONED, 108,107,123,127}, // Bag abandoned
	{8, 333, ABANDONED, 104,219,140,275}, // Car abandoned
	{9, 331, ABANDONED, 322,129,376,188}, // Bag abandoned
	{10, 73, OTHER_CHANGE, 109,126,269,225}, // Chair moved
	{10, 83, OTHER_CHANGE, 104,250,148,311} // Laptop opened
};


void MyApplication()
{
	char* file_location = "Media/Abandoned/";
	int number_of_videos = sizeof(abandoned_removed_video_files) / sizeof(abandoned_removed_video_files[0]);
	VideoCapture* video = new VideoCapture[number_of_videos];
	for (int video_file_no = 1; (video_file_no <= number_of_videos); video_file_no++)
	{
		string filename(file_location);
		filename.append(abandoned_removed_video_files[video_file_no-1]);
		video[video_file_no-1].open(filename);
		if (video[video_file_no-1].isOpened())
		{
			Mat current_frame;
			int frame_no = 1;
			video[video_file_no-1] >> current_frame;
			while (!current_frame.empty())
			{
				string frame_title("Frame no ");
				frame_title.append(std::to_string(frame_no));
				writeText(current_frame, frame_title, 15, 15);
				// Draw ground truth
				for (int current = 0; (current < sizeof(object_locations) / 7); current++)
				{
					if ((object_locations[current][IMAGE_NUMBER_INDEX] == video_file_no) && (object_locations[current][FRAME_NUMBER_INDEX] <= frame_no))
					{
						Scalar colour((object_locations[current][CHANGE_TYPE_INDEX] == OTHER_CHANGE) ? 0xFF : 0x00,
							(object_locations[current][CHANGE_TYPE_INDEX] == ABANDONED) ? 0xFF : 0x00,
							(object_locations[current][CHANGE_TYPE_INDEX] == REMOVED) ? 0xFF : 0x00 );
						rectangle(current_frame, Point(object_locations[current][LEFT_COLUMN_INDEX], object_locations[current][TOP_ROW_INDEX]),
							Point(object_locations[current][RIGHT_COLUMN_INDEX], object_locations[current][BOTTOM_ROW_INDEX]), colour, 1, 8, 0);
					}
				}
				imshow(abandoned_removed_video_files[video_file_no-1], current_frame);
				char choice = cv::waitKey(100);  // This introduces a delay between frame.  The length of the delay/number can be reduced.
				video[video_file_no-1] >> current_frame;
				frame_no++;
			}
		}
		else
		{
			cout << "Cannot open video file: " << filename << endl;
			//			return -1;
		}
	}

}
