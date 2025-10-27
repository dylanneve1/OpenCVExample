#include "Utilities.h"
#include <string>
#include <vector>
#include <iostream>
#include <limits>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

#define ARRAY_SIZE(arr) (static_cast<int>(sizeof(arr) / sizeof((arr)[0])))

// Video files
const char* abandoned_removed_video_files[] = {
    "Video01.avi", "Video02.avi", "Video03.avi", "Video04.avi", "Video05.avi",
    "Video06.avi", "Video07.avi", "Video08.avi", "Video09.avi", "Video10.avi"
};

// Ground truth labels / indices
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

// Label names array (index 0 unused, indices 1-3 correspond to ABANDONED, REMOVED, OTHER_CHANGE)
const char* label_strings[] = {"", "Abandoned", "Removed", "Other Change"};

const int NUM_OBJECT_LOCATIONS = 14;

// The items in the table below are
//   (1) Video number, (2) Frame number, (3) Type of change, (4) Top row,
//   (5) Left column, (6) Bottom row, (7) Right column
int object_locations[NUM_OBJECT_LOCATIONS][7] = {
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

// Region tracker for events
struct TrackedRegion {
    Rect   bbox;             // Max bounding box over time
    double maxArea;          // Max area (confirmed pixels) seen so far
    double lastArea;         // Area at last frame
    bool   locked;           // True once box is "finalised"
    int    label;            // ABANDONED / REMOVED / OTHER_CHANGE
    int    framesSinceSeen;  // Frames since last detection
    int    framesStable;     // Consecutive frames near maxArea
    bool   removalLogged;    // True once we logged a removal event
    bool   firstEventLogged; // True once we logged the first event for this region
};

// Detected Event
struct DetectedEvent {
    int   video; // Video number
    int   frame; // Frame number
    Rect  bbox;  // Bounding box
    int   label; // ABANDONED / REMOVED / OTHER_CHANGE
};

// Ground Truth Event
struct GTEvent {
    int  video; // Video number
    int  frame; // Frame number
    Rect bbox;  // Bounding box
    int  label; // ABANDONED / REMOVED / OTHER_CHANGE
};

// Per frame detections
struct Detection {
    Rect   r;
    double area;   // Area of confirmed pixels in r
    int    label;  // Current frame classification
};

// IoU calculator
static double IoU(const Rect& a, const Rect& b) {
    Rect inter = a & b;
    if (inter.empty()) {
        return 0.0;
    }
    double interArea = static_cast<double>(inter.area());
    double unionArea = static_cast<double>(a.area() + b.area() - inter.area());
    return interArea / unionArea;
}

// Performance Evaluation
void EvaluatePerformance(const vector<DetectedEvent>& detEvents, const vector<double>& videoFps) {
    // Build ground truth list from the object_locations array
    vector<GTEvent> gtEvents;
    gtEvents.reserve(NUM_OBJECT_LOCATIONS);
    
    for (int i = 0; i < NUM_OBJECT_LOCATIONS; ++i) {
        int video = object_locations[i][IMAGE_NUMBER_INDEX];
        int frame = object_locations[i][FRAME_NUMBER_INDEX];
        int type  = object_locations[i][CHANGE_TYPE_INDEX];
        
        // Extract bounding box coordinates
        int left   = object_locations[i][LEFT_COLUMN_INDEX];
        int top    = object_locations[i][TOP_ROW_INDEX];
        int width  = object_locations[i][RIGHT_COLUMN_INDEX] - left;
        int height = object_locations[i][BOTTOM_ROW_INDEX] - top;
        
        GTEvent gt;
        gt.video = video;
        gt.frame = frame;
        gt.bbox  = Rect(left, top, width, height);
        gt.label = type;
        gtEvents.push_back(gt);
    }

    const int numVideos = ARRAY_SIZE(abandoned_removed_video_files);

    // Global stats across all videos
    int totalTP = 0, totalFP = 0, totalFN = 0;
    int gtCountPerClass[4] = {0};      // How many GT events per class
    int detCountPerClass[4] = {0};     // How many detections per class
    int confMatrix[4][4] = {};         // Confusion matrix [GT][Pred]
    
    double totalDelay = 0.0;
    double totalIoU = 0.0;
    int numMatches = 0;

    // Pre-count GT and detection labels globally
    for (const auto& gt : gtEvents) {
        if (gt.label >= 1 && gt.label <= 3) 
            gtCountPerClass[gt.label]++;
    }
    for (const auto& det : detEvents) {
        if (det.label >= 1 && det.label <= 3) 
            detCountPerClass[det.label]++;
    }

    cout << "\n===================== PER-VIDEO PERFORMANCE =====================\n";

    // Evaluate each video separately
    for (int v = 1; v <= numVideos; ++v) {
        // Grab indices for GT and detections in this video
        vector<int> gtIdx, detIdx;
        
        for (size_t i = 0; i < gtEvents.size(); ++i) {
            if (gtEvents[i].video == v) gtIdx.push_back(i);
        }
        for (size_t j = 0; j < detEvents.size(); ++j) {
            if (detEvents[j].video == v) detIdx.push_back(j);
        }

        // Skip if nothing to evaluate here
        if (gtIdx.empty() && detIdx.empty())
            continue;

        // Get FPS for this video
        double fps = (v-1 < videoFps.size() && videoFps[v-1] > 0) ? videoFps[v-1] : 25.0;

        int TP = 0, FP = 0, FN = 0;
        vector<int> matched(detIdx.size(), -1);  // Track which GT each detection matched to
        
        int localGtCount[4] = {0};
        int localDetCount[4] = {0};
        int localConf[4][4] = {};

        // Count labels for this video
        for (int idx : gtIdx) {
            int lbl = gtEvents[idx].label;
            if (lbl >= 1 && lbl <= 3) localGtCount[lbl]++;
        }
        for (int idx : detIdx) {
            int lbl = detEvents[idx].label;
            if (lbl >= 1 && lbl <= 3) localDetCount[lbl]++;
        }

        double delaySum = 0.0, iouSum = 0.0;
        int matches = 0;

        cout << "\n--- Video " << v << " (" << abandoned_removed_video_files[v-1] << ") ---\n";

        // Show GT events for debugging
        cout << "  GT events:\n";
        for (size_t i = 0; i < gtIdx.size(); ++i) {
            const GTEvent& gt = gtEvents[gtIdx[i]];
            cout << "    #" << i << " frame=" << gt.frame 
                 << " label=" << label_strings[gt.label] << "\n";
        }
        
        // Show detections for debugging
        cout << "  Detections:\n";
        for (size_t i = 0; i < detIdx.size(); ++i) {
            const DetectedEvent& det = detEvents[detIdx[i]];
            cout << "    #" << i << " frame=" << det.frame 
                 << " label=" << label_strings[det.label] << "\n";
        }

        // Match detections to GT - for each GT, find best detection
        for (size_t i = 0; i < gtIdx.size(); ++i) {
            const GTEvent& gt = gtEvents[gtIdx[i]];
            
            int bestDet = -1;
            int minFrameDiff = INT_MAX;
            int actualFrameDiff = 0;
            double bestIoU = 0.0;

            // Find detection with smallest frame difference whose center is in GT box
            for (size_t j = 0; j < detIdx.size(); ++j) {
                if (matched[j] != -1) continue;  // Already used
                
                const DetectedEvent& det = detEvents[detIdx[j]];
                
                // Check if detection center is inside GT box
                Point center(det.bbox.x + det.bbox.width/2, det.bbox.y + det.bbox.height/2);
                
                if (center.x < gt.bbox.x || center.x >= gt.bbox.x + gt.bbox.width ||
                    center.y < gt.bbox.y || center.y >= gt.bbox.y + gt.bbox.height) {
                    continue;
                }

                int frameDiff = det.frame - gt.frame;
                int absDiff = abs(frameDiff);
                
                if (absDiff < minFrameDiff) {
                    minFrameDiff = absDiff;
                    actualFrameDiff = frameDiff;
                    bestDet = j;
                    bestIoU = IoU(gt.bbox, det.bbox);
                }
            }

            if (bestDet >= 0) {
                // Found a match
                matched[bestDet] = i;
                TP++;
                
                const DetectedEvent& det = detEvents[detIdx[bestDet]];
                double delay = actualFrameDiff / fps;
                
                delaySum += delay;
                iouSum += bestIoU;
                matches++;

                // Update confusion matrix
                int gtL = gt.label;
                int detL = det.label;
                if (gtL >= 1 && gtL <= 3 && detL >= 1 && detL <= 3) {
                    localConf[gtL][detL]++;
                    confMatrix[gtL][detL]++;
                }

                cout << "  MATCH: GT(" << label_strings[gt.label] << ", f=" << gt.frame 
                     << ") <-> Det(" << label_strings[det.label] << ", f=" << det.frame
                     << ") delay=" << delay << "s, IoU=" << bestIoU << "\n";
            } else {
                // Missed this GT event
                FN++;
                cout << "  MISS: GT(" << label_strings[gt.label] << ", f=" << gt.frame << ")\n";
            }
        }

        // Count false positives (unmatched detections)
        for (size_t j = 0; j < detIdx.size(); ++j) {
            if (matched[j] == -1) {
                FP++;
                const DetectedEvent& det = detEvents[detIdx[j]];
                cout << "  FP: Det(" << label_strings[det.label] << ", f=" << det.frame << ")\n";
            }
        }

        // Update global counters
        totalTP += TP;
        totalFP += FP;
        totalFN += FN;
        totalDelay += delaySum;
        totalIoU += iouSum;
        numMatches += matches;

        // Calculate metrics for this video
        double prec = (TP + FP > 0) ? (double)TP / (TP + FP) : 0.0;
        double rec = (TP + FN > 0) ? (double)TP / (TP + FN) : 0.0;
        double f1 = (prec + rec > 0) ? 2 * prec * rec / (prec + rec) : 0.0;

        cout << "\nDetection Stats:\n";
        cout << "  TP=" << TP << " FP=" << FP << " FN=" << FN << "\n";
        cout << "  Precision=" << prec << " Recall=" << rec << " F1=" << f1 << "\n";

        if (matches > 0) {
            cout << "  Avg delay: " << delaySum/matches << "s\n";
            cout << "  Avg IoU: " << iouSum/matches << "\n";
        }

        // Per-class metrics for this video
        cout << "Classification per class:\n";
        for (int c = 1; c <= 3; ++c) {
            int TPc = localConf[c][c];
            int FNc = localGtCount[c] - TPc;
            int FPc = localDetCount[c] - TPc;
            
            double p = (TPc + FPc > 0) ? (double)TPc / (TPc + FPc) : 0.0;
            double r = (TPc + FNc > 0) ? (double)TPc / (TPc + FNc) : 0.0;
            
            cout << "  " << label_strings[c] << ": GT=" << localGtCount[c] 
                 << " Det=" << localDetCount[c] << " TP=" << TPc 
                 << " Prec=" << p << " Rec=" << r << "\n";
        }
    }

    // Print global results
    cout << "\n===================== OVERALL PERFORMANCE =====================\n";

    double gPrec = (totalTP + totalFP > 0) ? (double)totalTP / (totalTP + totalFP) : 0.0;
    double gRec = (totalTP + totalFN > 0) ? (double)totalTP / (totalTP + totalFN) : 0.0;
    double gF1 = (gPrec + gRec > 0) ? 2 * gPrec * gRec / (gPrec + gRec) : 0.0;

    cout << "\nOverall Detection:\n";
    cout << "  TP=" << totalTP << " FP=" << totalFP << " FN=" << totalFN << "\n";
    cout << "  Precision=" << gPrec << " Recall=" << gRec << " F1=" << gF1 << "\n";

    if (numMatches > 0) {
        cout << "  Avg delay: " << totalDelay/numMatches << "s\n";
        cout << "  Avg IoU: " << totalIoU/numMatches << "\n";
    }

    // Global per-class metrics
    cout << "\nOverall Classification:\n";
    for (int c = 1; c <= 3; ++c) {
        int TPc = confMatrix[c][c];
        int FNc = gtCountPerClass[c] - TPc;
        int FPc = detCountPerClass[c] - TPc;
        
        double p = (TPc + FPc > 0) ? (double)TPc / (TPc + FPc) : 0.0;
        double r = (TPc + FNc > 0) ? (double)TPc / (TPc + FNc) : 0.0;
        
        cout << "  " << label_strings[c] << ": GT=" << gtCountPerClass[c] 
             << " Det=" << detCountPerClass[c] << " TP=" << TPc 
             << " Prec=" << p << " Rec=" << r << "\n";
    }

    // Confusion matrix
    cout << "\nConfusion Matrix (rows=GT, cols=Pred):\n";
    cout << "           Abandoned  Removed  Other\n";
    for (int r = 1; r <= 3; ++r) {
        cout << label_strings[r] << ":  ";
        for (int c = 1; c <= 3; ++c) {
            cout << confMatrix[r][c] << "       ";
        }
        cout << "\n";
    }
    cout << "===============================================================\n\n";
}

// Main Application
void MyApplication()
{
    const char* file_location = "Media/Abandoned/";
    const int number_of_videos = ARRAY_SIZE(abandoned_removed_video_files);
    VideoCapture* video = new VideoCapture[number_of_videos];

    bool quit_all = false;

    // Collect all detected events and FPS for evaluation
    vector<DetectedEvent> allDetectedEvents;
    vector<double> videoFps(number_of_videos, 25.0);

    // Process each video
    for (int video_file_no = 1; video_file_no <= number_of_videos && !quit_all; video_file_no++) {
        string filename(file_location);
        filename.append(abandoned_removed_video_files[video_file_no - 1]);
        video[video_file_no - 1].open(filename);

        if (!video[video_file_no - 1].isOpened()) {
            cout << "Cannot open video file: " << filename << endl;
            continue;
        }

        // Read first frame
        Mat current_frame;
        video[video_file_no - 1] >> current_frame;
        if (current_frame.empty())
            continue;

        // Initialize grayscale background from first frame
        Mat initial_gray;
        cvtColor(current_frame, initial_gray, COLOR_BGR2GRAY);
        GaussianBlur(initial_gray, initial_gray, Size(5, 5), 0);

        Mat background_gray = initial_gray.clone();
        Mat background_bgr;   // Colour background for histogram comparison

        // Flag set once we have a clean background from GMM
        bool bg_from_gmm_initialized = false;

        // Create short-term GMM background subtractor
        Ptr<BackgroundSubtractorMOG2> gmm = createBackgroundSubtractorMOG2();
        gmm->setDetectShadows(true);
        gmm->setHistory(50);      // ~2 seconds at 25 fps
        gmm->setVarThreshold(16); // Sensitivity for motion detection

        // Algorithm parameters
        const double STATIC_DIFF_THRESHOLD = 30.0;     // Static background difference threshold
        const int GMM_BINARY_THRESHOLD = 200;          // Threshold to remove shadows (127 detects shadows)
        Mat morph_kernel3 = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
        Mat morph_kernel5 = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));

        // Get video FPS to convert time thresholds to frame counts
        double fps = video[video_file_no - 1].get(CAP_PROP_FPS);
        if (fps <= 0) fps = 25.0;
        videoFps[video_file_no - 1] = fps;

        int abandoned_frames_threshold = static_cast<int>(fps * 6.0); // ~6 seconds
        int bg_init_frame              = static_cast<int>(fps * 4.0); // ~4 seconds to learn background

        // Tracker parameters
        const int   MAX_MISSED_FRAMES    = static_cast<int>(fps * 2.0); // Drop unlocked tracks after 2s
        const int   LOCK_STABLE_FRAMES   = static_cast<int>(fps * 1.0); // Need ~1s near max size to lock
        const double MASS_LOSS_FACTOR    = 0.5;                         // 50% mass loss = removed

        // Per-pixel counter: how long a pixel has been in "abandoned" state
        Mat abandoned_counter(current_frame.rows, current_frame.cols, CV_16U, Scalar(0));

        int frame_no = 1;

        // Prime GMM with first frame
        Mat dummy;
        gmm->apply(current_frame, dummy);

        // Region-level gating thresholds
        const double MIN_ABANDONED_AREA = 20.0;                 // Minimum confirmed pixel area
        const double MAX_MOTION_RATIO_IN_BOX = 0.01;            // Max 1% motion allowed in ROI
        const double MIN_CONFIRMED_OVER_CANDIDATE = 0.8;        // At least 80% of candidate must be confirmed
        const double INTENSITY_DELTA_THRESHOLD = 5.0;           // Intensity difference for classification

        // Tracker state for this video
        vector<TrackedRegion> tracks;

        // Main processing loop
        while (!current_frame.empty()) {
            Mat current_frame_with_gt = current_frame.clone();

            // Draw ground truth boxes
            for (int current = 0; current < NUM_OBJECT_LOCATIONS; current++) {
                if ((object_locations[current][IMAGE_NUMBER_INDEX] == video_file_no) && (object_locations[current][FRAME_NUMBER_INDEX] <= frame_no)) {
                    // Colour coding: cyan=other, green=abandoned, red=removed
                    Scalar colour(
                        (object_locations[current][CHANGE_TYPE_INDEX] == OTHER_CHANGE) ? 0xFF : 0x00,
                        (object_locations[current][CHANGE_TYPE_INDEX] == ABANDONED)    ? 0xFF : 0x00,
                        (object_locations[current][CHANGE_TYPE_INDEX] == REMOVED)      ? 0xFF : 0x00
                    );

                    rectangle(
                        current_frame_with_gt,
                        Point(object_locations[current][LEFT_COLUMN_INDEX],
                              object_locations[current][TOP_ROW_INDEX]),
                        Point(object_locations[current][RIGHT_COLUMN_INDEX],
                              object_locations[current][BOTTOM_ROW_INDEX]),
                        colour, 2
                    );
                }
            }

            // Short-term motion detection
            Mat gmm_raw;
            gmm->apply(current_frame, gmm_raw);   // Detect motion over recent frames

            Mat gmm_binary;
            threshold(gmm_raw, gmm_binary, GMM_BINARY_THRESHOLD, 255, THRESH_BINARY);
            morphologyEx(gmm_binary, gmm_binary, MORPH_CLOSE, morph_kernel5);
            morphologyEx(gmm_binary, gmm_binary, MORPH_OPEN, morph_kernel5);

            // Initialize static background from GMM
            if (!bg_from_gmm_initialized && frame_no >= bg_init_frame) {
                Mat bg_bgr_tmp;
                gmm->getBackgroundImage(bg_bgr_tmp);
                if (!bg_bgr_tmp.empty()) {
                    background_bgr = bg_bgr_tmp.clone();
                    cvtColor(bg_bgr_tmp, background_gray, COLOR_BGR2GRAY);
                    GaussianBlur(background_gray, background_gray, Size(5, 5), 0);

                    // Reset counter to avoid early frame contamination
                    abandoned_counter.setTo(Scalar(0));

                    bg_from_gmm_initialized = true;
                }
            }

            // Get static background difference
            Mat current_gray;
            cvtColor(current_frame, current_gray, COLOR_BGR2GRAY);
            GaussianBlur(current_gray, current_gray, Size(5, 5), 0);

            Mat diff_gray;
            absdiff(current_gray, background_gray, diff_gray);

            Mat static_mask;
            threshold(diff_gray, static_mask, STATIC_DIFF_THRESHOLD, 255, THRESH_BINARY);
            morphologyEx(static_mask, static_mask, MORPH_CLOSE, morph_kernel3);
            morphologyEx(static_mask, static_mask, MORPH_OPEN, morph_kernel3);

            // Canditate abandoned removed regions
            // (Static change AND not currently moving)
            Mat not_moving_mask;
            bitwise_not(gmm_binary, not_moving_mask);

            Mat candidate_abandoned;
            bitwise_and(static_mask, not_moving_mask, candidate_abandoned);

            // Confirmed abandoned regions (temporal persistence)
            Mat confirmed_abandoned(candidate_abandoned.size(), CV_8U, Scalar(0));

            if (bg_from_gmm_initialized) {
                // Count how long each pixel has been in "abandoned" state
                for (int y = 0; y < candidate_abandoned.rows; y++) {
                    const uchar* cand_ptr = candidate_abandoned.ptr<uchar>(y);
                    ushort*       cnt_ptr = abandoned_counter.ptr<ushort>(y);
                    uchar*        out_ptr = confirmed_abandoned.ptr<uchar>(y);

                    for (int x = 0; x < candidate_abandoned.cols; x++) {
                        if (cand_ptr[x] > 0) {
                            if (cnt_ptr[x] < 65535)
                                cnt_ptr[x]++;

                            // Mark as confirmed if threshold exceeded
                            out_ptr[x] = (cnt_ptr[x] >= abandoned_frames_threshold) ? 255 : 0;
                        } else {
                            // Reset counter if no longer a candidate
                            cnt_ptr[x] = 0;
                            out_ptr[x] = 0;
                        }
                    }
                }

                // Clean up confirmed mask
                morphologyEx(confirmed_abandoned, confirmed_abandoned, MORPH_CLOSE, morph_kernel5);
                morphologyEx(confirmed_abandoned, confirmed_abandoned, MORPH_OPEN, morph_kernel5);
            } else {
                confirmed_abandoned.setTo(Scalar(0));
            }

            Mat abandoned_display = current_frame.clone();

            // Extract detections from confirmed regions
            vector<Detection> detections;

            if (bg_from_gmm_initialized) {
                vector<vector<Point>> contours;
                findContours(confirmed_abandoned, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

                Rect img_bounds(0, 0, current_frame.cols, current_frame.rows);

                for (size_t i = 0; i < contours.size(); i++) {
                    double contour_area = contourArea(contours[i]);
                    // Check it's bigger than MIN_ABANDONED_AREA
                    if (contour_area < MIN_ABANDONED_AREA)
                        continue;

                    Rect r = boundingRect(contours[i]);
                    if (r.width <= 0 || r.height <= 0)
                        continue;

                    // Expand ROI around box for stability checks
                    int margin_x = max(5, r.width / 10);
                    int margin_y = max(5, r.height / 10);

                    Rect roi = r;
                    roi.x      = max(0, roi.x - margin_x);
                    roi.y      = max(0, roi.y - margin_y);
                    roi.width  = min(img_bounds.width  - roi.x, roi.width  + 2 * margin_x);
                    roi.height = min(img_bounds.height - roi.y, roi.height + 2 * margin_y);

                    roi = roi & img_bounds;
                    if (roi.width <= 0 || roi.height <= 0)
                        continue;

                    // Check motion and mask completeness in ROI
                    Mat roi_motion    = gmm_binary(roi);
                    Mat roi_candidate = candidate_abandoned(roi);
                    Mat roi_confirmed = confirmed_abandoned(roi);

                    int motion_pixels = countNonZero(roi_motion);
                    int roi_area      = roi.width * roi.height;
                    double motion_ratio = (roi_area > 0)
                                          ? static_cast<double>(motion_pixels) / roi_area
                                          : 1.0;

                    int cand_pixels = countNonZero(roi_candidate);
                    int conf_pixels = countNonZero(roi_confirmed);
                    double conf_ratio = (cand_pixels > 0)
                                        ? static_cast<double>(conf_pixels) / cand_pixels
                                        : 0.0;

                    // Gating checks
                    if (motion_ratio > MAX_MOTION_RATIO_IN_BOX)
                        continue;
                    if (conf_ratio < MIN_CONFIRMED_OVER_CANDIDATE)
                        continue;

                    // Classification: Abandoned vs Removed vs Other Change
                    Scalar mean_curr = mean(current_gray(roi));
                    Scalar mean_bg   = mean(background_gray(roi));
                    double delta = mean_bg[0] - mean_curr[0]; // >0: darker than BG (abandoned)

                    int label;
                    if (delta > INTENSITY_DELTA_THRESHOLD)
                        label = ABANDONED;      // Current darker so object added
                    else if (delta < -INTENSITY_DELTA_THRESHOLD)
                        label = REMOVED;        // Current brighter so object removed
                    else
                        label = OTHER_CHANGE;   // Ambiguous

                    // Count confirmed pixels in tight bounding box
                    Mat r_confirmed = confirmed_abandoned(r);
                    int area_confirmed = countNonZero(r_confirmed);

                    if (area_confirmed < MIN_ABANDONED_AREA)
                        continue;

                    // Create detection
                    Detection det;
                    det.r     = r;
                    det.area  = static_cast<double>(area_confirmed);
                    det.label = label;
                    detections.push_back(det);
                }
            }

            // Update tracker with current detections
            for (auto& tr : tracks)
                tr.framesSinceSeen++;

            // Match detections to existing tracks
            for (const auto& d : detections) {
                int bestIdx = -1;
                double bestIoU = 0.0;

                // Find best overlapping track
                for (size_t t = 0; t < tracks.size(); t++) {
                    if (tracks[t].removalLogged) continue; // Skip finished tracks

                    Rect inter = tracks[t].bbox & d.r;
                    if (inter.empty()) continue;

                    double interArea = static_cast<double>(inter.area());
                    double unionArea = static_cast<double>(tracks[t].bbox.area() +
                                                           d.r.area() - inter.area());
                    if (unionArea <= 0.0) continue;

                    double iou = interArea / unionArea;
                    if (iou > bestIoU) {
                        bestIoU = iou;
                        bestIdx = static_cast<int>(t);
                    }
                }

                if (bestIdx >= 0 && bestIoU > 0.3) {
                    // Update existing track
                    TrackedRegion& tr = tracks[bestIdx];
                    tr.bbox = tr.bbox | d.r;          // Expand to max area over time
                    tr.lastArea = d.area;
                    if (d.area > tr.maxArea)
                        tr.maxArea = d.area;

                    // Check if area is stable near maximum
                    if (d.area >= 0.9 * tr.maxArea)
                        tr.framesStable++;
                    else
                        tr.framesStable = 0;

                    tr.framesSinceSeen = 0;

                    // Before locking, prefer abandoned/removed over other change
                    if (!tr.locked && d.label != OTHER_CHANGE)
                        tr.label = d.label;
                } else {
                    // Create new track
                    TrackedRegion tr;
                    tr.bbox            = d.r;
                    tr.maxArea         = d.area;
                    tr.lastArea        = d.area;
                    tr.locked          = false;
                    tr.label           = d.label;
                    tr.framesSinceSeen = 0;
                    tr.framesStable    = 0;
                    tr.removalLogged   = false;
                    tr.firstEventLogged= false;
                    tracks.push_back(tr);
                }
            }

            // Drop stale, never-locked tracks
            tracks.erase(
                remove_if(tracks.begin(), tracks.end(),
                          [MAX_MISSED_FRAMES](const TrackedRegion& tr)
                          {
                              return (!tr.locked && !tr.removalLogged &&
                                      tr.framesSinceSeen > MAX_MISSED_FRAMES);
                          }),
                tracks.end()
            );

            // Lock bounding boxes and detect removals
            for (auto& tr : tracks) {
                // Lock track once it's been stable near max size
                if (!tr.locked && !tr.removalLogged) {
                    if (tr.framesStable >= LOCK_STABLE_FRAMES) {
                        tr.locked = true;

                        // Log initial event (abandoned/removed/other change) once
                        if (!tr.firstEventLogged) {
                            DetectedEvent ev;
                            ev.video = video_file_no;
                            ev.frame = frame_no;
                            ev.bbox  = tr.bbox;
                            ev.label = tr.label;

                            allDetectedEvents.push_back(ev);
                            tr.firstEventLogged = true;
                        }
                    }
                }

                // Detect mass loss
                if (tr.locked && !tr.removalLogged) {
                    Rect img_bounds(0, 0, current_frame.cols, current_frame.rows);
                    Rect b = tr.bbox & img_bounds;
                    if (b.width > 0 && b.height > 0) {
                        Mat b_conf = confirmed_abandoned(b);
                        int area_now = countNonZero(b_conf);

                        // Check if object has been removed
                        // Aka if it has lost MASS_LOSS_FACTOR of mass
                        if (area_now < MASS_LOSS_FACTOR * tr.maxArea) {
                            tr.removalLogged = true;
                            tr.label = REMOVED;

                            // Log removal event
                            DetectedEvent ev;
                            ev.video = video_file_no;
                            ev.frame = frame_no;
                            ev.bbox  = b;
                            ev.label = REMOVED;
                            allDetectedEvents.push_back(ev);
                        }
                    }
                }
            }

            // Draw bounding boxes for locked events
            for (const auto& tr : tracks) {
                if (!tr.locked)
                    continue; // Skip if it's not locked

                Rect img_bounds(0, 0, current_frame.cols, current_frame.rows);
                Rect b = tr.bbox & img_bounds;
                
                // Calculate the color for the box
                Scalar colour;
                if (tr.label == ABANDONED) {
                    colour = Scalar(0, 0, 255); // Red
                } else if (tr.label == REMOVED) {
                    colour = Scalar(255, 0, 0); // Blue
                } else {
                    colour = Scalar(0, 255, 255); // Yellow
                }

                rectangle(abandoned_display, b, colour, 2);
                putText(abandoned_display, label_strings[tr.label], Point(b.x, b.y - 5),
                        FONT_HERSHEY_SIMPLEX, 0.4, colour, 1);
            }

            // Get current GMM background
            Mat current_gmm_background;
            gmm->getBackgroundImage(current_gmm_background);

            // Prepare masks for visualization
            Mat diff_gray_bgr, static_mask_bgr, gmm_raw_bgr, gmm_binary_bgr;
            Mat candidate_abandoned_bgr, confirmed_abandoned_bgr;

            cvtColor(diff_gray, diff_gray_bgr, COLOR_GRAY2BGR);
            cvtColor(static_mask, static_mask_bgr, COLOR_GRAY2BGR);
            cvtColor(gmm_raw, gmm_raw_bgr, COLOR_GRAY2BGR);
            cvtColor(gmm_binary, gmm_binary_bgr, COLOR_GRAY2BGR);
            cvtColor(candidate_abandoned, candidate_abandoned_bgr, COLOR_GRAY2BGR);
            cvtColor(confirmed_abandoned, confirmed_abandoned_bgr, COLOR_GRAY2BGR);

            // Display the windows
            Mat r1_1 = JoinImagesHorizontally(
                current_frame_with_gt,
                "Original Ground Truth",
                current_gmm_background,
                "Short-term Background (GMM)",
                4
            );
            Mat row1 = JoinImagesHorizontally(
                r1_1,
                "",
                diff_gray_bgr,
                "Grayscale Difference",
                4
            );

            Mat r2_1 = JoinImagesHorizontally(
                static_mask_bgr,
                "Static Change Mask",
                gmm_raw_bgr,
                "GMM Raw Output",
                4
            );
            Mat row2 = JoinImagesHorizontally(
                r2_1,
                "",
                gmm_binary_bgr,
                "Moving Mask (GMM Binary)",
                4
            );

            Mat r3_1 = JoinImagesHorizontally(
                candidate_abandoned_bgr,
                "Static & Not Moving",
                confirmed_abandoned_bgr,
                "Confirmed Static Change",
                4
            );
            Mat row3 = JoinImagesHorizontally(
                r3_1,
                "",
                abandoned_display,
                "Locked Events",
                4
            );

            Mat top_rows = JoinImagesVertically(row1, "", row2, "", 4);
            Mat final_output = JoinImagesVertically(top_rows, "", row3, "", 4);

            imshow(abandoned_removed_video_files[video_file_no - 1], final_output);

            // Controls
            char key = (char)waitKey(1);
            if (key == 'n')
                break;
            if (key == 'q' || key == 27) {
                quit_all = true;
                break;
            }

            // Read next frame
            video[video_file_no - 1] >> current_frame;
            frame_no++;
        }

        destroyAllWindows();
    }

    delete[] video;

    // Performance evaluation
    EvaluatePerformance(allDetectedEvents, videoFps);
}