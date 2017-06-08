#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <cv_bridge/cv_bridge.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>

#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;

typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo, sensor_msgs::CameraInfo> ExactSyncPolicy;

cv::Rect getRegionOfInterest(cv::Mat frame);
void visualizeRegionOfInterest(cv::Mat frame, cv::Rect roi_b);
void cloudViewer(cv::Mat color, cv::Mat depth, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud);
void createCloud(const cv::Mat &depth, const cv::Mat &color, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &cloud);

string face_cascade_name = "/home/rnm_grp1/catkin_ws/src/rnmgroup1/data/cascades.xml";
string topicColor = "/kinect2/hd/image_color";
string topicDepth = "/kinect2/hd/image_depth_rect";
cv::CascadeClassifier face_cascade;
string window_name = "Capture - Face detection";
cv::Rect lastRoi;


pcl::visualization::PCLVisualizer::Ptr cloudVisualizer(new pcl::visualization::PCLVisualizer("Cloud Viewer"));
pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud;
bool cloudInitialized = false;
cv::Mat cameraMatrixColor;
cv::Mat lookupX, lookupY;
void createLookup(size_t width, size_t height)
{
    const float fx = 1.0f / cameraMatrixColor.at<double>(0, 0);
    const float fy = 1.0f / cameraMatrixColor.at<double>(1, 1);
    const float cx = cameraMatrixColor.at<double>(0, 2);
    const float cy = cameraMatrixColor.at<double>(1, 2);
    float *it;

    lookupY = cv::Mat(1, height, CV_32F);
    it = lookupY.ptr<float>();
    for(size_t r = 0; r < height; ++r, ++it)
    {
        *it = (r - cy) * fy;
    }

    lookupX = cv::Mat(1, width, CV_32F);
    it = lookupX.ptr<float>();
    for(size_t c = 0; c < width; ++c, ++it)
    {
        *it = (c - cx) * fx;
    }
}

void readImage(const sensor_msgs::Image::ConstPtr msgImage, cv::Mat &image)
{
    cv_bridge::CvImageConstPtr pCvImage;
    pCvImage = cv_bridge::toCvShare(msgImage, msgImage->encoding);
    pCvImage->image.copyTo(image);
}

void readCameraInfo(const sensor_msgs::CameraInfo::ConstPtr cameraInfo, cv::Mat &cameraMatrix)
{
    double *itC = cameraMatrix.ptr<double>(0, 0);
    for(size_t i = 0; i < 9; ++i, ++itC)
    {
        *itC = cameraInfo->K[i];
    }
}


void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
    try
    {
        cv_bridge::CvImageConstPtr imagePtr = cv_bridge::toCvShare(msg, "bgr8");
        cv::Mat image = imagePtr->image;
        cv::Rect roi = getRegionOfInterest(image);
        lastRoi = roi;
        visualizeRegionOfInterest(image, roi);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    }
}

void pclCallback(const sensor_msgs::Image::ConstPtr imageColor, const sensor_msgs::Image::ConstPtr imageDepth,
                 const sensor_msgs::CameraInfo::ConstPtr cameraInfoColor, const sensor_msgs::CameraInfo::ConstPtr cameraInfoDepth)
{
    cv::Mat color, depth;
    readImage(imageColor, color);
    readImage(imageDepth, depth);
    if (!cloudInitialized) {
        double cameraParams[9] = { 1.0533457154589462e+03, 0., 9.5255170711209871e+02, 0.,
                                   1.0529715956757291e+03, 5.4050396155113071e+02, 0., 0., 1. };
        cameraMatrixColor = cv::Mat(3, 3, CV_64FC1, cameraParams);
        // IR image input
        if(color.type() == CV_16U)
        {
            cv::Mat tmp;
            color.convertTo(tmp, CV_8U, 0.02);
            cv::cvtColor(tmp, color, CV_GRAY2BGR);
        }
        cloud = pcl::PointCloud<pcl::PointXYZRGBA>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBA>());
        cloud->height = color.rows;
        cloud->width = color.cols;
        cloud->is_dense = false;
        cloud->points.resize(cloud->height * cloud->width);
        createLookup(color.cols, color.rows);
    }
    cloudViewer(color, depth, cloud);
}

void cloudViewer(cv::Mat color, cv::Mat depth, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud)
{
    const std::string cloudName = "rendered";
    if (!cloudInitialized) {
        createCloud(depth, color, cloud);

        cloudVisualizer->addPointCloud(cloud, cloudName);
        cloudVisualizer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, cloudName);
        cloudVisualizer->initCameraParameters();
        cloudVisualizer->setBackgroundColor(0, 0, 0);
        cloudVisualizer->setPosition(false ? color.cols : 0, 0);
        cloudVisualizer->setSize(color.cols, color.rows);
        cloudVisualizer->setShowFPS(true);
        cloudVisualizer->setCameraPosition(0, 0, 0, 0, -1, 0);

        cloudVisualizer->spinOnce(10);
        cloudInitialized = true;
    } else {
        createCloud(depth, color, cloud);
        cloudVisualizer->updatePointCloud(cloud, cloudName);
        cloudVisualizer->spinOnce(10);
    }
}

void createCloud(const cv::Mat &depth, const cv::Mat &color, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &cloud)
{
    const float badPoint = std::numeric_limits<float>::quiet_NaN();

    #pragma omp parallel for
    for(int r = 0; r < depth.rows; ++r)
    {
        pcl::PointXYZRGBA *itP = &cloud->points[r * depth.cols];
        const uint16_t *itD = depth.ptr<uint16_t>(r);
        const cv::Vec3b *itC = color.ptr<cv::Vec3b>(r);
        const float y = lookupY.at<float>(0, r);
        const float *itX = lookupX.ptr<float>();

        for(size_t c = 0; c < (size_t)depth.cols; ++c, ++itP, ++itD, ++itC, ++itX)
        {
            register const float depthValue = *itD / 1000.0f;
            // Check for invalid measurements
            if(*itD == 0)
            {
                // not valid
                itP->x = itP->y = itP->z = badPoint;
                itP->rgba = 0;
                continue;
            }
            itP->z = depthValue;
            itP->x = *itX * depthValue;
            itP->y = y * depthValue;
            itP->b = itC->val[0];
            itP->g = itC->val[1];
            itP->r = itC->val[2];
            itP->a = 255;
        }
    }
}



int main(int argc, char **argv)
{
    if (!face_cascade.load(face_cascade_name))
    {
        ROS_ERROR("Could not load cascade");
    }
    ros::init(argc, argv, "image_listener");
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);

    // taken from kinect2_viewer to create pointcloud
    std::string topicCameraInfoColor = topicColor.substr(0, topicColor.rfind('/')) + "/camera_info";
    std::string topicCameraInfoDepth = topicDepth.substr(0, topicDepth.rfind('/')) + "/camera_info";

    image_transport::TransportHints hints(false ? "compressed" : "raw");
    // TODO - consider use of image_transport::CameraSubscriber do combine subscription to cameraImage and cameraInfo (create via image_transport::ImageTransport::subscribeCamera)
    // TODO - use Chain with TimeSynchronizer and TimeSequencer
    image_transport::SubscriberFilter * subImageColor = new image_transport::SubscriberFilter(it, topicColor, 1, hints);
    image_transport::SubscriberFilter * subImageDepth = new image_transport::SubscriberFilter(it, topicDepth, 1, hints);
    message_filters::Subscriber<sensor_msgs::CameraInfo> * subCameraInfoColor = new message_filters::Subscriber<sensor_msgs::CameraInfo>(nh, topicCameraInfoColor, 1);
    message_filters::Subscriber<sensor_msgs::CameraInfo> * subCameraInfoDepth = new message_filters::Subscriber<sensor_msgs::CameraInfo>(nh, topicCameraInfoDepth, 1);

    message_filters::Synchronizer<ExactSyncPolicy> * syncExact = new message_filters::Synchronizer<ExactSyncPolicy>(ExactSyncPolicy(1), *subImageColor, *subImageDepth, *subCameraInfoColor, *subCameraInfoDepth);
    syncExact->registerCallback(pclCallback);


    ros::spin();
    cv::destroyWindow("view");
}

// Function detectAndDisplay
cv::Rect getRegionOfInterest(cv::Mat frame)
{
    std::vector<Rect> faces;
    Mat frame_gray;

    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);

    // Detect faces
    face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

    // Set Region of Interest
    cv::Rect roi_b;
    cv::Rect roi_c;

    size_t ic = 0; // ic is index of current element
    int ac = 0; // ac is area of current element

    size_t ib = 0; // ib is index of biggest element
    int ab = 0; // ab is area of biggest element

    for (ic = 0; ic < faces.size(); ic++) // Iterate through all current elements (detected faces)
    {
        roi_c.x = faces[ic].x;
        roi_c.y = faces[ic].y;
        roi_c.width = (faces[ic].width);
        roi_c.height = (faces[ic].height);

        ac = roi_c.width * roi_c.height; // Get the area of current element (detected face)

        roi_b.x = faces[ib].x;
        roi_b.y = faces[ib].y;
        roi_b.width = (faces[ib].width);
        roi_b.height = (faces[ib].height);

        ab = roi_b.width * roi_b.height; // Get the area of biggest element, at beginning it is same as "current" element

        if (ac > ab)
        {
            ib = ic;
            roi_b.x = faces[ib].x;
            roi_b.y = faces[ib].y;
            roi_b.width = (faces[ib].width);
            roi_b.height = (faces[ib].height);
        }
    }
    if (faces.size() == 0) {
        roi_b.width = frame.cols / 10;
        roi_b.height = frame.rows / 10;
        roi_b.x = (frame.cols - roi_b.width) / 2;
        roi_b.y = (frame.rows - roi_b.height) / 2;
    }

    return roi_b;
}

void visualizeRegionOfInterest(cv::Mat frame, cv::Rect roi_b) {
    string text;
    stringstream sstm;

    // Show image
    sstm << "Crop area size: " << roi_b.width << "x" << roi_b.height;
    text = sstm.str();

    putText(frame, text, cvPoint(30, 30), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0, 0, 255), 1, CV_AA);

    Point pt1(roi_b.x, roi_b.y); // Display roi on frame
    Point pt2((roi_b.x + roi_b.height), (roi_b.y + roi_b.width));
    rectangle(frame, pt1, pt2, Scalar(0, 255, 0), 2, 8, 0);
    cv::imshow("view", frame);
}
