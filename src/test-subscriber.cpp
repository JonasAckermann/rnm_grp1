#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <cv_bridge/cv_bridge.h>

#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>

#include <pcl/point_cloud.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/PolygonMesh.h>
#include <pcl/filters/crop_box.h>
#include <pcl/point_types.h>
#include <pcl/conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/visualization/cloud_viewer.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>

#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace dlib;
using namespace std;

#undef DISPLAY_LANDMARKS
#undef DISPLAY_HEAD_MODEL
#undef DISPLAY_FACES
#define DISPLAY_CLOUD
#undef COLORED_CLOUD

#ifdef COLORED_CLOUD
typedef pcl::PointXYZRGBA CloudPoint;
#else
typedef pcl::PointXYZ CloudPoint;
#endif

typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo, sensor_msgs::CameraInfo> ExactSyncPolicy;

cv::Rect getRegionOfInterest(cv::Mat frame);
void visualizeRegionOfInterest(cv::Mat frame, cv::Rect roi_b);
void cloudViewer(pcl::PointCloud<CloudPoint>::Ptr cloud, const cv::Rect roi);
void createCloud(const cv::Mat &depth, const cv::Mat &color, pcl::PointCloud<CloudPoint>::Ptr &cloud, const cv::Rect roi);
void pclCallback(const sensor_msgs::Image::ConstPtr imageColor, const sensor_msgs::Image::ConstPtr imageDepth,
                 const sensor_msgs::CameraInfo::ConstPtr cameraInfoColor, const sensor_msgs::CameraInfo::ConstPtr cameraInfoDepth, const cv::Rect roi);

string headModelFileName = "/home/rnm_grp1/catkin_ws/src/rnmgrp1/data/head_model.stl";
string face_cascade_name = "/home/rnm_grp1/catkin_ws/src/rnmgrp1/data/cascades.xml";
string landmarksFileName = "/home/rnm_grp1/catkin_ws/src/rnmgrp1/data/shape_predictor_68_face_landmarks.dat";
string topicColor = "/kinect2/hd/image_color";
string topicDepth = "/kinect2/hd/image_depth_rect";
cv::CascadeClassifier face_cascade;
pcl::PointCloud<CloudPoint>::Ptr headCloud(new pcl::PointCloud<CloudPoint>);
string window_name = "Capture - Face detection";
cv::Rect lastRoi;

frontal_face_detector detector = get_frontal_face_detector();

#if defined(DISPLAY_LANDMARKS) || defined(DISPLAY_FACES)
image_window win;
#endif

shape_predictor pose_model;

pcl::PointCloud<CloudPoint>::Ptr cloud;
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


void imageCallback(const sensor_msgs::Image::ConstPtr imageColor, const sensor_msgs::Image::ConstPtr imageDepth,
                   const sensor_msgs::CameraInfo::ConstPtr cameraInfoColor, const sensor_msgs::CameraInfo::ConstPtr cameraInfoDepth)
{
    try
    {
        cv_bridge::CvImageConstPtr imagePtr = cv_bridge::toCvShare(imageColor, "bgr8");
        cv::Mat image = imagePtr->image;
        /*
        cv::Rect roi = getRegionOfInterest(image);
        lastRoi = roi;
        visualizeRegionOfInterest(image, roi);
        */
        // Turn OpenCV's Mat into something dlib can deal with.  Note that this just
        // wraps the Mat object, it doesn't copy anything.  So cimg is only valid as
        // long as temp is valid.  Also don't do anything to temp that would cause it
        // to reallocate the memory which stores the image as that will make cimg
        // contain dangling pointers.  This basically means you shouldn't modify temp
        // while using cimg.
        cv_image<bgr_pixel> cimg(image);

        // Detect faces
        std::vector<dlib::rectangle> faces = detector(cimg);
        cv::Rect roi;
        if (faces.size() >= 1) {
            dlib::rectangle r = faces.front();
            roi = cv::Rect(cv::Point2i(r.left(), r.top()), cv::Point2i(r.right() + 1, r.bottom() + 1));
        } else {
            roi.width = image.cols / 10;
            roi.height = image.rows / 10;
            roi.x = (image.cols - roi.width) / 2;
            roi.y = (image.rows - roi.height) / 2;
        }

#ifdef DISPLAY_LANDMARKS
        // Find the pose of each face.
        std::vector<full_object_detection> shapes;
        for (unsigned long i = 0; i < faces.size(); ++i)
            shapes.push_back(pose_model(cimg, faces[i]));
#endif
        // Display it all on the screen
#if defined(DISPLAY_FACES) || defined(DISPLAY_LANDMARKS)
        win.clear_overlay();
        win.set_image(cimg);
  #ifdef DISPLAY_LANDMARKS
        win.add_overlay(render_face_detections(shapes));
  #endif
  #ifdef DISPLAY_FACES
        win.add_overlay(faces);
  #endif
#endif
        pclCallback(imageColor, imageDepth, cameraInfoColor, cameraInfoDepth, roi);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", imageColor->encoding.c_str());
    }
}

void pclCallback(const sensor_msgs::Image::ConstPtr imageColor, const sensor_msgs::Image::ConstPtr imageDepth,
                 const sensor_msgs::CameraInfo::ConstPtr cameraInfoColor, const sensor_msgs::CameraInfo::ConstPtr cameraInfoDepth, const cv::Rect roi)
{
    static bool cloudInitialized = false;
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
        cloud = pcl::PointCloud<CloudPoint>::Ptr(new pcl::PointCloud<CloudPoint>());
        cloud->is_dense = false;
        createLookup(color.cols, color.rows);
        cloudInitialized = true;
    }
    cloud->height = roi.height;
    cloud->width = roi.width;
    cloud->points.resize(cloud->height * cloud->width);
    createCloud(depth, color, cloud, roi);
#ifdef DISPLAY_CLOUD
    cloudViewer(cloud, roi);
#endif
}

void cloudViewer(pcl::PointCloud<CloudPoint>::Ptr cloud, const cv::Rect roi)
{
    static bool viewerInitialized = false;
    static pcl::visualization::PCLVisualizer::Ptr cloudVisualizer(new pcl::visualization::PCLVisualizer("Cloud Viewer"));
    static std::string cloudName = "rendered";

    if (!viewerInitialized) {
        cloudVisualizer->addPointCloud(cloud, cloudName);
        cloudVisualizer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, cloudName);
        cloudVisualizer->initCameraParameters();
        cloudVisualizer->setBackgroundColor(0, 0, 0);
        cloudVisualizer->setPosition(0, 0);
        cloudVisualizer->setSize(roi.width, roi.height);
        cloudVisualizer->setShowFPS(true);
        cloudVisualizer->setCameraPosition(0, 0, 0, 0, -1, 0);
        viewerInitialized = true;
    } else {
        cloudVisualizer->setSize(roi.width, roi.height);
        cloudVisualizer->updatePointCloud(cloud, cloudName);
    }
    cloudVisualizer->spinOnce(10);
}

void createCloud(const cv::Mat &depth, const cv::Mat &color, pcl::PointCloud<CloudPoint>::Ptr &cloud, const cv::Rect roi)
{
    const float badPoint = std::numeric_limits<float>::quiet_NaN();

    #pragma omp parallel for
    for(int r = roi.y; r < roi.y + roi.height; ++r)
    {
        CloudPoint *itP = &cloud->points[(r - roi.y) * roi.width];
        const uint16_t *itD = depth.ptr<uint16_t>(r) + roi.x;
        const cv::Vec3b *itC = color.ptr<cv::Vec3b>(r)  + roi.x;
        const float y = lookupY.at<float>(0, r);
        const float *itX = lookupX.ptr<float>() + roi.x;

        for(size_t c = roi.x; c < (size_t)(roi.x + roi.width); ++c, ++itP, ++itD, ++itC, ++itX)
        {
            register const float depthValue = *itD / 1000.0f;
            // Check for invalid measurements
            if(*itD == 0)
            {
                // not valid
                itP->x = itP->y = itP->z = badPoint;
#ifdef COLORED_CLOUD
                itP->rgba = 0;
#endif
                continue;
            }
            itP->z = depthValue;
            itP->x = *itX * depthValue;
            itP->y = y * depthValue;
#ifdef COLORED_CLOUD
            itP->b = itC->val[0];
            itP->g = itC->val[1];
            itP->r = itC->val[2];
            itP->a = 255;
#endif
        }
    }
}



int main(int argc, char **argv)
{
    // load pointCloud file of head model
    pcl::PolygonMesh mesh;
    if (pcl::io::loadPolygonFileSTL(headModelFileName, mesh) == 0)
    {
      ROS_ERROR("Failed to load STL file\n");
    }
    // load landmarks file
    try {
        deserialize(landmarksFileName) >> pose_model;
    }
    catch(serialization_error& e)
    {
        ROS_ERROR("You need dlib's default face landmarking model file to run this example.");
        cout << "You can get it from the following URL: " << endl;
        cout << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
        cout << endl << e.what() << endl;
    }
    // create pointCloud of head model
    pcl::fromPCLPointCloud2(mesh.cloud, *headCloud);
#ifdef DISPLAY_HEAD_MODEL
    pcl::visualization::PCLVisualizer::Ptr headVisualizer(new pcl::visualization::PCLVisualizer("Cloud Viewer"));
    headVisualizer->addPointCloud(headCloud, "headCloud");
#endif
    // initialize ros node
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
    syncExact->registerCallback(imageCallback);

    ros::spin();
}

#undef DISPLAY_LANDMARKS
#undef DISPLAY_HEAD_MODEL
#undef DISPLAY_FACES
#undef DISPLAY_CLOUD
#undef COLORED_CLOUD
