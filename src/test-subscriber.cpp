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
#include <pcl/filters/filter.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/passthrough.h>
#include <pcl/point_types.h>
#include <pcl/conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/registration.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/common/transforms.h>
#include <pcl/common/centroid.h>

#include <Eigen/Geometry>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>

#include "std_msgs/MultiArrayLayout.h"
#include "std_msgs/MultiArrayDimension.h"
#include "std_msgs/Float64MultiArray.h"

#include <iostream>
#include <stdio.h>

#include<facedetector.h>
#include<headposeestimator.h>

using namespace cv;
using namespace dlib;
using namespace std;

#define DISPLAY_LANDMARKS
#undef DISPLAY_HEAD_MODEL
#define DISPLAY_FACES
#define DISPLAY_CLOUD
#define USE_ICP
#undef COLORED_CLOUD

#ifdef COLORED_CLOUD
typedef pcl::PointXYZRGBA CloudPoint;
#else
typedef pcl::PointXYZ CloudPoint;
#endif

typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo, sensor_msgs::CameraInfo> ExactSyncPolicy;

void cloudViewer(pcl::PointCloud<CloudPoint>::Ptr cloud, pcl::PointCloud<CloudPoint>::Ptr cloud2, const cv::Rect roi);
void createCloud(const cv::Mat &depth, const cv::Mat &color, pcl::PointCloud<CloudPoint>::Ptr &cloud, const cv::Rect roi);
void findPose(const cv::Mat &color, const cv::Mat &depth, const sensor_msgs::CameraInfo::ConstPtr cameraInfoColor,
              const sensor_msgs::CameraInfo::ConstPtr cameraInfoDepth, const cv::Rect &roi, std::vector<dlib::point> &keyPoints);

// publisher for head pose transformation
ros::Publisher transformPub;

string projectName = "head_pose";
// string headModelFileName = "/home/rnm_grp1/catkin_ws/src/" + projectName + "/data/head_model.stl";
string headModelFileName = "/home/rnm_grp1/catkin_ws/src/" + projectName + "/data/resampled_pc.pcd";
string face_cascade_name = "/home/rnm_grp1/catkin_ws/src/" + projectName + "/data/cascades.xml";
string landmarksFileName = "/home/rnm_grp1/catkin_ws/src/" + projectName + "/data/shape_predictor_68_face_landmarks.dat";
string topicColor = "/kinect2/qhd/image_color_rect";
string topicDepth = "/kinect2/qhd/image_depth_rect";

string window_name = "Capture - Face detection";
// bounding rectangle of face found in last image

// number of keyPoints for initial pose estimation
#define numKeyPoints 11
// indices into vector of facial landmarks detected by dlib's shapePredictor
int dlibKeyPointIndices[numKeyPoints] = {
    1,  // center of left ear
    15, // center of right ear
    27, // top of nose
    33, // bottom of nose
    36, // left edge of left eye
    39, // right edge of left eye
    42, // left edge of right eye
    45, // right edge of right eye
    48, // left edge of mouth
    54, // right edge of mouth
    8   // chin
};
// keyPoints in the head model
Eigen::Matrix<double, 3, Eigen::Dynamic> modelKeyPoints(3, numKeyPoints);

#if defined(DISPLAY_LANDMARKS) || defined(DISPLAY_FACES)
image_window win;
#endif

FaceDetector *faceDetector;
HeadPoseEstimator *headPoseEstimator;

#ifdef DISPLAY_CLOUD
pcl::visualization::PCLVisualizer::Ptr cloudVisualizer(new pcl::visualization::PCLVisualizer("Cloud Viewer"));
#endif

cv::Mat cameraMatrixColor = cv::Mat::zeros(3, 3, CV_64F);
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
        cv::Mat image, depth;
        readImage(imageColor, image);
        readImage(imageDepth, depth);
        // Detect face
        bool found = faceDetector->detectFace(image);
        if (found) {
          cv::Rect roi = faceDetector->getRoi();
          std::vector<dlib::point> keyPoints = faceDetector->getKeyPoints();
          findPose(image, depth, cameraInfoColor, cameraInfoDepth, roi, keyPoints);
#if defined(DISPLAY_FACES) || defined(DISPLAY_LANDMARKS)
          cv_image<bgr_pixel> cimg(image);
          win.clear_overlay();
          win.set_image(cimg);
  #ifdef DISPLAY_LANDMARKS
          for(int i = 0; i < numKeyPoints; i++) {
            for(int pIdx = 0; pIdx < keyPoints.size(); pIdx++) {
              win.add_overlay(dlib::image_window::overlay_rect(keyPoints.at(pIdx), rgb_pixel(255,0,0), std::to_string(dlibKeyPointIndices[pIdx])));
            }
          }
  #endif
  #ifdef DISPLAY_FACES
          dlib::rectangle face((long)roi.tl().x, (long)roi.tl().y, (long)roi.br().x - 1, (long)roi.br().y - 1);
          win.add_overlay(face);
  #endif
#endif
        } else {
          ROS_INFO("no face detected");
        }
#ifdef DISPLAY_CLOUD
        cloudVisualizer->spinOnce(10);
#endif
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", imageColor->encoding.c_str());
    }
}

void findPose(const cv::Mat &color, const cv::Mat &depth, const sensor_msgs::CameraInfo::ConstPtr cameraInfoColor,
              const sensor_msgs::CameraInfo::ConstPtr cameraInfoDepth, const cv::Rect &roi, std::vector<dlib::point> &keyPoints)
{
    static bool cloudInitialized = false;
    if (!cloudInitialized) {
        readCameraInfo(cameraInfoColor, cameraMatrixColor);
        /*
        double cameraParams[9] = { 1.0533457154589462e+03, 0., 9.5255170711209871e+02, 0.,
                                   1.0529715956757291e+03, 5.4050396155113071e+02, 0., 0., 1. };
        cameraMatrixColor = cv::Mat(3, 3, CV_64FC1, cameraParams);
        */
        // IR image input
        if(color.type() == CV_16U)
        {
            cv::Mat tmp;
            color.convertTo(tmp, CV_8U, 0.02);
            cv::cvtColor(tmp, color, CV_GRAY2BGR);
        }
        createLookup(color.cols, color.rows);
        headPoseEstimator->initLookUps(lookupX, lookupY);
        cloudInitialized = true;
    }

    std::pair<bool, Eigen::Matrix4d> transformationResult = headPoseEstimator->getTransformation(depth, roi, keyPoints);
    if (transformationResult.first)
    {
      Eigen::Matrix4d transformation = transformationResult.second;
#ifdef DISPLAY_CLOUD
      cloudViewer(headPoseEstimator->getHeadCloud(), headPoseEstimator->getTransformedModelCloud(), cv::Rect(cv::Point2i(0, 0), cv::Point2i(color.cols, color.rows)));
#endif
      std_msgs::Float64MultiArray msg;
      msg.data.clear();
      for (int idx = 0; idx < transformation.size(); idx++) {
        msg.data.push_back(*(transformation.data() + idx));
      }
      transformPub.publish(msg);
    } else
    {
      std::cout << "failed to detect head transformation" << std::endl << std::flush;
    }
}

void cloudViewer(pcl::PointCloud<CloudPoint>::Ptr cloud, pcl::PointCloud<CloudPoint>::Ptr cloud2, const cv::Rect roi)
{
    static bool viewerInitialized = false;
    static std::string cloudName = "rendered";

    if (!viewerInitialized) {
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> red(cloud, 255, 0, 0);
        cloudVisualizer->addPointCloud(cloud, red, cloudName);
        cloudVisualizer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, cloudName);
        if (cloud2 != NULL) {
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> green(cloud2, 0, 255, 0);
            cloudVisualizer->addPointCloud(cloud2, green, "cloud2");
            cloudVisualizer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud2");
        }
        cloudVisualizer->initCameraParameters();
        cloudVisualizer->setBackgroundColor(0, 0, 0);
        cloudVisualizer->setPosition(0, 0);
        cloudVisualizer->setSize(roi.width, roi.height);
        cloudVisualizer->setShowFPS(true);
        cloudVisualizer->setCameraPosition(0, 0, 0, 0, -1, 0);
        viewerInitialized = true;
    } else {
        cloudVisualizer->setSize(roi.width, roi.height);
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> red(cloud, 255, 0, 0);
        cloudVisualizer->updatePointCloud(cloud, cloudName);
        if (cloud2 != NULL) {
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> green(cloud2, 0, 255, 0);
            cloudVisualizer->updatePointCloud(cloud2, green, "cloud2");
        }
    }
}


int main(int argc, char **argv)
{
    // init model keyPoints
    modelKeyPoints.col(0) = Eigen::Vector3d(204.4, -290.0, 2236.0); // center of left ear
    modelKeyPoints.col(1) = Eigen::Vector3d(344.3, -290.4, 2245.0); // center of right ear
    modelKeyPoints.col(2) = Eigen::Vector3d(276.6, -291.2, 2148.0); // top of nose
    modelKeyPoints.col(3) = Eigen::Vector3d(278.6, -250.7, 2140.0); // bottom of nose
    modelKeyPoints.col(4) = Eigen::Vector3d(229.3, -293.2, 2171.0); // left edge of left eye
    modelKeyPoints.col(5) = Eigen::Vector3d(263.2, -295.7, 2168.0); // right edge of left eye
    modelKeyPoints.col(6) = Eigen::Vector3d(291.3, -294.6, 2170.0); // left edge of right eye
    modelKeyPoints.col(7) = Eigen::Vector3d(323.2, -298.0, 2169.0); // right edge of right eye
    modelKeyPoints.col(8) = Eigen::Vector3d(245.5, -230.2, 2181.0); // left edge of mouth
    modelKeyPoints.col(9) = Eigen::Vector3d(308.4, -230.4, 2182.0); // right edge of mouth
    modelKeyPoints.col(10) = Eigen::Vector3d(278.3, -188.7, 2192.0); // chin
    // initialize face detector
    faceDetector = new FaceDetector(landmarksFileName, dlibKeyPointIndices, numKeyPoints);
    // initialize head pose estimator
#ifdef USE_ICP
    bool useICP = true;
#else
    bool useICP = false;
#endif
    headPoseEstimator = new HeadPoseEstimator(headModelFileName, modelKeyPoints, useICP);

#ifdef DISPLAY_HEAD_MODEL
    pcl::visualization::PCLVisualizer::Ptr headVisualizer(new pcl::visualization::PCLVisualizer("Cloud Viewer"));
    headVisualizer->addPointCloud(headCloud, "headCloud");
#endif

    // initialize ros node
    ros::init(argc, argv, "pose_estimator");
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

    transformPub = nh.advertise<std_msgs::Float64MultiArray>("/headPose/transform", 100);

    // setup callback
    syncExact->registerCallback(imageCallback);

    /*
    pcl::PointCloud<CloudPoint>::Ptr ownCloud(new pcl::PointCloud<CloudPoint>);
    if (pcl::io::loadPCDFile<CloudPoint>("/home/rnm_grp1/catkin_ws/src/" + projectName + "/data/translated_pc.pcd", *ownCloud) == -1)
    {
      ROS_ERROR("Failed to load PCD ownCloud file\n");
    }
    std::vector<int> index;
    pcl::PointCloud<CloudPoint>::Ptr rmNan(new pcl::PointCloud<CloudPoint>());
    pcl::removeNaNFromPointCloud(*ownCloud, *rmNan, index);
    pcl::IterativeClosestPoint<CloudPoint, CloudPoint> icp;
    icp.setInputSource(rmNan);
    icp.setInputTarget(headCloud);
    // icp.setMaxCorrespondenceDistance(0.5f);
    pcl::PointCloud<CloudPoint>::Ptr Final(new pcl::PointCloud<CloudPoint>());
    icp.align(*Final);
    std::cout << "has converged:" << icp.hasConverged() << " score: " << icp.getFitnessScore() << std::endl;
    std::cout << icp.getFinalTransformation() << std::endl << std::flush;
    pcl::visualization::PCLVisualizer::Ptr headVisualizer(new pcl::visualization::PCLVisualizer("Cloud Viewer"));
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> red(headCloud, 255, 0, 0);
    headVisualizer->addPointCloud(headCloud, red, "headCloud");
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> green(Final, 0, 255, 0);
    headVisualizer->addPointCloud(Final, green, "finalCloud");
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> blue(ownCloud, 0, 0, 255);
    headVisualizer->addPointCloud(ownCloud, blue, "ownCloud");
    while(true) {
        headVisualizer->spinOnce(10);
    }
    */
    ros::spin();

    // free memory
    delete faceDetector;
    delete headPoseEstimator;
}

#undef DISPLAY_LANDMARKS
#undef DISPLAY_HEAD_MODEL
#undef DISPLAY_FACES
#undef DISPLAY_CLOUD
#undef COLORED_CLOUD
