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
#include <pcl/point_types.h>
#include <pcl/conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/registration.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/common/transforms.h>

#include <Eigen/Geometry>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>

#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace dlib;
using namespace std;

#define DISPLAY_LANDMARKS
#undef DISPLAY_HEAD_MODEL
#define DISPLAY_FACES
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
void cloudViewer(pcl::PointCloud<CloudPoint>::Ptr cloud, pcl::PointCloud<CloudPoint>::Ptr cloud2, const cv::Rect roi);
void createCloud(const cv::Mat &depth, const cv::Mat &color, pcl::PointCloud<CloudPoint>::Ptr &cloud, const cv::Rect roi);
void pclCallback(const sensor_msgs::Image::ConstPtr imageColor, const sensor_msgs::Image::ConstPtr imageDepth, const sensor_msgs::CameraInfo::ConstPtr cameraInfoColor,
                 const sensor_msgs::CameraInfo::ConstPtr cameraInfoDepth, const cv::Rect roi, std::vector<dlib::point> keyPoints, bool detected);

// string headModelFileName = "/home/rnm_grp1/catkin_ws/src/rnmgrp1/data/head_model.stl";
string headModelFileName = "/home/rnm_grp1/catkin_ws/src/rnmgrp1/data/resampled_pc.pcd";
string face_cascade_name = "/home/rnm_grp1/catkin_ws/src/rnmgrp1/data/cascades.xml";
string landmarksFileName = "/home/rnm_grp1/catkin_ws/src/rnmgrp1/data/shape_predictor_68_face_landmarks.dat";
string topicColor = "/kinect2/qhd/image_color_rect";
string topicDepth = "/kinect2/qhd/image_depth_rect";
cv::CascadeClassifier face_cascade;
pcl::PointCloud<CloudPoint>::Ptr headCloud(new pcl::PointCloud<CloudPoint>);
string window_name = "Capture - Face detection";
// bounding rectangle of face found in last image
dlib::rectangle lastFace;
// boolean indicating whether we have found a face in the last image
bool foundLastFace = false;

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
Eigen::Matrix<double, 3, numKeyPoints> modelKeyPoints;

#if defined(DISPLAY_LANDMARKS) || defined(DISPLAY_FACES)
image_window win;
#endif

shape_predictor pose_model;

pcl::PointCloud<CloudPoint>::Ptr cloud;
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

std::pair<bool, dlib::rectangle> getFaceBoundingBox(cv_image<bgr_pixel> image) {
  static frontal_face_detector detector = get_frontal_face_detector();

  // detect faces
  std::vector<dlib::rectangle> faces = detector(image);
  if (faces.size() > 0) {
    return std::pair<bool, dlib::rectangle>(true, faces.front());
  } else {
    dlib::rectangle dummy;
    return std::pair<bool, dlib::rectangle>(false, dummy);
  }
}

void imageCallback(const sensor_msgs::Image::ConstPtr imageColor, const sensor_msgs::Image::ConstPtr imageDepth,
                   const sensor_msgs::CameraInfo::ConstPtr cameraInfoColor, const sensor_msgs::CameraInfo::ConstPtr cameraInfoDepth)
{
    try
    {
        cv_bridge::CvImageConstPtr imagePtr = cv_bridge::toCvShare(imageColor, "bgr8");
        cv::Mat image = imagePtr->image;
        // Turn OpenCV's Mat into something dlib can deal with.  Note that this just
        // wraps the Mat object, it doesn't copy anything.  So cimg is only valid as
        // long as temp is valid.  Also don't do anything to temp that would cause it
        // to reallocate the memory which stores the image as that will make cimg
        // contain dangling pointers.  This basically means you shouldn't modify temp
        // while using cimg.
        cv_image<bgr_pixel> cimg(image);
        // Detect faces
        cv::Rect roi;
        std::pair<bool, dlib::rectangle> getFaceResult = getFaceBoundingBox(cimg);
        bool found = getFaceResult.first;
        std::vector<dlib::point> keyPoints;
        if (found) {
            // Find the pose of face
            const dlib::rectangle face = getFaceResult.second;
            // bounds of dlib rectangle are right, bottom exclusive
            roi = cv::Rect(cv::Point2i(face.left(), face.top()), cv::Point2i(face.right() + 1, face.bottom() + 1));

            full_object_detection shape = pose_model(cimg, face);
            for(int i = 0; i < numKeyPoints; i++) {
              int partIdx = dlibKeyPointIndices[i];
              keyPoints.push_back(shape.part(partIdx));
            }
            // TODO - get roi as bounding box of all keyPoints and rect of face
#if defined(DISPLAY_FACES) || defined(DISPLAY_LANDMARKS)
            win.clear_overlay();
            win.set_image(cimg);
  #ifdef DISPLAY_LANDMARKS
            for(int i = 0; i < numKeyPoints; i++) {
              for(int pIdx = 0; pIdx < keyPoints.size(); pIdx++) {
                win.add_overlay(dlib::image_window::overlay_rect(keyPoints.at(pIdx), rgb_pixel(255,0,0), std::to_string(dlibKeyPointIndices[pIdx])));
              }
            }
            //win.add_overlay(render_face_detections(shape));
  #endif
  #ifdef DISPLAY_FACES
            win.add_overlay(face);
  #endif
#endif
        } else {
            // set default roi
            roi.width = image.cols / 10;
            roi.height = image.rows / 10;
            roi.x = (image.cols - roi.width) / 2;
            roi.y = (image.rows - roi.height) / 2;
        }
        // Display it all on the screen
        pclCallback(imageColor, imageDepth, cameraInfoColor, cameraInfoDepth, roi, keyPoints, found);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", imageColor->encoding.c_str());
    }
}

Eigen::Matrix<double, 3, numKeyPoints> mapKeyPointsTo3D(std::vector<dlib::point> keyPoints, cv::Mat depth)
{
  const float badPoint = std::numeric_limits<float>::quiet_NaN();
  Eigen::Matrix<double, 3, numKeyPoints> mapped;
  for (int idx = 0; idx < numKeyPoints; idx++) {
    dlib::point p = keyPoints.at(idx);
    const float y = lookupY.at<float>(0, p.y());
    const float *itX = lookupX.ptr<float>() + p.x();
    const uint16_t *itD = depth.ptr<uint16_t>(p.y()) + p.x();
    const float depthValue = *itD;
    if (*itD == 0) {
      mapped.col(idx) = Eigen::Vector3d(badPoint, badPoint, badPoint);
      ROS_INFO("oh no, a bad point in mapping 2d to 3d space");
    } else {
      mapped.col(idx) = Eigen::Vector3d(*itX * depthValue, y * depthValue, depthValue);
    }
  }
  return mapped;
}

Eigen::Matrix<double, 4, 4> getInitialHeadTransformation(std::vector<dlib::point> keyPoints, cv::Mat depth)
{
  // map keyPoints to 3D space
  Eigen::Matrix<double, 3, numKeyPoints> cloudKeyPoints = mapKeyPointsTo3D(keyPoints, depth);
  // get transform using umeyama method including scale
  Eigen::Matrix<double, 4, 4> transform = Eigen::umeyama(modelKeyPoints, cloudKeyPoints, true);
  return transform;
  // TODO - convert eigen matrix to TransformEstimationSVD or another estimation usable by icp
  //pcl::registration::TransformationEstimationSVD<CloudPoint, CloudPoint>::Ptr trans_svd (new pcl::registration::TransformationEstimationSVD<CloudPoint, CloudPoint>);
  //return trans_svd;
}

void pclCallback(const sensor_msgs::Image::ConstPtr imageColor, const sensor_msgs::Image::ConstPtr imageDepth, const sensor_msgs::CameraInfo::ConstPtr cameraInfoColor,
                 const sensor_msgs::CameraInfo::ConstPtr cameraInfoDepth, const cv::Rect roi, std::vector<dlib::point> keyPoints, bool detected)
{
    static bool cloudInitialized = false;
    cv::Mat color, depth;
    readImage(imageColor, color);
    readImage(imageDepth, depth);
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
        cloud = pcl::PointCloud<CloudPoint>::Ptr(new pcl::PointCloud<CloudPoint>());
        cloud->is_dense = false;
        createLookup(color.cols, color.rows);
        cloudInitialized = true;
    }
    cloud->height = roi.height;
    cloud->width = roi.width;
    cloud->points.resize(cloud->height * cloud->width);
    createCloud(depth, color, cloud, roi);
    if (detected) {
        Eigen::Matrix<double, 3, numKeyPoints> cloudKeyPoints = mapKeyPointsTo3D(keyPoints, depth);
        pcl::PointCloud<CloudPoint>::Ptr keyPointsCloud(new pcl::PointCloud<CloudPoint>());
        for (int idx = 0; idx < numKeyPoints; idx++) {
          dlib::point kp = keyPoints.at(idx);
          if (kp.y() >= roi.y && kp.y() <= roi.y + roi.height && kp.x() >= roi.x && kp.x() <= roi.x + roi.width) {
            CloudPoint cp = cloud->points[(kp.y() - roi.y) * roi.width + kp.x()];
            Eigen::Vector3d col = cloudKeyPoints.col(idx);
            std::cout << "keyPoint " << dlibKeyPointIndices[idx] << ", cp: " << cp << ", mapped" << col << std::endl;
            keyPointsCloud->push_back(cp);
          } else {
            ROS_INFO("keyPoint %d outside region of interest", dlibKeyPointIndices[idx]);
          }
          //Eigen::Vector3d col = cloudKeyPoints.col(idx);
          // !! TODO - conversion does not work
          //keyPointsCloud->push_back(pcl::PointXYZ((float)col(0,0), (float)col(1,0), (float)col(2,0)));
        }
        cloudViewer(cloud, keyPointsCloud, cv::Rect(cv::Point2i(0, 0), cv::Point2i(color.cols, color.rows)));
        /*
        std::vector<int> index;
        pcl::PointCloud<CloudPoint>::Ptr rmNan(new pcl::PointCloud<CloudPoint>());
        pcl::removeNaNFromPointCloud(*cloud, *rmNan, index);
        Eigen::Matrix<double, 4, 4> estimTrans = getInitialHeadTransformation(keyPoints, depth);
        pcl::transformPointCloud(*headCloud, *headCloud, estimTrans);
        pcl::IterativeClosestPoint<CloudPoint, CloudPoint> icp;
        //icp.setMaximumIterations(500);
        //icp.setTransformationEpsilon(1e-8);
        //icp.setEuclideanFitnessEpsilon(1);
        icp.setInputSource(headCloud);
        icp.setInputTarget(rmNan);
        // icp.setMaxCorrespondenceDistance(0.5f);
        pcl::PointCloud<CloudPoint>::Ptr Final(new pcl::PointCloud<CloudPoint>());
        icp.align(*Final);
        std::cout << "has converged:" << icp.hasConverged() << " score: " << icp.getFitnessScore() << std::endl;
        std::cout << icp.getFinalTransformation() << std::endl << std::flush;
        //pcl::PointCloud<pcl::PointXYZ>::Ptr transformedHead(new pcl::PointCloud<CloudPoint>());
        //pcl::transformPointCloud(headCloud, *transformedHead, icp.getFinalTransformation());
        //pcl::PointCloud<CloudPoint> combined;
        //pcl::concatenateFields (Final, *cloud, combined);
#ifdef DISPLAY_CLOUD
        cloudViewer(rmNan, headCloud, cv::Rect(cv::Point2i(0, 0), cv::Point2i(color.cols, color.rows)));
#endif
        */
    }
}

void cloudViewer(pcl::PointCloud<CloudPoint>::Ptr cloud, pcl::PointCloud<CloudPoint>::Ptr cloud2, const cv::Rect roi)
{
    static bool viewerInitialized = false;
    static pcl::visualization::PCLVisualizer::Ptr cloudVisualizer(new pcl::visualization::PCLVisualizer("Cloud Viewer"));
    static std::string cloudName = "rendered";

    if (!viewerInitialized) {
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> red(cloud, 255, 0, 0);
        cloudVisualizer->addPointCloud(cloud, red, cloudName);
        cloudVisualizer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, cloudName);
        if (cloud2 != NULL) {
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> green(cloud2, 0, 255, 0);
            cloudVisualizer->addPointCloud(cloud2, green, "cloud2");
            cloudVisualizer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 20, "cloud2");
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
            register const float depthValue = *itD;
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
    /*
    if (pcl::io::loadPolygonFileSTL(headModelFileName, mesh) == 0)
    {
      ROS_ERROR("Failed to load STL file\n");
    }
    */
    if (pcl::io::loadPCDFile<CloudPoint>(headModelFileName, *headCloud) == -1)
    {
      ROS_ERROR("Failed to load PCD head model file\n");
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
    // create pointCloud of head model, only required for stl import
    // pcl::fromPCLPointCloud2(mesh.cloud, *headCloud);
    std::cout << "width: " << headCloud->width << ", height: " << headCloud->height << std::endl;
    if (headCloud->is_dense) {
        std::cout << "is dense" << std::endl;
    }
    if (headCloud->isOrganized()) {
        std::cout << "is organized" << std::endl;
    } else {
        std::cout << "not organized" << std::endl;
    }
    std::cout << std::flush;
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

    /*
    pcl::PointCloud<CloudPoint>::Ptr ownCloud(new pcl::PointCloud<CloudPoint>);
    if (pcl::io::loadPCDFile<CloudPoint>("/home/rnm_grp1/catkin_ws/src/rnmgrp1/data/translated_pc.pcd", *ownCloud) == -1)
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
}

#undef DISPLAY_LANDMARKS
#undef DISPLAY_HEAD_MODEL
#undef DISPLAY_FACES
#undef DISPLAY_CLOUD
#undef COLORED_CLOUD
