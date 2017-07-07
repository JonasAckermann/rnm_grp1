#ifndef HEADPOSEESTIMATOR_H
#define HEADPOSEESTIMATOR_H

#include <dlib/opencv.h>

#include <pcl/point_cloud.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/passthrough.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/icp.h>
#include <pcl/common/centroid.h>

#include <Eigen/Geometry>

#include <stdio.h>

/**
 * @brief The HeadPoseEstimator class
 * Estimates the pose of a head using iterative closest point based on a point cloud of a model and a point cloud generated
 * from a given depth image. For an initial transformation estimation keyPoints in the head model are used.
 */
class HeadPoseEstimator {
  public:
  /**
     * @brief HeadPoseEstimator
     * @param headModelFilePath Path to file containing point cloud of head model
     * @param modelKeyPoints list of keyPoints in point cloud of head model
     * @param useICP true if icp shall be used, otherwise transformation from estimation is returned
     */
    HeadPoseEstimator(const std::string &headModelFilePath, const Eigen::Matrix3Xd &modelKeyPoints, bool useICP = true);
    /**
     * @brief initLookUps Initialize the lookUp tables for 2D to 3D point conversion
     * @param lookUpX
     * @param lookUpY
     */
    void initLookUps(cv::Mat &lookUpX, cv::Mat &lookUpY);
    /**
     * @brief getTransformation Find transformation between a point in the head model and the generated point cloud of a head in the depth image
     * @param depthImage Depth image containg head
     * @param roi Region of interest in depth image containing a head
     * @param keyPoints keyPoints in 2D space detected on head in roi
     * @return true if a transformation was found, transformation
     */
    std::pair<bool, const Eigen::Matrix4d> getTransformation(const cv::Mat &depthImage, const cv::Rect &roi, const std::vector<dlib::point> &keyPoints);
    /**
     * @brief badPoint Representation of a bad point (measurement error) in the point cloud
     */
    static const float badPoint;
    /**
     * @brief getTransformedModelCloud
     * @return Point cloud of head model transformed onto the detected head cloud
     */
    const pcl::PointCloud<pcl::PointXYZ>::Ptr getTransformedModelCloud();
    /**
     * @brief getHeadCloud
     * @return Point cloud of detected head
     */
    const pcl::PointCloud<pcl::PointXYZ>::Ptr getHeadCloud();
  private:
    cv::Mat lookUpX;
    cv::Mat lookUpY;
    bool useICP = true;
    bool lookUpsInitialized = false;
    const Eigen::Matrix3Xd modelKeyPoints;
    Eigen::Matrix4Xd expandedModelKeyPoints;
    Eigen::Matrix4Xd transformedModelKeyPoints;
    const pcl::PointCloud<pcl::PointXYZ>::Ptr modelPointCloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr headCloud;
    bool validTransform = false;
    Eigen::Matrix4d lastTransform;

    /**
     * @brief validateTransform Validates given transformation by comparing transformed modelKeyPoints with modelKeyPoints transformed by previous transformation.
     * @param transform Transform to validate
     * @return True if transform is close enough to last transform to be considered valid
     */
    bool validateTransform(const Eigen::Matrix4d &transform);
    /**
     * @brief updateHeadCloud Updates the point cloud of the detected head
     * @param depthImage Depth image containing the head
     * @param roi region of interest in depthImage where the head is located
     */
    void updateHeadCloud(const cv::Mat &depthImage, const cv::Rect &roi);
    /**
     * @brief writeHeadCloud Write data of head cloud to variable
     * @param depthImage DepthImage containing head
     * @param roi Region of interest in depthImage where the head is located
     */
    void writeHeadCloud(const cv::Mat &depthImage, const cv::Rect &roi);
    /**
     * @brief filterHeadCloud Filter the point cloud of the head to reduce outliers
     */
    void filterHeadCloud();
    /**
     * @brief getInitialHeadTransformation Get estimation of transformation of head cloud based on keyPoints
     * @param keyPoints list of 2D keyPoints detected on face
     * @param depthImage DepthImage containing head
     * @return  true if a transformation was found, transformation
     */
    std::pair<bool, const Eigen::Matrix4d> getInitialHeadTransformation(const std::vector<dlib::point> &keyPoints, const cv::Mat &depthImage);
    /**
     * @brief mapKeyPointsTo3D Map points from 2D space into 3D space based on the depthImage
     * @param keyPoints List of 2D points to map
     * @param depthImage
     * @return Matrix where the columns contain the mapped points
     */
    const Eigen::Matrix3Xd mapKeyPointsTo3D(const std::vector<dlib::point> &keyPoints, const cv::Mat &depthImage);
};

#endif // HEADPOSEESTIMATOR_H
