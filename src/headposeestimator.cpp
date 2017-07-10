#include "headposeestimator.h"


//-------------------------------------------------------------------
//                          Public methods
//-------------------------------------------------------------------

HeadPoseEstimator::HeadPoseEstimator(const std::string &headModelFilePath, const Eigen::Matrix3Xd &modelKeyPoints, bool useICP)
  : modelKeyPoints(modelKeyPoints), modelPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>())), useICP(useICP)
{
  if (pcl::io::loadPCDFile<pcl::PointXYZ>(headModelFilePath, *this->modelPointCloud) == -1)
  {
    std::cout << "Failed to load PCD head model file" << std::endl << std::flush;
    throw std::invalid_argument("PCD file for head model not found");
  }
  this->expandedModelKeyPoints = Eigen::Matrix4Xd::Ones(4, this->modelKeyPoints.cols());
  this->expandedModelKeyPoints.block(0, 0, 3, this->modelKeyPoints.cols()) = this->modelKeyPoints.block(0, 0, 3, this->modelKeyPoints.cols());
  // create pointCloud of head model, only required for stl import
  pcl::RadiusOutlierRemoval<pcl::PointXYZ> outrem;
  outrem.setInputCloud(this->modelPointCloud);
  outrem.setRadiusSearch(10);
  outrem.setMinNeighborsInRadius(10);
  outrem.filter(*this->modelPointCloud);
}

const float HeadPoseEstimator::badPoint = std::numeric_limits<float>::quiet_NaN();

void HeadPoseEstimator::initLookUps(cv::Mat &lookUpX, cv::Mat &lookUpY)
{
  this->lookUpX = lookUpX;
  this->lookUpY = lookUpY;
  this->headCloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
  this->headCloud->is_dense = false;
  this->lookUpsInitialized = true;
}

std::pair<bool, const Eigen::Matrix4d> HeadPoseEstimator::getTransformation(const cv::Mat &depthImage, const cv::Rect &roi, const std::vector<dlib::point> &keyPoints)
{
  if (!this->lookUpsInitialized)
  {
    throw std::logic_error("invalid state, invocation of getTransformation but lookUps have not been initialized");
  } else
  {
    this->updateHeadCloud(depthImage, roi);
    std::pair<bool, const Eigen::Matrix4d> estimTransResult = this->getInitialHeadTransformation(keyPoints, depthImage);
    if (!estimTransResult.first)
    {
      this->validTransform = false;
    } else
    {
      Eigen::Matrix4d estimTrans = estimTransResult.second;
      pcl::PointCloud<pcl::PointXYZ>::Ptr transformedModelCloud(new pcl::PointCloud<pcl::PointXYZ>());
      pcl::transformPointCloud(*this->modelPointCloud, *transformedModelCloud, estimTrans);
      Eigen::Matrix4d transformation;
      if (this->useICP)
      {
        pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
        icp.setInputSource(transformedModelCloud);
        icp.setInputTarget(this->headCloud);
        icp.setMaxCorrespondenceDistance(9); // 9 mm
        icp.setRANSACOutlierRejectionThreshold(9);
        pcl::PointCloud<pcl::PointXYZ>::Ptr Final(new pcl::PointCloud<pcl::PointXYZ>());
        icp.align(*Final);
        std::cout << "has converged:" << icp.hasConverged() << " score: " << icp.getFitnessScore() << std::endl;
        std::cout << icp.getFinalTransformation() << std::endl << std::flush;
        transformation.noalias() = icp.getFinalTransformation().cast<double>() * estimTrans;
      } else
      {
        transformation = estimTrans;
      }
      this->validTransform = this->validateTransform(transformation);
      this->lastTransform = transformation;
    }
  }
  return std::pair<bool, Eigen::Matrix4d>(this->validTransform, this->lastTransform);
}

const pcl::PointCloud<pcl::PointXYZ>::Ptr HeadPoseEstimator::getTransformedModelCloud()
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr transformedModelCloud(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::transformPointCloud(*this->modelPointCloud, *transformedModelCloud, this->lastTransform);
  return transformedModelCloud;
}

const pcl::PointCloud<pcl::PointXYZ>::Ptr HeadPoseEstimator::getHeadCloud()
{
  return this->headCloud;
}

void HeadPoseEstimator::reset() {
  this->validTransform = false;
}


//-------------------------------------------------------------------
//                          Private methods
//-------------------------------------------------------------------

bool HeadPoseEstimator::validateTransform(const Eigen::Matrix4d &transform) {
  Eigen::Matrix4Xd transformedModelKeyPoints = transform * this->expandedModelKeyPoints;
  if (this->validTransform) {
    double norm = (this->transformedModelKeyPoints - transformedModelKeyPoints).squaredNorm();
    std::cout << "norm of transform diff: " << norm << std::endl << std::flush;
    // allow for a difference of 5 per component per column
    return norm < 4 * 5 * 5 * this->modelKeyPoints.cols();
  } else {
    this->transformedModelKeyPoints = transformedModelKeyPoints;
    return true;
  }
}

void HeadPoseEstimator::updateHeadCloud(const cv::Mat &depthImage, const cv::Rect &roi)
{
  this->headCloud->width = roi.width;
  this->headCloud->height = roi.height;
  this->headCloud->points.resize(roi.width * roi.height);

  this->writeHeadCloud(depthImage, roi);
  this->filterHeadCloud();
}

void HeadPoseEstimator::writeHeadCloud(const cv::Mat &depthImage, const cv::Rect &roi)
{
  #pragma omp parallel for
  for(int r = roi.y; r < roi.y + roi.height; ++r)
  {
      pcl::PointXYZ *itP = &this->headCloud->points[(r - roi.y) * roi.width];
      const uint16_t *itD = depthImage.ptr<uint16_t>(r) + roi.x;
      const float y = this->lookUpY.at<float>(0, r);
      const float *itX = this->lookUpX.ptr<float>() + roi.x;

      for(size_t c = roi.x; c < (size_t)(roi.x + roi.width); ++c, ++itP, ++itD, ++itX)
      {
          register const float depthValue = *itD;
          // Check for invalid measurements
          if(*itD == 0)
          {
              // not valid
              itP->x = itP->y = itP->z = this->badPoint;
              continue;
          }
          itP->z = depthValue;
          itP->x = *itX * depthValue;
          itP->y = y * depthValue;
      }
  }
}

void HeadPoseEstimator::filterHeadCloud()
{
  Eigen::Vector4f centroid;
  pcl::compute3DCentroid(*this->headCloud, centroid);
  pcl::PointCloud<pcl::PointXYZ>::Ptr filteredCloud(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::PassThrough<pcl::PointXYZ> pass;
  pass.setInputCloud(this->headCloud);
  pass.setFilterFieldName("z");
  pass.setFilterLimits(0.0, centroid(2, 0) + 20);
  pass.filter(*filteredCloud);
  std::vector<int> index;
  pcl::PointCloud<pcl::PointXYZ>::Ptr rmNan(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::removeNaNFromPointCloud(*filteredCloud, *rmNan, index);
  this->headCloud = rmNan;
}

std::pair<bool, const Eigen::Matrix4d> HeadPoseEstimator::getInitialHeadTransformation(const std::vector<dlib::point> &keyPoints, const cv::Mat &depthImage)
{
  // map keyPoints to 3D space ignoring NaN
  Eigen::Matrix3Xd cloudKeyPoints = this->mapKeyPointsTo3D(keyPoints, depthImage);
  std::vector<int> validColIndices;
  for (int idx = 0; idx < keyPoints.size(); idx++) {
    if (!std::isnan(cloudKeyPoints(0, idx))) {
      validColIndices.push_back(idx);
    }
  }
  if (validColIndices.size() >= 3)
  {
    Eigen::Matrix3Xd validCloudKeyPoints(3, validColIndices.size());
    Eigen::Matrix3Xd validModelKeyPoints(3, validColIndices.size());
    for (int idx = 0; idx < validColIndices.size(); idx++) {
      validCloudKeyPoints.col(idx) = cloudKeyPoints.col(validColIndices.at(idx));
      validModelKeyPoints.col(idx) = this->modelKeyPoints.col(validColIndices.at(idx));
    }

    // get transform using umeyama method including scale
    Eigen::Matrix4d transform = Eigen::umeyama(validModelKeyPoints, validCloudKeyPoints, true);
    std::cout << "transform" << std::endl << transform << std::endl << std::flush;
    return std::pair<bool, const Eigen::Matrix4d>(true, transform);
  } else
  {
    std::cout << "not enough correspondances to find initial tansformation" << std::endl << std::flush;
    return std::pair<bool, const Eigen::Matrix4d>(false, Eigen::Matrix4d());
  }
}

const Eigen::Matrix3Xd HeadPoseEstimator::mapKeyPointsTo3D(const std::vector<dlib::point> &keyPoints, const cv::Mat &depthImage)
{
  int numKeyPoints = keyPoints.size();
  Eigen::Matrix3Xd mapped(3, numKeyPoints);
  for (int idx = 0; idx < numKeyPoints; idx++) {
    dlib::point p = keyPoints.at(idx);
    const float y = this->lookUpY.at<float>(0, p.y());
    const float *itX = this->lookUpX.ptr<float>() + p.x();
    const uint16_t *itD = depthImage.ptr<uint16_t>(p.y()) + p.x();
    const float depthValue = *itD;
    if (*itD == 0) {
      mapped.col(idx) = Eigen::Vector3d(this->badPoint, this->badPoint, this->badPoint);
      std::cout << "oh no, a bad point in mapping 2d to 3d space" << std::endl << std::flush;
    } else {
      mapped.col(idx) = Eigen::Vector3d(*itX * depthValue, y * depthValue, depthValue);
    }
  }
  return mapped;
}
