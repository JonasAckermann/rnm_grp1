#include "facedetector.h"

//-------------------------------------------------------------------
//                      Predeclare helper methods
//-------------------------------------------------------------------
const cv::Rect scaleRect(const cv::Rect &rect, float scale);
int calcDiff(cv::Point point1, cv::Point point2);


//-------------------------------------------------------------------
//                          Public methods
//-------------------------------------------------------------------

FaceDetector::FaceDetector(const std::string &landmarksFilePath, int * const keyPointIndices, int numKeyPoints)
{
  // load landmarks file
  try {
      dlib::deserialize(landmarksFilePath) >> this->pose_model;
  } catch(dlib::serialization_error& e)
  {
      std::cout << "You need dlib's default face landmarking model file to run this example." << std::endl;
      std::cout << "You can get it from the following URL: " << std::endl;
      std::cout << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << std::endl;
      std::cout << std::endl << e.what() << std::endl << std::flush;
      throw e;
  }
  this->keyPointIndices = keyPointIndices;
  this->numKeyPoints = numKeyPoints;
}

bool FaceDetector::detectFace(const cv::Mat &image, float scaleLastRoi)
{
  this->previousKeyPoints.clear();
  // detect face in image
  // IDEAS: Rotate image along z-axis according to last transform to improve face detection and landmark detection
  if (this->detected)
  {
    return this->detectFaceInRegion(image, scaleRect(this->previousRoi, scaleLastRoi));
  } else
  {
    return this->detectFaceFull(image);
  }
}

cv::Rect FaceDetector::getRoi()
{
  if (this->detected)
  {
    return this->previousRoi;
  } else {
    throw std::logic_error("invalid state, invocation of getBoundingBox but face has not been detected");
  }
}

std::vector<dlib::point> FaceDetector::getKeyPoints()
{
  if (this->detected)
  {
    return this->previousKeyPoints;
  } else {
    throw std::logic_error("invalid state, invocation of getKeyPoints but face has not been detected");
  }
}

void FaceDetector::reset() {
  this->detected = false;
}


//-------------------------------------------------------------------
//                          Private methods
//-------------------------------------------------------------------

bool FaceDetector::detectFaceInRegion(const cv::Mat &image, const cv::Rect &region)
{
  // extract region from image
  const cv::Mat croppedImage = image(region);
  bool success = this->detectFaceFull(croppedImage);
  if (success)
  {
    std::cout << "detected in cropped image" << std::endl << std::flush;
    // adjust previousRoi and previousKeyPoints according to region, i.e. translate
    this->previousRoi = cv::Rect(this->previousRoi.x + region.x, this->previousRoi.y + region.y, this->previousRoi.width, this->previousRoi.height);
    std::cout << "roi: " << this->previousRoi.x << ", " << this->previousRoi.y << ", "  << this->previousRoi.width << ", "  << this->previousRoi.height << std::endl << std::flush;
    for (int idx = 0; idx < this->previousKeyPoints.size(); idx++) {
      dlib::point kp = this->previousKeyPoints.at(idx);
      if (kp != dlib::OBJECT_PART_NOT_PRESENT) {
        this->previousKeyPoints.at(idx) = dlib::point(kp.x() + region.x, kp.y() + region.y);
      }
    }
    this->detected = true;
    return true;
  } else {
    std::cout << "failed to detect in cropped image" << std::endl << std::flush;
    std::cout << "region: " << region.x << ", " << region.y << ", "  << region.width << ", "  << region.height  << std::endl << std::flush;
    return this->detectFaceFull(image);
  }
}

bool FaceDetector::detectFaceFull(const cv::Mat &image)
{
  dlib::cv_image<dlib::bgr_pixel> cimg(image);
  const std::vector<dlib::rectangle> faces = this->detector(cimg);
  if (faces.size() > 0) {
    const dlib::rectangle face = this->getMostLikelyFace(faces);
    std::tuple<bool, const cv::Rect, const std::vector<dlib::point>> detectionResult = this->detectKeyPointsAndRoi(cimg, face);
    if (std::get<0>(detectionResult))
    {
      this->previousKeyPoints = std::get<2>(detectionResult);
      this->previousRoi = std::get<1>(detectionResult);
      std::cout << "detected in full image" << std::endl << std::flush;
      std::cout << "roi: " << this->previousRoi.x << ", " << this->previousRoi.y << ", "  << this->previousRoi.width << ", "  << this->previousRoi.height << ", "  << std::endl << std::flush;
      this->detected = true;
    } else {
      std::cout << "Failed to detect enough key points" << std::endl << std::flush;
      this->detected = false;
    }
  } else {
    std::cout << "Failed to find a face in the full image" << std::endl << std::flush;
    this->detected = false;
  }
  return this->detected;
}

std::tuple<bool, const cv::Rect, const std::vector<dlib::point>> FaceDetector::detectKeyPointsAndRoi(const dlib::cv_image<dlib::bgr_pixel> &image, const dlib::rectangle &face)
{
  int detectedParts = 0;
  // bounds of dlib rectangle are right, bottom exclusive
  int minX = face.left(), minY = face.top(), maxX = face.right() + 1, maxY = face.bottom() + 1;
  // find facial keyPoints
  dlib::full_object_detection shape = this->pose_model(image, face);
  std::vector<dlib::point> keyPoints;
  for(int i = 0; i < this->numKeyPoints; i++) {
    int partIdx = this->keyPointIndices[i];
    dlib::point keyPoint = shape.part(partIdx);
    if (keyPoint != dlib::OBJECT_PART_NOT_PRESENT) {
      detectedParts++;
      // adjust bounds of roi
      if (keyPoint.x() > maxX) maxX = keyPoint.x();
      else if (keyPoint.x() < minX) minX = keyPoint.x();
      if (keyPoint.y() > maxY) maxY = keyPoint.y();
      else if (keyPoint.y() < minY) minY = keyPoint.y();
    }
    keyPoints.push_back(keyPoint);
  }
  bool detected = detectedParts >= 3;
  // let roi contain face and all detected keyPoints
  return std::make_tuple(detected, cv::Rect(cv::Point(minX, minY), cv::Point(maxX, maxY)), keyPoints);
}

const dlib::rectangle FaceDetector::getMostLikelyFace(const std::vector<dlib::rectangle> &faces)
{
  if (!this->detected || faces.size() == 1)
  {
    return faces.front();
  } else {
    // calculate previous width, height and center for comparision
    int lastWidth = this->previousRoi.width;
    int lastHeight = this->previousRoi.height;
    const cv::Point lastCenter(this->previousRoi.x + lastWidth >> 1, this->previousRoi.y + lastHeight >> 1);
    // init values for current best face to values of first face
    dlib::rectangle likelyFace = faces.front();
    int width = (likelyFace.right() - likelyFace.left()) >> 1;
    int height = (likelyFace.bottom() - likelyFace.top()) >> 1;
    // TODO - include scale in difference calculation?
    int diff = calcDiff(lastCenter, cv::Point((likelyFace.left() + likelyFace.right() + 1) >> 1, (likelyFace.top() + likelyFace.bottom() + 1) >> 1));
    for (int idx = 0; idx < faces.size(); idx++) {
      dlib::rectangle currentFace = faces.at(idx);
      int currentDiff = calcDiff(lastCenter, cv::Point((currentFace.left() + currentFace.right() + 1) >> 1, (currentFace.top() + currentFace.bottom() + 1) >> 1));
      if (currentDiff < diff) {
        diff = currentDiff;
        likelyFace = currentFace;
      }
    }
    return likelyFace;
  }
}


//-------------------------------------------------------------------
//                        Helper methods
//-------------------------------------------------------------------

const cv::Rect scaleRect(const cv::Rect &rect, float scale)
{
  float newWidth = rect.width * scale;
  float newHeight = rect.height * scale;
  float newX = rect.x - newWidth / 4;
  float newY = rect.y - newHeight / 4;
  return cv::Rect(static_cast<int>(newX), static_cast<int>(newY), static_cast<int>(newWidth), static_cast<int>(newHeight));
}

int calcDiff(cv::Point point1, cv::Point point2)
{
  int diffX = point1.x - point2.y;
  int diffY = point1.y - point2.y;
  return diffX * diffX + diffY * diffY;
}
