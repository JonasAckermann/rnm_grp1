#include <ros/ros.h>
#include <stdio.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <cv_bridge/cv_bridge.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>

#include <queue>

#define PARAM_BUFFER_SIZE "buffer_size"
#define DEFAULT_BUFFER_SIZE 5

typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo, sensor_msgs::CameraInfo> ExactSyncPolicy;

const std::string topicColor = "/kinect2/qhd/image_color_rect";
const std::string topicDepth = "/kinect2/qhd/image_depth_rect";
const std::string pubColorTopic = "/headPose/color/rect";
const std::string pubColorInfoTopic = "/headPose/color/camera_info";
const std::string pubDepthTopic = "/headPose/depth/rect";
const std::string pubDepthInfoTopic = "/headPose/depth/camera_info";

// publishers
ros::Publisher pubColor;
ros::Publisher pubColorInfo;
ros::Publisher pubDepth;
ros::Publisher pubDepthInfo;

int bufferSize;

void imageCallback(const sensor_msgs::Image::ConstPtr imageColor, const sensor_msgs::Image::ConstPtr imageDepth,
                   const sensor_msgs::CameraInfo::ConstPtr cameraInfoColor, const sensor_msgs::CameraInfo::ConstPtr cameraInfoDepth);
void readImage(const sensor_msgs::Image::ConstPtr msgImage, cv::Mat &image);
void createImage(const cv::Mat &image, const std_msgs::Header &header, sensor_msgs::Image &msgImage);

/**
 * @brief updateAverageDepthImage Update the average depth image
 * @param newDepthImage New depth image to process
 * @return currentAverage
 */
const cv::Mat updateAverageDepthImage(cv::Mat &newDepthImage);


int main(int argc, char **argv) {
    // initialize ros node
    ros::init(argc, argv, "depth_averaging");
    ros::NodeHandle nh;
    ros::NodeHandle nhPrivate("~");

    // get console args
    if (!nhPrivate.getParam(PARAM_BUFFER_SIZE, bufferSize)) {
      bufferSize = DEFAULT_BUFFER_SIZE;
    }
    std::cout << PARAM_BUFFER_SIZE << ": " << bufferSize << std::endl << std::flush;

    image_transport::ImageTransport it(nh);

    // subscribe to kinect_bridge data
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

    pubColor = nh.advertise<sensor_msgs::Image>(pubColorTopic, 100);
    pubColorInfo = nh.advertise<sensor_msgs::CameraInfo>(pubColorInfoTopic, 100);
    pubDepth = nh.advertise<sensor_msgs::Image>(pubDepthTopic, 100);
    pubDepthInfo = nh.advertise<sensor_msgs::CameraInfo>(pubDepthInfoTopic, 100);

    // setup callback
    syncExact->registerCallback(imageCallback);

    ros::spin();
}

void imageCallback(const sensor_msgs::Image::ConstPtr imageColor, const sensor_msgs::Image::ConstPtr imageDepth,
                   const sensor_msgs::CameraInfo::ConstPtr cameraInfoColor, const sensor_msgs::CameraInfo::ConstPtr cameraInfoDepth) {
  cv::Mat depth;
  readImage(imageDepth, depth);
  const cv::Mat averagedDepth = updateAverageDepthImage(depth);
  // create depth image msg
  sensor_msgs::ImagePtr depthMsg = sensor_msgs::ImagePtr(new sensor_msgs::Image);
  std_msgs::Header _header = imageDepth->header;
  createImage(averagedDepth, _header, *depthMsg);

  pubColor.publish(imageColor);
  pubColorInfo.publish(cameraInfoColor);
  pubDepth.publish(*depthMsg);
  pubDepthInfo.publish(cameraInfoDepth);
}

void readImage(const sensor_msgs::Image::ConstPtr msgImage, cv::Mat &image) {
    cv_bridge::CvImageConstPtr pCvImage;
    pCvImage = cv_bridge::toCvShare(msgImage, msgImage->encoding);
    pCvImage->image.copyTo(image);
}

void createImage(const cv::Mat &image, const std_msgs::Header &header, sensor_msgs::Image &msgImage) {
  size_t step, size;
  step = image.cols * image.elemSize();
  size = image.rows * step;

  msgImage.encoding = sensor_msgs::image_encodings::TYPE_16UC1;
  msgImage.header = header;
  msgImage.height = image.rows;
  msgImage.width = image.cols;
  msgImage.is_bigendian = false;
  msgImage.step = step;
  msgImage.data.resize(size);
  memcpy(msgImage.data.data(), image.data, size);
}

const cv::Mat updateAverageDepthImage(cv::Mat &newDepthImage) {
  static std::queue<cv::Mat> depthImageBuffer;
  static std::queue<cv::Mat> usedImagesBuffer;
  static cv::Mat currentAverage;
  if (depthImageBuffer.empty()) {
    currentAverage = cv::Mat::zeros(newDepthImage.rows, newDepthImage.cols, CV_16UC1);
  }
  if (depthImageBuffer.size() < bufferSize) {
    ROS_INFO("< bufferSize, %d, %d", depthImageBuffer.size(), bufferSize);
    cv::Mat previousUsedImages;
    if (depthImageBuffer.empty()) {
      previousUsedImages = cv::Mat::zeros(currentAverage.cols, currentAverage.rows, CV_8UC1);
    } else {
      previousUsedImages = depthImageBuffer.back();
    }
    cv::Mat usedImages = previousUsedImages.clone();
    for (int r = 0; r < currentAverage.rows; r++) {
      uint16_t *avg = currentAverage.ptr<uint16_t>(r);
      uint16_t *img = newDepthImage.ptr<uint16_t>(r);
      uint8_t *numImgs = usedImages.ptr<uint8_t>(r);
      for (int c = 0; c < currentAverage.cols; c++, avg++, img++, numImgs++) {
        if (*img != 0) {
          *numImgs += 1;
          *avg += (*img - *avg) / *numImgs;
        }
      }
    }
    depthImageBuffer.push(newDepthImage);
    usedImagesBuffer.push(usedImages);
  } else {
    ROS_INFO("= bufferSize");
    cv::Mat removedImage = depthImageBuffer.front();
    // will be updated for added image
    cv::Mat removedUsedImages = usedImagesBuffer.front();
    cv::Mat previousUsedImages = usedImagesBuffer.back();
    for (int r = 0; r < currentAverage.cols; r++) {
      uint16_t *avg = currentAverage.ptr<uint16_t>(r);
      uint16_t *img = newDepthImage.ptr<uint16_t>(r);
      uint16_t *rImg = removedImage.ptr<uint16_t>(r);
      uint8_t *numImgs = removedUsedImages.ptr<uint8_t>(r);
      uint8_t *prevNumImgs = previousUsedImages.ptr<uint8_t>(r);
      for (int c = 0; c < currentAverage.rows; c++, avg++, numImgs++, prevNumImgs++, img++, rImg++) {
        if (*rImg != 0) {
          *avg -= (*rImg - *avg) / *numImgs;
          *numImgs = *prevNumImgs - 1;
        } else {
          *numImgs = *prevNumImgs;
        }
        if (*img != 0) {
          *numImgs += 1;
          *avg += (*img - *avg) / *numImgs;
        }
      }
    }
    depthImageBuffer.pop();
    usedImagesBuffer.pop();
    depthImageBuffer.push(newDepthImage);
    usedImagesBuffer.push(removedUsedImages);
  }
  return currentAverage;
}
