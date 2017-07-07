#ifndef FACEDETECTOR_H
#define FACEDETECTOR_H

#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <stdio.h>

/**
 * @brief The FaceDetector class
 * Performs face and facial landmark detection.
 * For face detection dlibs frontal_face_detector is used and dlibs shape_predictor is used
 * for facial landmark detection.
 */
class FaceDetector {
  public:
  /**
     * @brief FaceDetector
     * @param landmarksFilePath Path to file containing facial landmark specifications
     * @param keyPointIndices Indices of keyPoints of intererst
     * @param numKeyPoints Number of keyPoints of interest
     */
    FaceDetector(const std::string &landmarksFilePath, int * const keyPointIndices, int numKeyPoints);
    /**
     * @brief detectFace Detect a face in given image, if a face has previously been detected this method
     * will try to follow that face
     * @param image Image in which the face shall be detected
     * @param scaleLastRoi Scale factor by which roi shall be upscaled, if a face has previously been detected
     * it is assumed that the face will stay in lastRoi * scaleLastRoi, where lastRoi is scaled around its center.
     * @return true if a face has been detected
     */
    bool detectFace(const cv::Mat &image, float scaleLastRoi = 2.0f);
    /**
     * @brief getRoi Get region of face detected by detectFace. The last call to detectFace must have returned true,
     * otherwise an error is raised
     * @return Rectangular region containing the face
     */
    cv::Rect getRoi();
    /**
     * @brief getKeyPoints Get list of detected keyPoints in image as detected by last invocation of detectFace. The last
     * call to detectFace must have returned true, otherwise an error is raised
     * @return
     */
    std::vector<dlib::point> getKeyPoints();
  private:
    // detector for face detection
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
    // detector for facial landmarks
    dlib::shape_predictor pose_model;
    // the roi found in last successful invocation of detectFace
    cv::Rect previousRoi;
    // keyPoints found in last successful invocation of detectFace
    std::vector<dlib::point> previousKeyPoints;
    // true if last invocation of detectFace was successful
    bool detected = false;
    // indices of keyPoints of interest
    int * keyPointIndices;
    // number of keyPoints of interest
    int numKeyPoints;

    /**
     * @brief detectFaceInRegion detect a face in a given region. This method also updates previousRoi and previousKeyPoints.
     * @param image Image in which to search for a face
     * @param region region in which faces shall be searched
     * @return true if a face was found
     */
    bool detectFaceInRegion(const cv::Mat &image, const cv::Rect &region);
    /**
     * @brief detectFaceFull detect a face in entire image. This method also updates previousRoi and previousKeyPoints.
     * @param image Image in which to search for a face
     * @return  true if a face was found
     */
    bool detectFaceFull(const cv::Mat &image);
    /**
     * @brief detectKeyPointsAndRoi Find keyPoints in an image and create a region of interest around them
     * @param image Image in which to search for keyPoints
     * @param face region in which to search for the keyPoints
     * @return true if keyPoints were detected, roi and keyPoints
     */
    std::tuple<bool, const cv::Rect, const std::vector<dlib::point>> detectKeyPointsAndRoi(const dlib::cv_image<dlib::bgr_pixel> &image, const dlib::rectangle &face);
    /**
     * @brief getMostLikelyFace Get the face which is closest to previously detected face
     * @param faces List of rectangles containing faces
     * @return rectangle containing face which is most likely the same as detected previously
     */
    const dlib::rectangle getMostLikelyFace(const std::vector<dlib::rectangle> &faces);
};

#endif // FACEDETECTOR_H
