#pragma once

#include "data_structure.hpp"
#include "frame.hpp"
#include "map.hpp"

#include <vector>
#include <string>
#include <unistd.h>

namespace slam
{

enum State
{
    Init,
    Track,
    Lost
};

class VO
{

public:
    // eigen macro for fixed-size vectorizable Eigen types
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    Frame::Ptr frame_last_;
    Frame::Ptr frame_current_;
    Map &map_;

    std::string dataset_;
    cv::Ptr<cv::FeatureDetector> detector_;
    cv::Ptr<cv::DescriptorExtractor> descriptor_;
    cv::Ptr<cv::DescriptorMatcher> matcher_;

    int num_inliers_ = 0; // number of inliers after RANSAC

    SE3 T_cl_ = SE3(); // T_current(camera)_last(camera)
    SE3 T_cw_ = SE3(); // T_current(camera)_world

    int seq_ = 1; // sequence number

    State state_ = Init; // current tracking state
    int num_lost_ = 0;        // number of continuous lost frames

    int curr_keyframe_id_ = 0;
    int curr_landmark_id_ = 0;

    //bool if_rviz_;


    VO(Map &map);

    /*! \brief initialize VO
    *
    *  \param dataset - the address of the dataset
    *  \param map - the map owned by the ros node
    */
    VO(std::string dataset, Map &map);

    /*! \brief read left and right images into a frame
    *
    *  \param id - id of the frame, starts from zero
    *  \param frame - the frame onject
    *  \return if the reading is successful
    */
    int read_frame(int id, Frame::Ptr frame);

    /*! \brief calculate the disparity map from two left-right images
    *
    *  \param frame - a frame includes both L/R images
    *  \param disparity - the postion the disparity map is write to
    *  \return if the calculation is successful
    */
    int disparity_map(Frame::Ptr frame, cv::Mat &disparity);

    /*! \brief initialization pipline
    * 
    *  \return if successful
    */
    bool init();

    /*! \brief tracking pipline
    * 
    *  \param if_insert_keyframe - return whether this frame is added as keyframe
    *  \return whether the motion estimation is rejected or not
    */
    bool tracking(bool &if_insert_keyframe);

    /*! \brief feature detection
    *
    *  \param img - the image to detect features
    *  \param keypoints - the keypoints detected
    *  \param descriptors - the descriptors detected
    *  \return if_successful
    */
    int feature_detection(const cv::Mat &img, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors);

    /*! \brief feature matching
    *
    *  \param descriptors_1 - the descriptors in first image
    *  \param descriptors_2 - the descriptors in second image
    *  \param feature_matches - feature matches
    *  \return if successful
    */
    int feature_matching(const cv::Mat &descriptors_1, const cv::Mat &descriptors_2,
                         std::vector<cv::DMatch> &feature_matches);

    /*! \brief set reference 3d positions for the last frame
    *
    * filter the descriptors in the reference frame, calculate the 3d positions of those features
    * 
    *  \param pts_3d - output of 3d points
    *  \param keypoints - filtered keypoints
    *  \param descriptors - filted descriptors
    *  \param frame - the current frame
    *  \return - a list of reliable depth
    */
    std::vector<bool> get_3d_position(std::vector<cv::Point3f> &pts_3d, std::vector<cv::KeyPoint> &keypoints,
                                          cv::Mat &descriptors, Frame::Ptr frame);

    /*! \brief estimate the motion using PnP
    *
    *  \param frame - the current frame to etimate motion
    */
    void motion_estimation(Frame::Ptr frame);

    /*! \brief check if the estimated motion is valid
    * 
    *  \return if valid
    */
    bool check_motion_estimation();

    /*! \brief move everything from current frame to last frame
    */
    void move_frame();

    /*! \brief adaptive non-maximal supression algorithm to generate uniformly distributed feature points
    *  \param keypoints - keypoints list
    *  \param num - number of keypoints to keep
    */
    void adaptive_non_maximal_suppresion(std::vector<cv::KeyPoint> &keypoints,
                                         const int num);

    /*! \brief pipeline of the tracking thread
    *
    *  \param if_insert_keyframe - return whether this frame is added as keyframe
    *  \return return false if VO is lost
    */
    bool pipeline(bool &if_insert_keyframe);

    /*! \brief insert current frame as the keyframe
    *
    *  \param check - the frame is rejected or not
    *  \param pts_3d - output of 3d points
    *  \param keypoints - filtered keypoints
    *  \param descriptors - filted descriptors
    */
    bool insert_key_frame(bool check, std::vector<cv::Point3f> &pts_3d, std::vector<cv::KeyPoint> &keypoints,
                          cv::Mat &descriptors);
};

} // namespace slam