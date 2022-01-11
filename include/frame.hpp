#pragma once

#include "data_structure.hpp"
#include <memory>

namespace slam
{

class Frame
{

public:
    // eigen macro for fixed-size vectorizable Eigen types
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Frame> Ptr;

    // data memebers
    int frame_id_ = -1;
    cv::Mat left_img_, right_img_;
    cv::Mat disparity_;

    SE3 T_cw_ = SE3(); // T_current(camera)_world

    bool is_keyframe_ = false;
    int keyframe_id_ = -1;
    std::vector<Feature> features_;

    // camera intrinsic parameters
    double fx_ = 718.856, fy_ = 718.856, cx_ = 607.1928, cy_ = 185.2157;
    double b_ = 0.573;

    Frame() = default;

    Frame(int frame_id, const cv::Mat &left, const cv::Mat &right)
        : frame_id_(frame_id), left_img_(left), right_img_(right) {}
    ~Frame(){}
    
    static Frame::Ptr createFrame();
    static Frame::Ptr createFrame(int frame_id, const cv::Mat &left, const cv::Mat &right);

    /*! find the 3D position of a feature point
     *  @param kp - keypoint in the image
     *  @param relative_pt3d - the relative 3d position in this frame
     *  @return 3D position of the landmark in world coordinate
    */
    Eigen::Vector3d find_world_frame_pos (cv::KeyPoint &kp, Eigen::Vector3d &relative_pt3d);

    /*! fill in the remaining information needed for the frame 
     *  \param T_cw - transformation T_c_w
     *  \param is_keyframe - whether this frame is a keyframe
     *  \param keyframe_id - keyframe id (different from frame id)
    */
    void fill_frame(SE3 T_cw, bool is_keyframe, int keyframe_id);
};

}