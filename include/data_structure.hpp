#pragma once

#include <iostream>
#include <cmath>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>

#include "sophus/se3.hpp"
#include "sophus/so3.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

using SE3 = Sophus::SE3d;
using SO3 = Sophus::SO3d;

namespace slam
{
struct Feature
{
public:
    int feature_id_;
    int frame_id_;
    int landmark_id_ = -1;
    cv::KeyPoint keypoint_; // 2d position in the pixel frame
    cv::Mat descriptor_;    // feature descriptor of this landmark
    bool is_inlier = false;

    Feature() {}
    Feature(int feature_id, int frame_id, cv::KeyPoint keypoint, cv::Mat descriptor)
        : feature_id_(feature_id), frame_id_(frame_id), keypoint_(keypoint), descriptor_(descriptor) {}
    Feature(int feature_id, int frame_id, int landmark_id, cv::KeyPoint keypoint, cv::Mat descriptor)
        : feature_id_(feature_id), frame_id_(frame_id), landmark_id_(landmark_id), keypoint_(keypoint), descriptor_(descriptor) {}
};

struct Observation
{
    int keyframe_id_;
    int feature_id_;
    bool to_delete = false;

    Observation(int keyframe_id, int feature_id)
        : keyframe_id_(keyframe_id), feature_id_(feature_id) {}
};

struct Landmark
{

public:
    int landmark_id_;
    cv::Point3f pt_3d_;      // 3d position in the world frame;
    cv::Mat descriptor_;     // feature descriptor of this landmark
    int observed_times_ = 1; // number of times being observed
    std::vector<Observation> observations_;
    bool is_inlier = true;
    bool reliable_depth_ = false;

    Landmark() {}

    Landmark(int landmark_id, cv::Point3f pt_3d, cv::Mat descriptor, bool reliable_depth, Observation observation)
        : landmark_id_(landmark_id), pt_3d_(pt_3d), descriptor_(descriptor), reliable_depth_(reliable_depth)
    {
        observations_.push_back(observation);
    }

    /*! \brief convert a OpenCV point3f to Eigen Vector3d
     * 
     *  \return converted Eigen Vector3d
    */
    Eigen::Vector3d to_vector_3d()
    {
        Eigen::Vector3d pos_vec;
        pos_vec << pt_3d_.x, pt_3d_.y, pt_3d_.z;
        return pos_vec;
    }
};


}
