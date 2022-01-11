#include "data_structure.hpp"
#include "frame.hpp"

namespace slam
{
Frame::Ptr Frame::createFrame()
{
    return Frame::Ptr(new Frame());
}

Frame::Ptr Frame::createFrame(int frame_id, const cv::Mat &left, const cv::Mat &right)
{
    return Frame::Ptr(new Frame(frame_id, left, right));
}

Eigen::Vector3d Frame::find_world_frame_pos (cv::KeyPoint &kp, Eigen::Vector3d &relative_pt3d)
{
    double x = (kp.pt.x - cx_) / fx_;
    double y = (kp.pt.y - cy_) / fy_;
    double z = fx_ * b_ / disparity_.at<float>(kp.pt.y, kp.pt.x);// + 1e-18);

    relative_pt3d = Eigen::Vector3d(x * z, y * z, z);

    return T_cw_.inverse() * relative_pt3d;
}

void Frame::fill_frame(SE3 T_cw, bool is_keyframe, int keyframe_id)
{
    T_cw_ = T_cw;
    is_keyframe_ = is_keyframe;
    if (is_keyframe)
    {
        keyframe_id_ = keyframe_id;
    }
}

}