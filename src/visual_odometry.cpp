#include "visual_odometry.hpp"
#include "optimizer.hpp"
#include <algorithm>

namespace slam
{

VO::VO(Map &map) : map_(map)
{
    detector_ = cv::ORB::create(3000);
    descriptor_ = cv::ORB::create();
    matcher_ = cv::BFMatcher::create(cv::NORM_HAMMING, true);
}

VO::VO(std::string dataset, Map &map) : dataset_(dataset), map_(map)
{
    detector_ = cv::ORB::create(3000);
    descriptor_ = cv::ORB::create();
    matcher_ = cv::BFMatcher::create(cv::NORM_HAMMING, true);
}

int VO::read_frame(int id, Frame::Ptr frame)
{
    frame->frame_id_ = id;
    std::string id_str = std::to_string(id);
    std::string filename = std::string(6 - id_str.size(), '0') + id_str + ".png";
    frame->left_img_ = cv::imread(dataset_ + "image_0/" + filename, 0);
    frame->right_img_ = cv::imread(dataset_ + "image_1/" + filename, 0);

    if (!frame->left_img_.data)
    {
        std::cout << "Could not open or find the image" << std::endl;
        return -1;
    }
    return 0;
}

int VO::disparity_map(Frame::Ptr frame, cv::Mat &disparity)
{
    // parameters tested for the kitti dataset
    // needs modification if use other dataset
    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(
        0, 96, 9, 8 * 9 * 9, 32 * 9 * 9, 1, 63, 10, 100, 32);

    cv::Mat disparity_sgbm;
    sgbm->compute(frame->left_img_, frame->right_img_, disparity_sgbm);
    disparity_sgbm.convertTo(disparity, CV_32F, 1.0 / 16.0f);
    //cv::imshow("disparity", disparity/ 96.0);
    //cv::waitKey(0);

    return 0;
}

bool VO::init()
{
    frame_last_ = Frame::createFrame();
    read_frame(0, frame_last_);

    std::vector<cv::KeyPoint> keypoints; 
    cv::Mat descriptors;
    std::vector<cv::Point3f> pts_3d;

    feature_detection(frame_last_->left_img_, keypoints, descriptors);
    disparity_map(frame_last_, frame_last_->disparity_);
    std::vector<bool> reliable_depth = get_3d_position(pts_3d, keypoints, descriptors, frame_last_);

    for (int i = 0; i < keypoints.size(); ++i) {
        Feature temp_feature(i, 0, curr_landmark_id_,keypoints[i], descriptors.row(i));
        frame_last_->features_.push_back(temp_feature);

        Observation observation(0, i);
        Landmark temp_landmark(curr_landmark_id_, pts_3d.at(i), descriptors.row(i), reliable_depth.at(i), observation);
        curr_landmark_id_++;

        map_.insert_landmark(temp_landmark);
    }
    frame_last_->fill_frame(SE3(), true, curr_keyframe_id_);
    curr_keyframe_id_++;

    map_.insert_keyframe(frame_last_);

    return true;
}

int VO::feature_detection(const cv::Mat &img, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors)
{
    if (!img.data)
    {
        std::cout << "Could not open or find the image" << std::endl;
        return -1;
    }
    detector_->detect(img, keypoints);
    adaptive_non_maximal_suppresion(keypoints, 600);
    descriptor_->compute(img, keypoints, descriptors);

    cv::Mat outimg1;
    cv::drawKeypoints(img, keypoints, outimg1);
    cv::imshow("ORB features", outimg1);
    cv::waitKey(1);

    return 0;
}

int VO::feature_matching(const cv::Mat &descriptors1, const cv::Mat &descriptors2,
                         std::vector<cv::DMatch> &feature_matches)
{
    feature_matches.clear();
    std::vector<cv::DMatch> matches;
    matcher_->match(descriptors1, descriptors2, matches);

    auto min_max = minmax_element(matches.begin(), matches.end(), [](const cv::DMatch &lhs, 
                                                                     const cv::DMatch &rhs) {
        return lhs.distance < rhs.distance;
    });
    auto min_dist_match = min_max.first;
    //auto max_dist_match = min_max.second;
    double frame_gap = frame_current_->frame_id_ - frame_last_->frame_id_;
    for (int i = 0; i < matches.size(); i++)
    {
        if (matches[i].distance <= std::max(2.0 * min_dist_match->distance, 30.0 * frame_gap))
        {
            feature_matches.push_back(matches[i]);
        }
    }
    return 0;
}

std::vector<bool> VO::get_3d_position(std::vector<cv::Point3f> &pts_3d, 
                                          std::vector<cv::KeyPoint> &keypoints,
                                          cv::Mat &descriptors, 
                                          Frame::Ptr frame)
{
    pts_3d.clear();
    std::vector<cv::KeyPoint> keypoints_filtered;
    cv::Mat descriptors_filtered;
    std::vector<bool> reliable_depth;

    for(int i = 0; i < keypoints.size(); ++i) {
        Eigen::Vector3d world_pos, relative_pos;
        world_pos = frame->find_world_frame_pos(keypoints[i], relative_pos);
        if (relative_pos(2) < 10 || relative_pos(2) > 400) {
            continue;
        }
        keypoints_filtered.push_back(keypoints[i]);
        descriptors_filtered.push_back(descriptors.row(i));
        pts_3d.push_back(cv::Point3f(world_pos(0), world_pos(1), world_pos(2)));
        if (relative_pos(2) < 40)
        {
            reliable_depth.push_back(true);
        }
        else
        {
            reliable_depth.push_back(false);
        }
    }
    keypoints = keypoints_filtered;
    descriptors = descriptors_filtered;
    return reliable_depth;
}

void VO::motion_estimation(Frame::Ptr frame)
{
    std::vector<cv::Point3f> pts3d;
    std::vector<cv::Point2f> pts2d;

    for (int i = 0; i < frame->features_.size(); ++i) {

        int landmark_id = frame->features_[i].landmark_id_;

        if (landmark_id == -1) {
            std::cout << "No lankmark id associated. Skip."<<std::endl;
            continue;
        }
        pts3d.push_back(map_.landmarks_[landmark_id].pt_3d_);
        pts2d.push_back(frame->features_[i].keypoint_.pt);
    }

    //std::cout << "feature num: " << frame.features_.size() << std::endl;

    cv::Mat K = (cv::Mat_<double>(3, 3) << frame->fx_, 0, frame->cx_,
                 0, frame->fy_, frame->cy_,
                 0, 0, 1);
    
    cv::Mat rvec, tvec, inliers;
    cv::solvePnPRansac(pts3d, pts2d, K, cv::Mat(), rvec, tvec, false, 100, 4.0, 0.99, inliers);

    num_inliers_ = inliers.rows;

    cv::Mat R_cv;
    cv::Rodrigues(rvec, R_cv);
    Eigen::Matrix3d R;
    R << R_cv.at<double>(0, 0), R_cv.at<double>(0, 1), R_cv.at<double>(0, 2),
        R_cv.at<double>(1, 0), R_cv.at<double>(1, 1), R_cv.at<double>(1, 2),
        R_cv.at<double>(2, 0), R_cv.at<double>(2, 1), R_cv.at<double>(2, 2);

    T_cw_ = SE3(
        R,
        Eigen::Vector3d(tvec.at<double>(0, 0), tvec.at<double>(1, 0), tvec.at<double>(2, 0)));
    
    
    for (int idx = 0; idx < inliers.rows; idx++)
    {
        int index = inliers.at<int>(idx, 0);
        frame->features_[index].is_inlier = true;
    }

}

bool VO::check_motion_estimation()
{
    if (num_inliers_ < 10)
    {
        std::cout << "Frame id: " << frame_last_->frame_id_ << " and " << frame_current_->frame_id_ << std::endl;
        std::cout << "Rejected - inliers not enough: " << num_inliers_ << std::endl;
        return false;
    }

    // check if the motion is too large
    Sophus::Vector6d displacement = T_cl_.log();
    double frame_gap = frame_current_->frame_id_ - frame_last_->frame_id_; // get the idx gap between last and current frame
    if (displacement.norm() > (5.0 * frame_gap))
    {
        std::cout << "Frame id: " << frame_last_->frame_id_ << " and " << frame_current_->frame_id_ << std::endl;
        std::cout << "Rejected - motion is too large: " << displacement.norm() << std::endl;
        return false;
    }

    return true;
}

void VO::move_frame()
{
    frame_last_ = frame_current_;
}

void VO::adaptive_non_maximal_suppresion(std::vector<cv::KeyPoint> &keypoints,
                                         const int num)
{
    if (keypoints.size() < num)
    {
        return;
    }

    // sort the keypoints according to its reponse (strength)
    std::sort(keypoints.begin(), keypoints.end(), [&](const cv::KeyPoint &lhs, const cv::KeyPoint &rhs) {
        return lhs.response > rhs.response;
    });

    std::vector<cv::KeyPoint> ANMSpt;

    std::vector<double> rad_i;
    rad_i.resize(keypoints.size());

    std::vector<double> rad_i_sorted;
    rad_i_sorted.resize(keypoints.size());

    // robust coefficient: 1/0.9 = 1.1
    const float c_robust = 1.11;

    // computing the suppression radius for each feature (strongest overall has radius of infinity)
    // the smallest distance to another point that is significantly stronger (based on a robustness parameter)
    for (int i = 0; i < keypoints.size(); ++i)
    {
        const float response = keypoints.at(i).response * c_robust;

        // maximum finit number of double
        double radius = std::numeric_limits<double>::max();

        for (int j = 0; j < i && keypoints.at(j).response > response; ++j)
        {
            radius = std::min(radius, cv::norm(keypoints.at(i).pt - keypoints.at(j).pt));
        }

        rad_i.at(i) = radius;
        rad_i_sorted.at(i) = radius;
    }

    // sort it
    std::sort(rad_i_sorted.begin(), rad_i_sorted.end(), [&](const double &lhs, const double &rhs) {
        return lhs > rhs;
    });

    // find the final radius
    const double final_radius = rad_i_sorted.at(num - 1);
    for (int i = 0; i < rad_i.size(); ++i)
    {
        if (rad_i.at(i) >= final_radius)
        {
            ANMSpt.push_back(keypoints.at(i));
        }
    }

    // swap address to keypoints, O(1) time
    keypoints.swap(ANMSpt);
}

bool VO::insert_key_frame(bool check, std::vector<cv::Point3f> &pts_3d, 
                      std::vector<cv::KeyPoint> &keypoints,
                      cv::Mat &descriptors)
{
    if ((num_inliers_ >= 80 && T_cl_.angleY() < 0.03) || check == false)
    {
        return false;
    }

    // fill the extra information
    frame_current_->is_keyframe_ = true;
    frame_current_->keyframe_id_ = curr_keyframe_id_;

    // add observations
    for (int i = 0; i < frame_current_->features_.size(); i++)
    {
        int landmark_id = frame_current_->features_[i].landmark_id_;
        map_.landmarks_[landmark_id].observed_times_++;
        Observation observation(frame_current_->keyframe_id_, frame_current_->features_[i].feature_id_);
        map_.landmarks_[landmark_id].observations_.push_back(observation);

        // std::cout << "Landmark " << landmark_id << " "
        //           << "has obsevation times: " << my_map_.landmarks_.at(landmark_id).observed_times_ << std::endl;
        // std::cout << "Landmark " << landmark_id << " "
        //           << "last observation keyframe: " << my_map_.landmarks_.at(landmark_id).observations_.back().keyframe_id_ << std::endl;
    }

    // add more features with triangulated points to the map
    disparity_map(frame_current_, frame_current_->disparity_);
    std::vector<bool> reliable_depth = get_3d_position(pts_3d, keypoints, descriptors, frame_current_);

    // calculate the world coordinate
    // no relative motion any more
    //std::cout << "Feature num before: " << frame_current_.features_.size() << std::endl; 
    int feature_id = frame_current_->features_.size();
    // if the feature does not exist in the frame already, add it
    for (int i = 0; i < keypoints.size(); i++)
    {
        bool exist = false;
        for (auto &feat : frame_current_->features_)
        {
            if (feat.keypoint_.pt.x == keypoints[i].pt.x && feat.keypoint_.pt.y == keypoints[i].pt.y)
            {
                exist = true;

                // try to update the landmark position if already exist
                if ((map_.landmarks_[feat.landmark_id_].reliable_depth_ == false) && reliable_depth[i])
                {
                    map_.landmarks_[feat.landmark_id_].pt_3d_ = pts_3d[i];
                    map_.landmarks_[feat.landmark_id_].reliable_depth_ = true;
                }
            }
        }
        if (!exist)
        {
            // add this feature
            // put the features into the frame with feature_id, frame_id, keypoint, descriptor
            // build the connection from feature to frame
            Feature feature_to_add(feature_id, frame_current_->frame_id_,
                                   keypoints[i], descriptors.row(i));

            // build the connection from feature to landmark
            feature_to_add.landmark_id_ = curr_landmark_id_;
            frame_current_->features_.push_back(feature_to_add);
            // create a landmark
            // build the connection from landmark to feature
            Observation observation(frame_current_->keyframe_id_, feature_id);
            Landmark landmark_to_add(curr_landmark_id_, pts_3d[i], descriptors.row(i), reliable_depth[i], observation);
            curr_landmark_id_++;
            // insert the landmark
            map_.insert_landmark(landmark_to_add);
            feature_id++;
        }
        
    }
    //std::cout << "Feature num after: " << frame_current_.features_.size() << std::endl;
    curr_keyframe_id_++;

    // insert the keyframe
    map_.insert_keyframe(frame_current_);

    // std::cout << "Set frame: " << frame_current_.frame_id_ << " as keyframe "
    //           << frame_current_.keyframe_id_ << std::endl;

    return true;
}

bool VO::tracking(bool &if_insert_keyframe)
{
    frame_current_ = Frame::createFrame();

    // get optimized last frame from the map
    if (frame_last_->is_keyframe_ == true)
    {
        frame_last_ = map_.keyframes_[frame_last_->keyframe_id_];
    }

    read_frame(seq_, frame_current_);

    std::vector<cv::KeyPoint> keypoints_current, keypoints_last;
    cv::Mat descriptors_current, descriptors_last;
    std::vector<cv::DMatch> matches;

    feature_detection(frame_current_->left_img_, keypoints_current, descriptors_current);
    //<< "kpts num: " << keypoints_current.size() << std::endl;

    //std::cout<<"frame_last_ feature num: "<<frame_last_.features_.size()<<std::endl;

    for(int i = 0; i < frame_last_->features_.size(); ++i)
    {
        keypoints_last.push_back(frame_last_->features_[i].keypoint_);
        descriptors_last.push_back(frame_last_->features_[i].descriptor_);
    }
    
    feature_matching(descriptors_last, descriptors_current, matches);

    for (int i = 0; i < matches.size(); ++i)
    {
        Feature feature = Feature(i, seq_, 
                                  keypoints_current[matches[i].trainIdx],
                                  descriptors_current.row(matches[i].trainIdx));
        feature.landmark_id_ = frame_last_->features_[matches[i].queryIdx].landmark_id_;
        frame_current_->features_.push_back(feature);
    }

    motion_estimation(frame_current_);
    frame_current_->T_cw_ = T_cw_;

    if (num_inliers_ >= 10)
    {
        optimize_between_frame(frame_current_, frame_last_, matches, map_.landmarks_, 10);
    }

    
    frame_current_->features_.erase(std::remove_if(
                              frame_current_->features_.begin(), frame_current_->features_.end(),
                              [](const Feature &x) {
                                  return x.is_inlier == false;
                              }),
                          frame_current_->features_.end());

    

    T_cl_ = frame_current_->T_cw_ * frame_last_->T_cw_.inverse();

    bool check = check_motion_estimation();
    //bool check = true;
    
    
    std::vector<cv::Point3f> pts_3d;
    if_insert_keyframe = insert_key_frame(check, pts_3d, keypoints_current, descriptors_current);
    //std::cout<<"move?: " << check << std::endl;

    if (check)
    {
        move_frame();
    }

    seq_++;
    return check;
}

bool VO::pipeline(bool &if_insert_keyframe)
{
    switch (state_)
    {
    case Init:
    {
        if (init())
        {
            state_ = Track;
        }
        else
        {
            num_lost_++;
            if (num_lost_ > 10)
            {
                state_ = Lost;
            }
        }
        break;
    }

    case Track:
    {
        if (tracking(if_insert_keyframe))
        {
            num_lost_ = 0;
        }
        else
        {
            num_lost_++;
            if (num_lost_ > 10)
            {
                state_ = Lost;
            }
        }
        break;
    }
    
    case Lost:
    {
        std::cout << "VO IS LOST" << std::endl;
        return false;
        break;
    }
    default:
        std::cout << "Invalid state" << std::endl;
        return false;
        break;
    }

    return true;
}

}