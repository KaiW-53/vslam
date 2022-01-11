#include "map.hpp"  

namespace slam
{

int Map::insert_keyframe(Frame::Ptr frame_to_add)
{
    current_keyframe_id_ = frame_to_add->keyframe_id_;
    keyframes_[current_keyframe_id_] = frame_to_add;

    if(keyframes_.size() > num_keyframes_) {
        remove_keyframe();
    }

    return 0;
}

int Map::insert_landmark(Landmark landmark_to_add)
{
    landmarks_[landmark_to_add.landmark_id_] = landmark_to_add;
    
    return 0;
}

int Map::remove_keyframe()
{
    double max_distance = 0, min_distance = 1000000;
    double max_keyframe_id = 0, min_keyframe_id = 0;
    SE3 T_wc = keyframes_[current_keyframe_id_]->T_cw_.inverse();

    for (auto &kf : keyframes_)
    {
        if (kf.first == current_keyframe_id_)
        {
            continue;
        }
        double distance = (kf.second->T_cw_ * T_wc).log().norm();
        if (distance > max_distance)
        {
            max_distance = distance;
            max_keyframe_id = kf.first;
        }
        if (distance < min_distance)
        {
            min_distance = distance;
            min_keyframe_id = kf.first;
        }
    }

    int keyframe_to_remove_id = -1;
    if (min_distance < min_threshold)
    {
        // remove the closest if distance is lower than 0.2
        keyframe_to_remove_id = min_keyframe_id;
        // std::cout << "Removed closest frame" << std::endl;
    }
    else
    {
        keyframe_to_remove_id = max_keyframe_id;
    }

    for (auto feat : keyframes_[keyframe_to_remove_id]->features_)
    {
        // mark the observations that need to delete
        for (int i = 0; i < landmarks_[feat.landmark_id_].observations_.size(); i++)
        {
            if (landmarks_[feat.landmark_id_].observations_[i].keyframe_id_ == keyframe_to_remove_id &&
                landmarks_[feat.landmark_id_].observations_[i].feature_id_ == feat.feature_id_)
            {
                landmarks_[feat.landmark_id_].observations_[i].to_delete = true;
            }
        }
        // delete them
        landmarks_.at(feat.landmark_id_).observations_.erase
                   (std::remove_if(landmarks_.at(feat.landmark_id_).observations_.begin(), 
                                   landmarks_.at(feat.landmark_id_).observations_.end(), 
                                   [](const Observation &x) 
                                   {
                                        return x.to_delete == true;
                                    }),
                    landmarks_.at(feat.landmark_id_).observations_.end());
        // observation times minus one
        landmarks_.at(feat.landmark_id_).observed_times_--;
    }
    if(save_txt)
    {
        write_to_file(keyframes_[keyframe_to_remove_id]);
    }
    keyframes_.erase(keyframe_to_remove_id);
    // std::cout << "number of keyframes after: " << keyframes_.size() << std::endl;

    clean_map();

    return 0;
}

int Map::clean_map()
{
    int num_landmark_removed = 0;
    for (auto iter = landmarks_.begin();
         iter != landmarks_.end();)
    {
        if (iter->second.observed_times_ <= 0)
        {
            iter = landmarks_.erase(iter);
            num_landmark_removed++;
        }
        else
        {
            ++iter;
        }
    }

    return 0;
}

void Map::write_to_file(Frame::Ptr frame)
{
    SE3 T_wc;
    T_wc = frame->T_cw_.inverse();
    double r00, r01, r02, r10, r11, r12, r20, r21, r22, x, y, z;
    Eigen::Matrix3d rotation = T_wc.rotationMatrix();
    Eigen::Vector3d translation = T_wc.translation();
    r00 = rotation(0, 0);
    r01 = rotation(0, 1);
    r02 = rotation(0, 2);
    r10 = rotation(1, 0);
    r11 = rotation(1, 1);
    r12 = rotation(1, 2);
    r20 = rotation(2, 0);
    r21 = rotation(2, 1);
    r22 = rotation(2, 2);
    x = translation(0);
    y = translation(1);
    z = translation(2);

    std::ofstream file;
    file.open("traj_between_opti5.txt", std::ios_base::app);

    // alows dropping frame
    /*file << frame.frame_id_ << " " << r00 << " " << r01 << " " << r02 << " " << x << " "
         << r10 << " " << r11 << " " << r12 << " " << y << " "
         << r20 << " " << r21 << " " << r22 << " " << z << std::endl;*/
    
    file << frame->frame_id_ << " " << x << " " << z << std::endl;
    file.close();
}

}