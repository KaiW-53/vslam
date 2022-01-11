#include "visual_odometry.hpp"
#include "map.hpp"
#include "optimizer.hpp"


int main() {
    std::string data_path = "/home/kaiwen/deep_blue/slam/data/data_odometry_gray/dataset/sequences/00/";
    slam::Map map = slam::Map();
    slam::VO vo = slam::VO(data_path, map);

    double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
    double b = 0.573;
    cv::Mat K = (cv::Mat_<double>(3, 3) << fx, 0, cx,
                 0, fy, cy,
                 0, 0, 1);

    bool save_txt = true;

    for (int ite = 0; ite < 4541; ite++)
    {
        //std::cout << ite << std::endl;
        bool not_lost = true;
        bool if_insert_keyframe = false;
        not_lost = vo.pipeline(if_insert_keyframe);

        if (if_insert_keyframe && map.keyframes_.size() >= 10)
        {
            // reject the outliers
            slam::optimize_map(map.keyframes_, map.landmarks_, K, false, false, 5);
            slam::optimize_map(map.keyframes_, map.landmarks_, K, false, false, 5);
            // do the optimization
            slam::optimize_map(map.keyframes_, map.landmarks_, K, true, false, 10);
            slam::optimize_pose_only(map.keyframes_, map.landmarks_, K, true, 10);
        }

        if(!not_lost) {
            break;
        }
    }

    for (const auto &f : map.keyframes_) {
        map.write_to_file(f.second);
    }

    return 0;
    
}