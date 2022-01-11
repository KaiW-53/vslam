#pragma once

#include "data_structure.hpp"
#include "frame.hpp"

namespace slam
{

struct Map
{
public:
    std::unordered_map<unsigned long, Frame::Ptr> keyframes_;
    std::unordered_map<unsigned long, Landmark> landmarks_;

    // number of active keyframes
    const int num_keyframes_ = 10;

    // minimum distance interval for keyrframe
    const double min_threshold = 0.2;

    // id of the current keyframe
    int current_keyframe_id_ = 0;

    bool save_txt = true;
    

public:
    Map() = default;

    /*! \brief inset a keyframe into the map
    *
    *  \param frame_to_add - the frame to be added
    *  \return if successful
    */
    int insert_keyframe(Frame::Ptr frame_to_add);

    /*! \brief inset a landmark into the map
    *
    *  \param landmark_to_add - the landmark to be added
    *  \return if successful
    */
    int insert_landmark(Landmark landmark_to_add);

    /*! \brief remove a keyframe from the map
    *
    *  \return if successful
    */
    int remove_keyframe();

    /*! \brief clean the map: remove landmarks with no observations
    *
    *  \return if successful
    */
    int clean_map();

    void write_to_file(Frame::Ptr frame);
    
};

}