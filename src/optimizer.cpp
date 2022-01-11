#include "optimizer.hpp"

namespace slam
{
void VertexPose::oplusImpl(const double *update)
{
    Eigen::Matrix<double, 6, 1> update_se3;
    update_se3 << update[0], update[1], update[2],
                update[3], update[4], update[5];
    _estimate = Sophus::SE3d::exp(update_se3) * _estimate;
}

void VertexLandmark::oplusImpl(const double *update)
{
    _estimate[0] += update[0];
    _estimate[1] += update[1];
    _estimate[2] += update[2];
}

void EdgeProjection::computeError()
{
    const VertexPose *v0 = static_cast<VertexPose *>(_vertices[0]);
    const VertexLandmark *v1 = static_cast<VertexLandmark *>(_vertices[1]);
    Sophus::SE3d T = v0->estimate();
    Eigen::Vector3d pixel = K_ * (T * v1->estimate());
    pixel /= pixel[2];
    _error = _measurement - pixel.head<2>(); 
}

void EdgeProjection::linearizeOplus()
{
    const VertexPose *v0 = static_cast<VertexPose *>(_vertices[0]);
    const VertexLandmark *v1 = static_cast<VertexLandmark *>(_vertices[1]);
    Sophus::SE3d T = v0->estimate();
    Eigen::Vector3d pw = v1->estimate();
    Eigen::Vector3d pos_cam = T * pw;
    double fx = K_(0, 0);
    double fy = K_(1, 1);
    double cx = K_(0, 2);
    double cy = K_(1, 2);
    double X = pos_cam[0];
    double Y = pos_cam[1];
    double Z = pos_cam[2];
    double Zinv = 1.0 / (Z + 1e-18);
    double Zinv2 = Zinv * Zinv;
    _jacobianOplusXi << -fx * Zinv, 0, fx * X * Zinv2, fx * X * Y * Zinv2,
        -fx - fx * X * X * Zinv2, fx * Y * Zinv, 0, -fy * Zinv,
        fy * Y * Zinv2, fy + fy * Y * Y * Zinv2, -fy * X * Y * Zinv2,
        -fy * X * Zinv;
    _jacobianOplusXj = _jacobianOplusXi.block<2, 3>(0, 0) * T.rotationMatrix();
}

void optimize_map(std::unordered_map<unsigned long, Frame::Ptr> &keyframes,
                  std::unordered_map<unsigned long, Landmark> &landmarks,
                  const cv::Mat &K, bool if_update_map, bool if_update_landmark, int num_ite)
{
    using BlockSolverType = g2o::BlockSolverPL<6, 3>;
    using LinearSolverType = g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType>;

    auto linearSolver = g2o::make_unique<LinearSolverType>();
    auto solver_ptr = g2o::make_unique<BlockSolverType>(std::move(linearSolver));
    auto solver = new g2o::OptimizationAlgorithmLevenberg(std::move(solver_ptr));

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    std::map<unsigned long, VertexPose *> vertices;
    unsigned long max_kf_id = 0;

    for (auto &keyframe : keyframes)
    {
        auto kf = keyframe.second;
        VertexPose *vertex_pose = new VertexPose();
        vertex_pose->setId(kf->keyframe_id_);
        vertex_pose->setEstimate(kf->T_cw_);
        optimizer.addVertex(vertex_pose);
        if (kf->keyframe_id_ > max_kf_id)
        {
            max_kf_id = kf->keyframe_id_;
        }

        vertices.insert({kf->keyframe_id_, vertex_pose});
    }

    std::map<unsigned long, VertexLandmark *> vertices_landmarks;

    Eigen::Matrix3d K_eigen;
    K_eigen << K.at<double>(0, 0), K.at<double>(0, 1), K.at<double>(0, 2),
        K.at<double>(1, 0), K.at<double>(1, 1), K.at<double>(1, 2),
        K.at<double>(2, 0), K.at<double>(2, 1), K.at<double>(2, 2);

    // add edges
    int index = 1;

    // set robust kernel threshold
    double chi2_th = 5.991;

    std::map<EdgeProjection *, Feature> edges_and_features;

    for (auto &landmark : landmarks)
    {
        if (!landmark.second.is_inlier || !landmark.second.reliable_depth_ )
        {
            continue;
        }

        unsigned long landmark_id = landmark.second.landmark_id_;
        std::vector<Observation> observations = landmark.second.observations_;
        for (Observation &obs : observations)
        {

            Feature feat = keyframes[obs.keyframe_id_]->features_.at(obs.feature_id_);

            // create a new edge
            EdgeProjection *edge = new EdgeProjection(K_eigen);

            // add the landmark as vertices
            if (vertices_landmarks.find(landmark_id) ==
                vertices_landmarks.end())
            {
                VertexLandmark *v = new VertexLandmark;
                v->setEstimate(landmark.second.to_vector_3d());
                // set the id following the maximum id of keyframes
                v->setId(landmark_id + max_kf_id + 1);
                v->setMarginalized(true);
                vertices_landmarks.insert({landmark_id, v});
                optimizer.addVertex(v);
            }

            edge->setId(index);

            edge->setVertex(0, vertices.at(obs.keyframe_id_));      // pose
            edge->setVertex(1, vertices_landmarks.at(landmark_id)); // landmark

            // convert cv point to eigen
            Eigen::Matrix<double, 2, 1> measurement(feat.keypoint_.pt.x, feat.keypoint_.pt.y);
            edge->setMeasurement(measurement);
            edge->setInformation(Eigen::Matrix<double, 2, 2>::Identity());

            // set the robust kernel
            auto rk = new g2o::RobustKernelHuber();
            rk->setDelta(chi2_th);
            edge->setRobustKernel(rk);

            edges_and_features.insert({edge, feat});

            optimizer.addEdge(edge);

            index++;
        }
    }

    optimizer.initializeOptimization();
    optimizer.optimize(num_ite);

    int cnt_outlier = 0, cnt_inlier = 0;
    int iteration = 0;
    while (iteration < 5)
    {
        cnt_outlier = 0;
        cnt_inlier = 0;
        // determine if we want to adjust the outlier threshold
        for (auto &ef : edges_and_features)
        {
            if (ef.first->chi2() > chi2_th)
            {
                cnt_outlier++;
            }
            else
            {
                cnt_inlier++;
            }
        }
        double inlier_ratio = cnt_inlier / double(cnt_inlier + cnt_outlier);
        if (inlier_ratio > 0.5)
        {
            break;
        }
        else
        {
            chi2_th *= 2;
            iteration++;
        }
    }

    for (auto &ef : edges_and_features)
    {
        if (ef.first->chi2() > chi2_th)
        {
            int landmark_id_to_update = ef.second.landmark_id_;
            landmarks.at(landmark_id_to_update).is_inlier = false;
        }
        else
        {
            int landmark_id_to_update = ef.second.landmark_id_;
            landmarks.at(landmark_id_to_update).is_inlier = true;
        }
    }

    if (if_update_map)
    {

        for (auto &v : vertices)
        {
            keyframes.at(v.first)->T_cw_ = v.second->estimate();
        }
        if (if_update_landmark)
        {
            for (auto &v : vertices_landmarks)
            {
                Eigen::Vector3d modified_pos_3d = v.second->estimate();
                landmarks.at(v.first).pt_3d_ = cv::Point3f(modified_pos_3d(0), modified_pos_3d(1), modified_pos_3d(2));
            }
        }
    }
}

void PoseOnlyEdgeProjection::computeError()
{
    const VertexPose *v = static_cast<VertexPose *>(_vertices[0]);
    Sophus::SE3d T = v->estimate();
    Eigen::Vector3d pos_pixel = K_ * (T * point_3d_);
    pos_pixel /= pos_pixel[2];
    _error = _measurement - pos_pixel.head<2>();
}

void PoseOnlyEdgeProjection::linearizeOplus()
{
    const VertexPose *v = static_cast<VertexPose *>(_vertices[0]);
    Sophus::SE3d T = v->estimate();
    Eigen::Vector3d pos_cam = T * point_3d_;
    double fx = K_(0, 0);
    double fy = K_(1, 1);
    double cx = K_(0, 2);
    double cy = K_(1, 2);
    double X = pos_cam[0];
    double Y = pos_cam[1];
    double Z = pos_cam[2];
    double Z2 = Z * Z;
    _jacobianOplusXi
        << -fx / Z,
        0, fx * X / Z2, fx * X * Y / Z2, -fx - fx * X * X / Z2, fx * Y / Z,
        0, -fy / Z, fy * Y / (Z * Z), fy + fy * Y * Y / Z2, -fy * X * Y / Z2, -fy * X / Z;
}

void optimize_pose_only(std::unordered_map<unsigned long, Frame::Ptr> &keyframes,
                        std::unordered_map<unsigned long, Landmark> &landmarks,
                        const cv::Mat &K, bool if_update_map, int num_ite)
{
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType;
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;

    // Use GaussNewton/Levenberg Method
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    // optimizer.setVerbose(true);

    // add poses as vertices
    std::map<unsigned long, VertexPose *> vertices;

    for (auto &keyframe : keyframes)
    {
        auto kf = keyframe.second;
        VertexPose *vertex_pose = new VertexPose();
        vertex_pose->setId(kf->keyframe_id_);
        vertex_pose->setEstimate(kf->T_cw_);
        optimizer.addVertex(vertex_pose);

        vertices.insert({kf->keyframe_id_, vertex_pose});
    }

    // K: convert c:mat to eigen matrix
    Eigen::Matrix3d K_eigen;
    K_eigen << K.at<double>(0, 0), K.at<double>(0, 1), K.at<double>(0, 2),
        K.at<double>(1, 0), K.at<double>(1, 1), K.at<double>(1, 2),
        K.at<double>(2, 0), K.at<double>(2, 1), K.at<double>(2, 2);

    // add edges
    int index = 1;

    // set robust kernel threshold
    double chi2_th = 5.991;

    std::map<PoseOnlyEdgeProjection *, Feature> edges_and_features;

    for (auto &landmark : landmarks)
    {
        if (landmark.second.is_inlier == false)
        {
            continue;
        }

        unsigned long landmark_id = landmark.second.landmark_id_;
        std::vector<Observation> observations = landmark.second.observations_;
        for (Observation &obs : observations)
        {

            Feature feat = keyframes[obs.keyframe_id_]->features_.at(obs.feature_id_);

            // if it is not an inlier, do not add it to optimization
            // if (feat.is_inlier == false)
            //     continue;

            // create a new edge
            PoseOnlyEdgeProjection *edge = nullptr;
            edge = new PoseOnlyEdgeProjection(landmark.second.to_vector_3d(), K_eigen);

            edge->setId(index);
            edge->setVertex(0, vertices.at(obs.keyframe_id_));

            // convert cv point to eigen
            Eigen::Matrix<double, 2, 1> measurement(feat.keypoint_.pt.x, feat.keypoint_.pt.y);
            edge->setMeasurement(measurement);
            edge->setInformation(Eigen::Matrix<double, 2, 2>::Identity());

            // set the robust kernel
            auto rk = new g2o::RobustKernelHuber();
            rk->setDelta(chi2_th);
            edge->setRobustKernel(rk);

            edges_and_features.insert({edge, feat});

            optimizer.addEdge(edge);

            index++;
        }
    }

    // do optimization
    optimizer.initializeOptimization();
    optimizer.optimize(num_ite);

    // print the first pose optimized
    // optimizer.setVerbose(true);

    int cnt_outlier = 0, cnt_inlier = 0;
    int iteration = 0;
    while (iteration < 5)
    {
        cnt_outlier = 0;
        cnt_inlier = 0;
        // determine if we want to adjust the outlier threshold
        for (auto &ef : edges_and_features)
        {
            if (ef.first->chi2() > chi2_th)
            {
                cnt_outlier++;
            }
            else
            {
                cnt_inlier++;
            }
        }
        double inlier_ratio = cnt_inlier / double(cnt_inlier + cnt_outlier);
        if (inlier_ratio > 0.5)
        {
            break;
        }
        else
        {
            chi2_th *= 2;
            iteration++;
        }
    }

    for (auto &ef : edges_and_features)
    {
        if (ef.first->chi2() > chi2_th)
        {
            int landmark_id_to_update = ef.second.landmark_id_;
            landmarks.at(landmark_id_to_update).is_inlier = false;
        }
        else
        {
            int landmark_id_to_update = ef.second.landmark_id_;
            landmarks.at(landmark_id_to_update).is_inlier = true;
        }
    }

    // std::cout << "Outlier/Inlier in optimization: " << cnt_outlier << "/"
    //           << cnt_inlier << " at threshold " << chi2_th << std::endl;

    if (if_update_map)
    {
        for (auto &v : vertices)
        {
            keyframes.at(v.first)->T_cw_ = v.second->estimate();
        }
    }
}

void optimize_pose_between_frame(Frame::Ptr current, 
                            Frame::Ptr last, 
                            std::vector<cv::DMatch> &matches,
                            std::unordered_map<unsigned long, Landmark> &landmarks,
                            int num_ite)
{
    Eigen::Matrix3d K;
    K<<last->fx_, 0, last->cx_,
       0, last->fy_, last->cy_,
       0, 0, 1;
    std::vector<cv::Point3f> pts3d;
    std::vector<cv::Point2f> pts2d_curr;
    std::vector<cv::Point2f> pts2d_last;

    for (int i = 0; i < current->features_.size(); ++i) {

        if (!current->features_[i].is_inlier) continue;

        int landmark_id = current->features_[i].landmark_id_;

        pts3d.push_back(landmarks[landmark_id].pt_3d_);
        pts2d_curr.push_back(current->features_[i].keypoint_.pt);
        pts2d_last.push_back(last->features_[matches[i].queryIdx].keypoint_.pt);
    }

    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType;
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;

    // Use GaussNewton/Levenberg Method
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    VertexPose *vertex_pose1 = new VertexPose();
    vertex_pose1->setId(0);
    vertex_pose1->setFixed(true);
    vertex_pose1->setEstimate(last->T_cw_);
    optimizer.addVertex(vertex_pose1);

    VertexPose *vertex_pose2 = new VertexPose();
    vertex_pose2->setId(1);
    vertex_pose2->setEstimate(current->T_cw_);
    optimizer.addVertex(vertex_pose2);

    for (int i = 0; i < pts3d.size(); i++) 
    {
        Eigen::Vector3d pos_vec;
        pos_vec << pts3d[i].x, pts3d[i].y, pts3d[i].z; 

        PoseOnlyEdgeProjection *edge1 = nullptr;
        edge1 = new PoseOnlyEdgeProjection(pos_vec, K);
        edge1->setId(i * 2);
        edge1->setVertex(0, vertex_pose1);
        Eigen::Matrix<double, 2, 1> measurement1(pts2d_last[i].x, pts2d_last[i].y);
        edge1->setMeasurement(measurement1);
        edge1->setInformation(Eigen::Matrix<double, 2, 2>::Identity());

        PoseOnlyEdgeProjection *edge2 = nullptr;
        edge2 = new PoseOnlyEdgeProjection(pos_vec, K);
        edge2->setId(i * 2 + 1);
        edge2->setVertex(0, vertex_pose2);
        Eigen::Matrix<double, 2, 1> measurement2(pts2d_curr[i].x, pts2d_curr[i].y);
        edge2->setMeasurement(measurement2);
        edge2->setInformation(Eigen::Matrix<double, 2, 2>::Identity());

        optimizer.addEdge(edge1);
        optimizer.addEdge(edge2);
    }

    optimizer.initializeOptimization();
    optimizer.optimize(num_ite);

    current->T_cw_ = vertex_pose2->estimate();
}

void optimize_between_frame(Frame::Ptr current, 
                            Frame::Ptr last, 
                            std::vector<cv::DMatch> &matches,
                            std::unordered_map<unsigned long, Landmark> &landmarks,
                            int num_ite)
{
    Eigen::Matrix3d K;
    K<<last->fx_, 0, last->cx_,
       0, last->fy_, last->cy_,
       0, 0, 1;
    std::vector<cv::Point3f> pts3d;
    std::vector<cv::Point2f> pts2d_curr;
    std::vector<cv::Point2f> pts2d_last;
    std::vector<int> landmark_ids;

    for (int i = 0; i < current->features_.size(); ++i) {

        if (!current->features_[i].is_inlier) continue;

        int landmark_id = current->features_[i].landmark_id_;

        pts3d.push_back(landmarks[landmark_id].pt_3d_);
        pts2d_curr.push_back(current->features_[i].keypoint_.pt);
        pts2d_last.push_back(last->features_[matches[i].queryIdx].keypoint_.pt);
        landmark_ids.push_back(landmark_id);
    }

    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType;
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;

    // Use GaussNewton/Levenberg Method
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    VertexPose *vertex_pose1 = new VertexPose();
    vertex_pose1->setId(0);
    vertex_pose1->setFixed(true);
    vertex_pose1->setEstimate(last->T_cw_);
    optimizer.addVertex(vertex_pose1);

    VertexPose *vertex_pose2 = new VertexPose();
    vertex_pose2->setId(1);
    vertex_pose2->setEstimate(current->T_cw_);
    optimizer.addVertex(vertex_pose2);

    std::vector<VertexLandmark *> vertices_landmarks;
    int idx = 3;

    for (int i = 0; i < pts3d.size(); i++) 
    {
        Eigen::Vector3d pos_vec;
        pos_vec << pts3d[i].x, pts3d[i].y, pts3d[i].z;

        VertexLandmark *v = new VertexLandmark;
        v->setEstimate(pos_vec);
        v->setId(idx);
        v->setMarginalized(true);
        vertices_landmarks.push_back(v);
        optimizer.addVertex(v); 
        ++idx;

        EdgeProjection *edge1 = new EdgeProjection(K);
        edge1->setId(i * 2);
        edge1->setVertex(0, vertex_pose1);
        edge1->setVertex(1, v);
        Eigen::Matrix<double, 2, 1> measurement1(pts2d_last[i].x, pts2d_last[i].y);
        edge1->setMeasurement(measurement1);
        edge1->setInformation(Eigen::Matrix<double, 2, 2>::Identity());

        EdgeProjection *edge2 = new EdgeProjection(K);
        edge2->setId(i * 2 + 1);
        edge2->setVertex(0, vertex_pose2);
        edge2->setVertex(1, v);
        Eigen::Matrix<double, 2, 1> measurement2(pts2d_curr[i].x, pts2d_curr[i].y);
        edge2->setMeasurement(measurement2);
        edge2->setInformation(Eigen::Matrix<double, 2, 2>::Identity());

        optimizer.addEdge(edge1);
        optimizer.addEdge(edge2);
    }

    optimizer.initializeOptimization();
    optimizer.optimize(num_ite);

    current->T_cw_ = vertex_pose2->estimate();

    for (int i = 0; i < landmark_ids.size(); ++i)
    {
        Eigen::Vector3d modified_pos_3d = vertices_landmarks[i]->estimate();
        landmarks[landmark_ids[i]].pt_3d_ = cv::Point3f(modified_pos_3d(0), modified_pos_3d(1), modified_pos_3d(2));
    }
}

}

