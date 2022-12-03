/**
* BSD 3-Clause License
* Copyright (c) 2021, The Trustees of Princeton University. All rights reserved.
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are
* met:
*     1. Redistributions of source code must retain the above copyrigh
*        notice, this list of conditions and the following disclaimer.
*     2. Redistributions in binary form must reproduce the above
*        copyright notice, this list of conditions and the following
*        disclaimer in the documentation and/or other materials provided
*        with the distribution
*     3. Neither the name of the copyright holder nor the names of its
*        contributors may be used to endorse or promote products derived
*        from this software without specific prior written permission.
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, TH
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE US
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

* Please contact the author(s) of this library if you have any questions.
* Authors:    Zixu Zhang       ( zixuz@princeton.edu )

 **/

#include "frontend/tag_detector.h"

namespace tagslam_ros
{   
    void TagDetector::drawDetections(cv_bridge::CvImagePtr image,
            TagDetectionArrayPtr tag_detection)
    {
        drawDetections(image->image, tag_detection);
    }

    void TagDetector::drawDetections(cv::Mat & image,
            TagDetectionArrayPtr tag_detection)
    {
        for (auto & det : tag_detection->detections)
        {

        // Draw tag outline with edge colors green, blue, blue, red
        // (going counter-clockwise, starting from lower-left corner in
        // tag coords). cv::Scalar(Blue, Green, Red) format for the edge
        // colors! 

        cv::Point p1((int)det.corners[0].x, (int)det.corners[0].y);
        cv::Point p2((int)det.corners[1].x, (int)det.corners[1].y);
        cv::Point p3((int)det.corners[2].x, (int)det.corners[2].y);
        cv::Point p4((int)det.corners[3].x, (int)det.corners[3].y);

        line(image, p1, p2, cv::Scalar(0,0xff,0)); // green
        line(image, p2, p3, cv::Scalar(0,0,0xff));
        line(image, p3, p4, cv::Scalar(0xff,0,0)); // green
        line(image, p4, p1, cv::Scalar(0xff,0,0)); // green
        
        // Print tag ID in the middle of the tag
        std::stringstream ss;
        ss << det.id;
        cv::String text = ss.str();
        int fontface = cv::FONT_HERSHEY_SCRIPT_SIMPLEX;
        double fontscale = 0.5;
        int baseline;
        cv::Size textsize = cv::getTextSize(text, fontface,
                                            fontscale, 2, &baseline);
        cv::Point center((int)det.center.x, (int)det.center.y);
        cv::putText(image, text, center,
                    fontface, fontscale, cv::Scalar(0xff, 0x99, 0), 2);
        }
    }
}