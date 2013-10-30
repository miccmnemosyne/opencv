/*
 * softcascade_plus.hpp
 *
 *  Created on: Oct 29, 2013
 *      Author: federico
 */

#ifndef SOFTCASCADE_FAST_HPP_
#define SOFTCASCADE_FAST_HPP_

#include "softcascade.hpp"

namespace cv { namespace softcascade {


class CV_EXPORTS_W DetectorFast: public Detector{

public:
    // An empty cascade will be created.
    // Param minScale is a minimum scale relative to the original size of the image on which cascade will be applied.
    // Param minScale is a maximum scale relative to the original size of the image on which cascade will be applied.
    // Param scales is a number of scales from minScale to maxScale.
    // Param rejCriteria is used for NMS.
    CV_WRAP DetectorFast(double minScale = 0.4, double maxScale = 5., int scales = 55, int rejCriteria = 1);

    CV_WRAP ~DetectorFast();
};

}}


#endif /* SOFTCASCADE_PLUS_HPP_ */
