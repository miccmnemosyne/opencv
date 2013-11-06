/*
 * softcascade_plus.hpp
 *
 *  Created on: Oct 29, 2013
 *      Author: federico
 */

#ifndef SOFTCASCADE_FAST_HPP_
#define SOFTCASCADE_FAST_HPP_

#include "softcascade.hpp"


struct Level;
struct ChannelStorage;

namespace cv { namespace softcascade {


// ============================================================================================================== //
//					     Declaration of DetectorFast (with trace evaluation reduction)
// ============================================================================================================= //

class CV_EXPORTS_W DetectorFast: public Detector{

public:

	// An empty cascade will be created.
    // Param minScale 		is a minimum scale relative to the original size of the image on which cascade will be applied.
    // Param minScale 		is a maximum scale relative to the original size of the image on which cascade will be applied.
    // Param scales 		is a number of scales from minScale to maxScale.
    // Param rejCriteria 	is used for NMS.
    CV_WRAP DetectorFast(double minScale = 0.4, double maxScale = 5., int scales = 55, int rejCriteria = 1,uint maxNumStages=1024/2);

    CV_WRAP ~DetectorFast();

    // Load soft cascade from FileNode and trace-model.
    // Param fileNode 		is a root node for cascade.
    // Param fileNodeModel 	is a root node for trace-model.
    CV_WRAP virtual bool load(const FileNode& fileNode,const FileNode& fileNodeModel);


    // Return the vector of Detection objects (with fast evaluation).
    // Param image is a frame on which detector will be applied.
    // Param rois is a vector of regions of interest. Only the objects that fall into one of the regions will be returned.
    // Param objects is an output array of Detections
    virtual void detectFast(cv::InputArray _image,std::vector<Detection>& objects);



private:

    // Load trace-model
    CV_WRAP virtual bool loadModel(const FileNode& fileNodeModel);


    // Last Stage evaluated, after is estimated the final score through trace-model
    uint	lastStage;

	struct CV_EXPORTS TempInfo{
    	int rejCriteria;
    	int*	index;
    	int*	weaks;
    };

	TempInfo tempI;

};


// ============================================================================================================== //
//		     Declaration of DetectorTrace (without trace evaluation reduction) and other structures nedded
// ============================================================================================================= //

// Representation of detector results, included subscores and pyramid info
struct CV_EXPORTS Trace{

	// Creates Trace from an object bounding box, confidence and subscores.
	// Param index 			is identifier for each trace inner the pyramid
	// Param localMaxIndex  is identifier of local maximum trace that reject me (only if classification==POSITIVE)
	// Param octaveIndex	is index of octave (inner of pyramid) that contain the positive detection window (the enumeration starts with 0)
	// Param numLevel  		is number of level  (inner of pyramid) that contain the positive detection window
	// Param dw				is detection window
	// Param subScores		is detector result (positive detection window)
	// Param classification is detector result (positive detection window)
	Trace(const uint64 ind,const uint octave, const uint level, const Detection& dw, const std::vector<float>& scores, const int classification);

	enum{NEGATIVE=0,POSITIVE,LOCALMAXIMUM};
	uint64 	index;
	uint64 	localMaxIndex;
	uint 	octaveIndex;
	uint 	numLevel;
	Detection detection;
	std::vector<float> subscores;
	int classType;
};


class CV_EXPORTS_W DetectorTrace: public Detector{

public:

	// An empty cascade will be created.
    // Param minScale 		is a minimum scale relative to the original size of the image on which cascade will be applied.
    // Param minScale 		is a maximum scale relative to the original size of the image on which cascade will be applied.
    // Param scales 		is a number of scales from minScale to maxScale.
    // Param rejCriteria 	is used for NMS.
    CV_WRAP DetectorTrace(double minScale = 0.4, double maxScale = 5., int scales = 55, int rejCriteria = 1);

    CV_WRAP ~DetectorTrace();

    // Type of traces to return
    enum{NEGATIVE_TR=0,POSITIVE_TR,ALL_TR};

    // Return the vector of Trace objects.
    // Param image 			is a frame on which detector will be applied.
    // Param rois 			is a vector of regions of interest. Only the objects that fall into one of the regions will be returned.
    // Param positiveTrace 	is an output array of Positive Traces (eventually included local-maxima)
    // Param negativeTrace 	is an output array of Positive Traces
    // Param traceType 		is an output array of Trace
    virtual void detectTrace(InputArray image, InputArray rois, std::vector<Trace>& positiveTrace,std::vector<Trace>& negativeTrace, int traceType=cv::softcascade::DetectorTrace::ALL_TR);

private:
    void detectNoRoiTrace(const Mat& image, std::vector<Trace>& positiveTrace,std::vector<Trace>& negativeTrace);
    void detectAtTrace(const int dx, const int dy, const Level& level, const ChannelStorage& storage, std::vector<Trace>& positiveTrace,std::vector<Trace>& negativeTrace,uint levelI);

    // type to traces to return
    int traceType2Return;

};



}}


#endif /* SOFTCASCADE_PLUS_HPP_ */
