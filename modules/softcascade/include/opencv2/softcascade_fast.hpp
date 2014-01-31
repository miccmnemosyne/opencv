/*
 * softcascade_plus.hpp
 *
 *  Created on: Oct 29, 2013
 *      Author: federico
 */

#ifndef SOFTCASCADE_FAST_HPP_
#define SOFTCASCADE_FAST_HPP_

#include "softcascade.hpp"

#include <fstream>
#include <map>
#include <list>
#include <numeric>
#include <set>

struct Level;
struct ChannelStorage;

#define DEBUG_MSG

#ifdef DEBUG_MSG
#define DEBUG_MSG(str) do { std::cout << "<<debug>>" << str << std::endl; } while( false )
#else
#define DEBUG_MSG(str) do { } while ( false )
#endif

namespace cv { namespace softcascade {


// ============================================================================================================== //
//					     Declaration of DetectorFast (with trace evaluation reduction)
// ============================================================================================================= //
struct CV_EXPORTS ParamDetectorFast
{
	ParamDetectorFast();
	ParamDetectorFast(double minScale, double maxScale, uint nScale, int nMS, uint lastStage, uint gridSize,double gamma,uint round);

	// pyramid settings
	double	minScale;
	double	maxScale;
	uint 	nScales;

	// No Maximum Soppression
	int 	nMS;

	// Linear Cascade Approximation parameter
	uint 	lastStage;

	// Geometric moedl: grid size
	uint 	gridSize;
	double 	gamma;
	uint 	round;
};


struct CV_EXPORTS FastDtModel
{
	// tag for model (xml file)
	static const char *const ROOT;
	static const char *const PYRAMID;
	static const char *const PYRAMID_MINS;
	static const char *const PYRAMID_MAXS;
	static const char *const PYRAMID_NS;

	static const char *const TRAININGS;
	static const char *const TRAININGS_DATAF;
	static const char *const TRAININGS_NIMG;
	static const char *const TRAININGS_IMGS;

	static const char *const MODELS;

	struct Block;

	FastDtModel(ParamDetectorFast paramDtFast, String datase,uint numImages,Size imgSize);
	FastDtModel();
	//FastDtModel(uint numLevels);


    void write(cv::FileStorage& fso) const;
    void read(const cv::FileNode& node);

    // Interface for Trace-Model
    void addTraceForTraceModel(uint stage,uint level,const std::vector<Point2d>& trace);
    void computeModel();
    bool getSlopeAt(uint stage,uint level,double& slope);
    void getLastSt(std::vector<uint>& stages);
    bool getLevelsForStage(uint  lastStage, std::vector<uint>& levels);

    // Interface for Geometry-Model
    void addStrongWithROI(Rect dw,uint64 rank,uint level);
    void setGridsSize(std::vector<uint> grids);
    std::vector<Block>& getBlocks4Grid(uint gridSize);
    void resolveWrongStd();
    void smoothLocations();
    void saveModelIntoDat(String path);



    // ------------ Parameters ---------------------

    // List of octaves of the soft cascade (in logarithmic scale)
    std::vector<int> octaves;
    // List of levels of the soft cascade
    std::vector<double> levels;

    ParamDetectorFast paramDtFast;

    // Additional information for training of trace/geometry models
    String	dataset;
    uint	numImages;
    Size	imgSize;

	struct AverageStd{
		AverageStd(){};
		AverageStd(Mat a, Mat c): avg(a),std(c){};

		Mat avg;
		Mat std;
	};

	struct Block{
		Block(uint levels)
		:levelsHist(std::vector<double>(levels,0.)),
		locationsHist(std::vector<AverageStd>(levels,AverageStd())),
		energy(0.){};

		Block(std::vector<double> lvH,std::vector<AverageStd> locH, Rect rt, double e)
		:levelsHist(lvH), locationsHist(locH), rect(rt), energy(e){};

		std::vector<double>  	levelsHist;
		std::vector<AverageStd> locationsHist;
		Rect rect;
		double 					energy;
 	};

private:
    struct TraceModel{
#define Vx  0
#define Vy  1

    	static const char *const TRACEMODEL;
    	static const char *const TRACEMODEL_LASTSTAGES;
    	static const char *const TRACEMODEL_LASTSTAGES_LASTS;
    	static const char *const TRACEMODEL_LASTSTAGES_SLOPES;



    	typedef std::map<uint,std::map<uint,std::vector<Vec4f> > > LinesMap;
    	typedef std::map<uint,std::vector<Vec4f> >  LevelsMap;
    	//typedef std::map<uint,std::map<uint,double > > SlopesMap;
    	typedef std::map<uint,std::vector<double> > SlopesMap;


    	TraceModel(){}

    	void compute(uint levels);
    	void write(FileStorage& fso) const;
    	void read(const FileNode& node);

    	//output line parameter in affine coordinate
    	//stage->level->lines
    	//std::map<uint,std::map<uint,std::vector<Vec4f> > > linesParam;
    	//std::map<uint,std::map<uint,double> > slopes;
    	LinesMap linesParam;
    	SlopesMap slopes;

    }traceModel;

    struct GeomModel{
    	static const char *const GEOMMODEL;
    	static const char *const GEOMMODEL_GRIDS;
    	static const char *const GEOMMODEL_GRID_SIZE;
    	static const char *const GEOMMODEL_GRID_BLOCKS;
    	static const char *const GEOMMODEL_GRID_BLOCKS_ID;
    	static const char *const GEOMMODEL_GRID_BLOCKS_LEVELSH;
    	static const char *const GEOMMODEL_GRID_BLOCKS_LOCATIONSH;
    	static const char *const GEOMMODEL_GRID_BLOCKS_LOCATIONSH_AVG;
    	static const char *const GEOMMODEL_GRID_BLOCKS_LOCATIONSH_STD;
    	static const char *const GEOMMODEL_GRID_BLOCKS_RECT;
    	static const char *const GEOMMODEL_GRID_BLOCKS_ENERGY;

    	struct StrongROI{
    		StrongROI(Rect d, uint64 r):dw(d),rank(r){};

    		Rect dw;
    		uint64 rank;

    	};

    	typedef std::map<uint,std::vector<StrongROI> > StrongsROI;


    	typedef std::map<uint,std::vector<Block> > Grids;

    	GeomModel(){}

    	void compute(Size imageSize,uint levels);
    	void write(FileStorage& fso) const;
    	void read(const FileNode& node);


    	// variables for storage input data
    	StrongsROI  		upperLeftPonts;
    	std::vector<uint> 	gridsSize;

    	// model
    	Grids	grids;

    }geomModel;
};
const char *const FastDtModel::ROOT="Fast_Detector";
const char *const FastDtModel::PYRAMID="Pyramid_Setting";
const char *const FastDtModel::PYRAMID_MINS="minScale";
const char *const FastDtModel::PYRAMID_MAXS="maxScale";
const char *const FastDtModel::PYRAMID_NS="nScales";
const char *const FastDtModel::TRAININGS="Training_Set";
const char *const FastDtModel::TRAININGS_DATAF="datasetFolder";
const char *const FastDtModel::TRAININGS_NIMG="numImages";
const char *const FastDtModel::TRAININGS_IMGS="imageSize";
const char *const FastDtModel::MODELS="Models";

const char *const FastDtModel::TraceModel::TRACEMODEL="Trace_Model";
const char *const FastDtModel::TraceModel::TRACEMODEL_LASTSTAGES="LastStages";
const char *const FastDtModel::TraceModel::TRACEMODEL_LASTSTAGES_LASTS="lastStage";
const char *const FastDtModel::TraceModel::TRACEMODEL_LASTSTAGES_SLOPES="slopes";


const char *const FastDtModel::GeomModel::GEOMMODEL="Geometry_Model";
const char *const FastDtModel::GeomModel::GEOMMODEL_GRIDS="Grids";
const char *const FastDtModel::GeomModel::GEOMMODEL_GRID_SIZE="size";
const char *const FastDtModel::GeomModel::GEOMMODEL_GRID_BLOCKS="Blocks";
const char *const FastDtModel::GeomModel::GEOMMODEL_GRID_BLOCKS_ID="id";
const char *const FastDtModel::GeomModel::GEOMMODEL_GRID_BLOCKS_LEVELSH="levelsHist";
const char *const FastDtModel::GeomModel::GEOMMODEL_GRID_BLOCKS_LOCATIONSH="locationsHist";
const char *const FastDtModel::GeomModel::GEOMMODEL_GRID_BLOCKS_LOCATIONSH_AVG="averages";
const char *const FastDtModel::GeomModel::GEOMMODEL_GRID_BLOCKS_LOCATIONSH_STD="standardDev";
const char *const FastDtModel::GeomModel::GEOMMODEL_GRID_BLOCKS_RECT="rect";
const char *const FastDtModel::GeomModel::GEOMMODEL_GRID_BLOCKS_ENERGY="energy";


// required for cv::FileStorage serialization
inline void write(cv::FileStorage& fso, const std::string&, const FastDtModel& x){
	x.write(fso);
}
inline void read(const cv::FileNode& node, FastDtModel& x, const FastDtModel& default_value){
	if(node.empty())
		x=default_value;
	else
		x.read(node);

}
// For print FastModel to the console
std::ostream& operator<<(std::ostream& out, const FastDtModel& m);



class CV_EXPORTS_W DetectorFast: public Detector{

public:

	// An empty cascade will be created.
    // Param minScale 		is a minimum scale relative to the original size of the image on which cascade will be applied.
    // Param minScale 		is a maximum scale relative to the original size of the image on which cascade will be applied.
    // Param scales 		is a number of scales from minScale to maxScale.
    // Param rejCriteria 	is used for NMS.
    //CV_WRAP DetectorFast(double minScale = 0.4, double maxScale = 5., int scales = 55, int rejCriteria = 1);
	CV_WRAP DetectorFast(ParamDetectorFast parameters);

    CV_WRAP ~DetectorFast();

    // Load soft cascade from FileNode and trace-model.
    // Param fileNode 		is a root node for cascade.
    // Param fileNodeModel 	is a root node for trace-model.
    CV_WRAP virtual bool load(const FileNode& cascadeModel,const FileNode& fastModel);


    // Return the vector of Detection objects (with fast evaluation).
    // Param image is a frame on which detector will be applied.
    // Param rois is a vector of regions of interest. Only the objects that fall into one of the regions will be returned.
    // Param objects is an output array of Detections
    virtual void detectFast(cv::InputArray _image,std::vector<Detection>& objects);

    // Save both models (trace and geometry) respectively in path/Trace_Model.dat and path/Geometry_Model.dat
    void saveModelIntoDat(String path);

    CV_WRAP uint getNumLevels();
private:

    // Load trace-model
    CV_WRAP virtual bool loadModel(const FileNode& fastNode);


	struct CV_EXPORTS TempInfo{
    	int rejCriteria;
    	int*	index;
    	int*	weaks;
    };

	TempInfo 	tempI;
	FastDtModel fastModel;

};
struct classPoint2iComp{
	inline bool operator()(const Point2i& a, const Point2i& b){
		return (a.x<b.x)&&(a.y!=b.y);

	}
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
	Trace(const uint64 ind,const uint octave, const uint level, const Detection& dw, const std::vector<float>& scores, const std::vector<float>& stagesResp, const int classification);

	enum{NEGATIVE=0,POSITIVE,LOCALMAXIMUM};
	uint64 	index;
	uint64 	localMaxIndex;
	uint 	octaveIndex;
	uint 	levelIndex;
	Detection detection;
	std::vector<float> subscores;
	std::vector<float> stages;
	int classType;
};


class CV_EXPORTS_W DetectorTrace: public Detector{

public:

	// An empty cascade will be created.
    // Param minScale 		is a minimum scale relative to the original size of the image on which cascade will be applied.
    // Param minScale 		is a maximum scale relative to the original size of the image on which cascade will be applied.
    // Param scales 		is a number of scales from minScale to maxScale.
    // Param rejCriteria 	is used for NMS.
    //CV_WRAP DetectorTrace(double minScale = 0.4, double maxScale = 5., int scales = 55, int rejCriteria = 1);
    CV_WRAP DetectorTrace(ParamDetectorFast param);

    CV_WRAP ~DetectorTrace();

    // Type of traces to return
    enum{NEGATIVE_TR=0,POSITIVE_TR,LOCALMAXIMUM_TR,NEG_POS_TR};

    // Return the vector of Trace objects.
    // Param image 			is a frame on which detector will be applied.
    // Param rois 			is a vector of regions of interest. Only the objects that fall into one of the regions will be returned.
    // Param positiveTrace 	is an output array of Positive Traces (eventually included local-maxima)
    // Param negativeTrace 	is an output array of Positive Traces
    // Param traceType 		is an output array of Trace
    virtual void detectTrace(InputArray image, InputArray rois, std::vector<Trace>& positiveTrace,std::vector<Trace>& negativeTrace, int traceType=cv::softcascade::DetectorTrace::LOCALMAXIMUM_TR);

private:
    void detectNoRoiTrace(const Mat& image, std::vector<Trace>& positiveTrace,std::vector<Trace>& negativeTrace);
    void detectAtTrace(const int dx, const int dy, const Level& level, const ChannelStorage& storage, std::vector<Trace>& positiveTrace,std::vector<Trace>& negativeTrace,uint levelI);

    // type to traces to return
    int traceType2Return;

};



}}


#endif /* SOFTCASCADE_PLUS_HPP_ */
