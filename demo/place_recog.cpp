/**
 * File: Demo.cpp
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: demo application of DBoW2
 * License: see the LICENSE.txt file
 */

#include <iostream>
#include <vector>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <Eigen/Dense>
#include <ctime> 

using namespace Eigen;
using namespace std;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

// number of training images
const int NIMAGES = 4;

typedef Matrix<int, 1, 32> MatrixOrbf;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

void wait()
{
  cout << endl << "Press enter to continue" << endl;
  getchar();
}

void loadFeatures(vector<cv::Mat > &features);
void showFeatures(const vector<cv::KeyPoint>& keypoints, const cv::Mat& image);
void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out);
void  kmeanClusterFeatures(const vector<cv::Mat> &features, int k = 5);
double calculateDistance(const cv::Mat& f1, const cv::Mat& f2);


// ----------------------------------------------------------------------------

int main()
{
  vector<cv::Mat > featuresDB;
  loadFeatures(featuresDB);

  return 0;
}

// ----------------------------------------------------------------------------

void loadFeatures(vector<cv::Mat > &features)
{
  features.clear();
  features.reserve(NIMAGES);

  cv::Ptr<cv::ORB> orb = cv::ORB::create();

  cout << "Extracting ORB features..." << endl;
  for(int i = 0; i < NIMAGES; ++i)
  {
    stringstream ss;
    ss << "images/image" << i << ".png";

    cv::Mat image = cv::imread(ss.str(), 0);
    cv::Mat mask;
    vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    orb->detectAndCompute(image, mask, keypoints, descriptors);

    //showFeatures(keypoints, image);

    changeStructure(descriptors, features);

    //std::cout<<"size of the descriptors is : "<<descriptors.size()<<std::endl;
  }

  kmeanClusterFeatures(features);


}

// ----------------------------------------------------------------------------

void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out)
{

  for(int i = 0; i < plain.rows; ++i)
  {
    out.push_back(plain.row(i));

  }
  std::cout<<"added  "<<plain.rows<<" descriptors"<<std::endl;

}

//-----------------------------------------------------------------------------
void showFeatures(const vector<cv::KeyPoint>& keypoints, const cv::Mat& image)
{
    cv::Mat keypoints_image;

    cv::drawKeypoints(image, keypoints, keypoints_image, (0,255,0), 0);

        //-- Show detected matches
    cv::imshow("Matches", keypoints_image);
    cv::waitKey(-1);

}

double calculateDistance(const cv::Mat& f1, const cv::Mat& f2)
{

/*
	cv::Mat r = features[1]-features[0];
	std::cout<<norm(features[1], features[0])<<std::endl;

	double L =0;
	for(int i=0;i<features[0].cols;++i){
		int x = (int)features[1].at<uchar>(0,i)-(int)features[0].at<uchar>(0,i);
		L += pow( x, 2.0);
	}
	std::cout<<sqrt(L)<<std::endl;
*/

	return norm(f1, f2);
}

void  kmeanClusterFeatures(const vector<cv::Mat> &features, int k)
{
 	srand((unsigned)time(0)); 

	vector<cv::Mat> clusterMean;
	clusterMean.resize(k);
	std::vector<int> random_indexes; 
	random_indexes.resize(k);
	for(int i=0; i<k; ){
		int r = (rand()%features.size());
    	auto it = std::find (random_indexes.begin(), random_indexes.end(), r);
    	if(it == random_indexes.end()){
    		i++;
    		random_indexes.push_back(r);
			clusterMean.push_back(features[r]);
    	}
	}	

	for(auto it = features.begin(); it < features.end(); it++){
		for(auto ij = it+1; ij < features.end(); ij++ ){

		}
	}
}