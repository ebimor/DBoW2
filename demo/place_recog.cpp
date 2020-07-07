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
void showFeatures(const vector<cv::KeyPoint>& keypoints, const cv::Mat& image, cv::Scalar color = (0,255,0));
void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out);
vector<cv::Mat> kmeanClusterFeatures(const vector<cv::Mat> &features, int k = 500, int max_iteration = 1000, double threshold = 1.0);
vector<double> createHistogram(const vector<cv::Mat>& features, const vector<cv::Mat>& cluster_mean);
int findClosestCluster(const cv::Mat& feature, const vector<cv::Mat>& cluster_mean);

// ----------------------------------------------------------------------------

int main()
{
  vector<cv::Mat > featuresDB;
  loadFeatures(featuresDB);

    //create the BoW
  int k = 500;
  vector<cv::Mat> cluster_mean = kmeanClusterFeatures(featuresDB, k, 10, 10);

  cv::Ptr<cv::ORB> orb = cv::ORB::create();

  //going over the images to find histograms
  vector<vector<double>> hist_of_images;
  for(int i = 0; i < NIMAGES; ++i)
  {
  	stringstream ss;
    ss << "images/image" << i << ".png";
    cv::Mat image = cv::imread(ss.str(), 0);
    cv::Mat mask;
    vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
	vector<cv::Mat > features;

    orb->detectAndCompute(image, mask, keypoints, descriptors);
    changeStructure(descriptors, features);

    hist_of_images.push_back(createHistogram(features, cluster_mean));

  }

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
void showFeatures(const vector<cv::KeyPoint>& keypoints, const cv::Mat& image, cv::Scalar color)
{
    cv::Mat keypoints_image;

   	cv::drawKeypoints(image, keypoints, keypoints_image, color, 0);

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

vector<cv::Mat>  kmeanClusterFeatures(const vector<cv::Mat> &features, int k, int max_iteration, double threshold)
{
 	srand((unsigned)time(0)); 
	vector<cv::Mat> cluster_mean;
	std::vector<int> random_indexes; 
	random_indexes.resize(k);
	for(int i=0; i<k; ){
		int r = (rand()%features.size());
    	auto it = std::find (random_indexes.begin(), random_indexes.end(), r);
    	if(it == random_indexes.end()){
    		i++;
    		random_indexes.push_back(r);
			cluster_mean.push_back(features[r]);
    	}
	}


	int itr = 0;
	while(itr < max_iteration){
		vector<std::vector<int>> cluster_mean_elements; // this holds indexes of the feature points in cluster j<k
		cluster_mean_elements.resize(k);



		vector<int> features_index(features.size(), 0); // holds cluster of index of each feature

		for(size_t j = 0; j < features.size(); j++){

			//FIND THE CLUSTER THAT IS THE CLOSEST TO THE FEATURE J
			/*
			double minDist = 1e9;
			int cluster_index = 0;
			for (int i = 0; i < k ; i++){
				double dist_to_cluster_i = norm(features[j], cluster_mean[i]);
				if(dist_to_cluster_i < minDist){
					minDist = dist_to_cluster_i;
					cluster_index = i;
				}
			}
			*/

			int cluster_index = findClosestCluster(features[j], cluster_mean);

			features_index[j] = cluster_index;
			cluster_mean_elements[cluster_index].push_back(j);

		}

		double total_change_of_clusters = 0;
		
		for(int j = 0; j < k; j++){
			auto mean_f = features[0] - features[0]; 
			for(size_t i =0; i < cluster_mean_elements[j].size(); i++){
				mean_f += features[cluster_mean_elements[j][i]];
			}
			mean_f = mean_f/cluster_mean_elements[j].size();
			total_change_of_clusters += norm(cluster_mean[j], mean_f);
		}

		if(total_change_of_clusters < threshold){
			cout<<"threshold acheived!"<<endl;
			break;
		}

		cout<<"finished iteration "<<itr++<<endl;
	}

	return cluster_mean;
	
}

vector<double> createHistogram(const vector<cv::Mat>& features, const vector<cv::Mat>& cluster_mean){

	vector<double> hist(cluster_mean.size(), 0);

	for(size_t i = 0; i < features.size(); i++){
		int cluster_index_feature_i = findClosestCluster(features[i], cluster_mean);
		hist[cluster_index_feature_i] += (double)1.0/features.size();
	}

	return hist;

}

int findClosestCluster(const cv::Mat& feature, const vector<cv::Mat>& cluster_mean){
	double minDist = DBL_MAX;
	int cluster_index = 0;
	for (size_t i = 0; i < cluster_mean.size() ; i++){
		double dist_to_cluster_i = norm(feature, cluster_mean[i]);
		if(dist_to_cluster_i < minDist){
			minDist = dist_to_cluster_i;
			cluster_index = i;
		}
	}

	return cluster_index;
}