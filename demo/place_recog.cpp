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
vector<cv::Mat> kmeanClusterFeatures(const vector<cv::Mat> &features, int k = 500, int max_iteration = 100000, double threshold = 1.0);
vector<vector<double>>  kmeanClusterFeaturesDouble(const vector<vector<double>> &features, int k, int max_iteration, double threshold);
vector<double> createHistogram(const vector<cv::Mat>& features, const vector<cv::Mat>& cluster_mean);
int findClosestCluster(const cv::Mat& feature, const vector<cv::Mat>& cluster_mean);
int findClosestClusterDouble(const vector<double>& feature, const vector<vector<double>>& cluster_mean);
double cosineSimilarity(const vector<double>& a, const vector<double>& b);
double L1Similarity(const vector<double>& a, const vector<double>& b);
cv::Mat calculateMeanFeature(vector<cv::Mat>& f);

// ----------------------------------------------------------------------------

int main()
{

  vector<cv::Mat > featuresDB;
  loadFeatures(featuresDB);

  vector<cv::Mat> df;
  df.push_back(featuresDB[0]);
  df.push_back(featuresDB[1]);

  cout<<" featuresDB[0] : "<<featuresDB[0]<<endl;
  cout<<" featuresDB[1] : "<<featuresDB[1]<<endl;

    cv::Mat mean = calculateMeanFeature(df);

  cout<<"mean is "<<mean<<endl;


    //create the BoW
  int k = 1000;
  vector<cv::Mat> cluster_mean = kmeanClusterFeatures(featuresDB, k, 1000, 1e-6);

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

  for(int i = 0; i < NIMAGES; i++){
  	for(int j = 0; j < NIMAGES; j++){
  		cout<<cosineSimilarity(hist_of_images[i], hist_of_images[j])<<" ";
  	}
  	cout<<endl;
  }

  cout<<endl<<endl;

   for(int i = 0; i < NIMAGES; i++){
  	for(int j = 0; j < NIMAGES; j++){
  		cout<<L1Similarity(hist_of_images[i], hist_of_images[j])<<" ";
  	}
  	cout<<endl;
  }


  /* Test kmean for 2D data
  vector<vector<double>> data;
  for(int k = 0 ; k< 1000; k++){
  	std::vector<double> v(2,0);
  	v[1] = rand() % 10;
  	v[0] = rand() % 10;
  	data.push_back(v);
  }

  for(int k = 0 ; k< 1000; k++){
  	std::vector<double> v(2,0);
  	v[1] = rand() % 10+30;
  	v[0] = rand() % 10+30;
  	data.push_back(v);
  }

  vector<vector<double>> cluster_mean = kmeanClusterFeaturesDouble(data, 2, 100000, 1);

  cout<<cluster_mean[0][0]<<" "<<cluster_mean[0][1]<<endl;
  cout<<cluster_mean[1][0]<<" "<<cluster_mean[1][1]<<endl;
  */

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

cv::Mat calculateMeanFeature(vector<cv::Mat>& f)
{
	vector<int> mean(f[0].cols, 0);

	for(int i = 0; i < f.size(); i++){
		
		//mean += f[i]
		for(int j = 0; j < f[i].cols; j++){
			mean[j] += f[i].at<uchar>(0,j);
		}
	}

	cv::Mat mean_f = f[0];
	for(int i = 0; i < mean.size(); i++){
		int d = mean[i] / f.size();
		mean_f.at<uchar>(0,i) = (uchar)d;
	}

	return mean_f;

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

		//find the closest cluster to feature j, store the result in cluster_mean_elements
		for(size_t j = 0; j < features.size(); j++){

			int cluster_index = findClosestCluster(features[j], cluster_mean);
			cluster_mean_elements[cluster_index].push_back(j);

		}

		double total_change_of_clusters = 0;
		//update clusters centeroid
		for(int j = 0; j < k; j++){

			vector<cv::Mat> cluster_j_features; 
			for(size_t i =0; i < cluster_mean_elements[j].size(); i++){
				cluster_j_features.push_back(features[cluster_mean_elements[j][i]]);
			}

			auto mean_f = calculateMeanFeature(cluster_j_features);

			total_change_of_clusters += norm(cluster_mean[j], mean_f);

			cluster_mean[j] = mean_f;
		}

		if(total_change_of_clusters < threshold){
			cout<<"threshold acheived!"<<endl;
			break;
		}

		cout<<"finished iteration "<<itr++<<endl;
	}

	return cluster_mean;
	
}

vector<vector<double>>  kmeanClusterFeaturesDouble(const vector<vector<double>> &features, int k, int max_iteration, double threshold)
{
 	srand((unsigned)time(0)); 
	vector<vector<double>> cluster_mean;
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

			int cluster_index = findClosestClusterDouble(features[j], cluster_mean);

			features_index[j] = cluster_index;
			cluster_mean_elements[cluster_index].push_back(j);

		}

		double total_change_of_clusters = 0;
		
		for(int j = 0; j < k; j++){
			vector<double> mean_f(features[0].size(), 0); 
			for(size_t i =0; i < cluster_mean_elements[j].size(); i++){
				for(int ii = 0; ii<mean_f.size();ii++){
					mean_f[ii] += features[cluster_mean_elements[j][i]][ii];
				}
			}


			for(auto it = mean_f.begin(); it < mean_f.end(); it++){
				*it = *it/(double)cluster_mean_elements[j].size();
			}

			double norm_f_j = 0;
			for(int i = 0; i < mean_f.size(); i++){
				norm_f_j += (mean_f[i]-cluster_mean[j][i])*(mean_f[i]-cluster_mean[j][i]);
			}
			total_change_of_clusters += sqrt(norm_f_j);

			cluster_mean[j] = mean_f;
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

int findClosestClusterDouble(const vector<double>& feature, const vector<vector<double>>& cluster_mean){
	double minDist = DBL_MAX;
	int cluster_index = 0;
	for (size_t i = 0; i < cluster_mean.size() ; i++){

		double dist_to_cluster_i = 0;
		for(int ii = 0; ii < feature.size(); ii++){
			dist_to_cluster_i += (feature[ii]-cluster_mean[i][ii])*(feature[ii]-cluster_mean[i][ii]);
		}

		dist_to_cluster_i = sqrt(dist_to_cluster_i);

		if(dist_to_cluster_i < minDist){
			minDist = dist_to_cluster_i;
			cluster_index = i;
		}
	}

	return cluster_index;
}

double cosineSimilarity(const vector<double>& a, const vector<double>& b){

	double abDot = 0, a_abs = 0, b_abs = 0;
	for(int i = 0; i < a.size(); i++){
		abDot += a[i]*b[i];
		a_abs += a[i]*a[i];
		b_abs += b[i]*b[i];
	}

	return abDot/(sqrt(a_abs)*sqrt(b_abs));
}

double L1Similarity(const vector<double>& a, const vector<double>& b){

	double L1norm = 0;
	for(int i = 0; i < a.size(); i++){
		L1norm += abs(a[i]-b[i]);
	}

	return L1norm/(double)a.size();
}