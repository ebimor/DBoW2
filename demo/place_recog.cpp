/**
 * File: Demo.cpp
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: demo application of DBoW2
 * License: see the LICENSE.txt file
 */

#include <iostream>
#include <vector>
#include <fstream>

// OpenCV
#include <opencv2/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <Eigen/Dense>
#include <ctime>
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/flann.hpp>


using namespace Eigen;
using namespace std;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

// number of training images
const int NIMAGES = 8;

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
void changeStructure(const vector<cv::Mat>& in, cv::Mat& out);
vector<cv::Mat> kmeanClusterFeatures(const vector<cv::Mat> &features, int k = 500, int max_iteration = 100000, double threshold = 1.0);
vector<vector<double>>  kmeanClusterFeaturesDouble(const vector<vector<double>> &features, int k, int max_iteration, double threshold);
vector<double> createHistogram(const vector<cv::Mat>& features, const vector<cv::Mat>& cluster_mean);
vector<double> createHistogram(const cv::Mat& features, const cv::Mat& cluster_mean);
int findClosestCluster(const cv::Mat& feature, const vector<cv::Mat>& cluster_mean);
int findClosestClusterDouble(const vector<double>& feature, const vector<vector<double>>& cluster_mean);
double cosineSimilarity(const vector<double>& a, const vector<double>& b);
double L1Similarity(const vector<double>& a, const vector<double>& b);
cv::Mat calculateMeanFeature(vector<cv::Mat>& f);
string type2str(int type);

using namespace cv;
using namespace cv::xfeatures2d;

// ----------------------------------------------------------------------------

int main()
{

  vector<cv::Mat > featuresDB_vec;
  featuresDB_vec.clear();
  featuresDB_vec.reserve(NIMAGES);

  cv::Ptr<cv::ORB> orb = cv::ORB::create();
  cv::Size size(320, 450);//the dst image size,e.g.100x100

  cv::Mat featuresDB;

  cout << "Extracting ORB features..." << endl;
  for(int i = 0; i < NIMAGES; ++i)
  {
    stringstream ss;
    ss << "camera/image" << i << ".jpg";

    cv::Mat image_org = cv::imread(ss.str(), 0);
    cout<<image_org.rows<<" x "<<image_org.cols<<endl;
	cv::Mat image;
	cv::resize(image_org,image,size);//resize image
    cv::Mat mask;
    vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    orb->detectAndCompute(image, mask, keypoints, descriptors);


    //showFeatures(keypoints, image);

    changeStructure(descriptors, featuresDB_vec);

    if(i == 0){
        	featuresDB = descriptors;
    }else{
        	cv::vconcat(featuresDB, descriptors, featuresDB);
    }
  }


    //create the BoW
  int k = 500;
  vector<cv::Mat> cluster_mean = kmeanClusterFeatures(featuresDB_vec, k, 1000, 1e-9);

  cv::Mat centroid_orb;
  changeStructure(cluster_mean, centroid_orb);

  // https://stackoverflow.com/questions/29694490/flann-error-in-opencv-3
  if(centroid_orb.type()!=CV_32F) {
	centroid_orb.convertTo(centroid_orb, CV_32F);
  }

  //going over the images to find histograms
  vector<vector<double>> hist_of_images, hist_of_images_flann;
  for(int i = 0; i < NIMAGES; ++i)
  {
  	stringstream ss;
    ss << "camera/image" << i << ".jpg";
    cv::Mat image_org = cv::imread(ss.str(), 0);
	cv::Mat image;
	cv::resize(image_org,image,size);//resize image
    cv::Mat mask;
    vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
	vector<cv::Mat > features;

    orb->detectAndCompute(image, mask, keypoints, descriptors);
    changeStructure(descriptors, features);

	hist_of_images.push_back(createHistogram(features, cluster_mean));

	// https://stackoverflow.com/questions/29694490/flann-error-in-opencv-3

	if(descriptors.type()!=CV_32F) {
    	descriptors.convertTo(descriptors, CV_32F);
	}

	hist_of_images_flann.push_back(createHistogram(descriptors, centroid_orb));
  }

  cout<<"ORB cosine similarity matrix without FLANN"<<endl;
  for(int i = 0; i < NIMAGES; i++){
  	vector<double> sim;
  	for(int j = 0; j < NIMAGES; j++){
  		sim.push_back(cosineSimilarity(hist_of_images[i], hist_of_images[j]));
  	}

  	vector<size_t> idx(sim.size());
    iota(idx.begin(), idx.end(), 0);

  	sort(idx.begin(), idx.end(), [&sim](size_t i1, size_t i2){ return sim[i1] > sim[i2];});

    cout << "Searching for Image " << i << ".jpg " << endl;
  	for(int j = 0; j< 4; j++){
  		cout << "Image " << i << " vs Image " << idx[j] << ": " << sim[idx[j]] << endl;
  	}
  	cout<<endl;
  }

  cout<<endl<<endl;

  cout<<"ORB L1 similarity matrix without FLANN"<<endl;
   for(int i = 0; i < NIMAGES; i++){
  	for(int j = 0; j < NIMAGES; j++){
  		cout<<L1Similarity(hist_of_images[i], hist_of_images[j])<<" ";
  	}
  	cout<<endl;
  }

  cout<<endl<<endl;

  cout<<"ORB cosine similarity matrix with FLANN"<<endl;
  for(int i = 0; i < NIMAGES; i++){
  	vector<double> sim;
  	for(int j = 0; j < NIMAGES; j++){
  		sim.push_back(cosineSimilarity(hist_of_images_flann[i], hist_of_images_flann[j]));
  	}

  	vector<size_t> idx(sim.size());
    iota(idx.begin(), idx.end(), 0);

  	sort(idx.begin(), idx.end(), [&sim](size_t i1, size_t i2){ return sim[i1] > sim[i2];});

    cout << "Searching for Image " << i << ".jpg " << endl;
  	for(int j = 0; j< 4; j++){
  		cout << "Image " << i << " vs Image " << idx[j] << ": " << sim[idx[j]] << endl;
  	}

	double fontScale = 0.3;
	
  	stringstream ss_i, display_text_i;
    ss_i << "camera/image" << i << ".jpg";
    cv::Mat image_org_i = cv::imread(ss_i.str(), 0);
	cv::Mat image_i;
	cv::resize(image_org_i,image_i,size);//resize image
	display_text_i<<"Image "<<i;
    cv::putText(image_i, display_text_i.str().c_str(), cv::Point(0,410),cv::FONT_HERSHEY_TRIPLEX,fontScale,cv::Scalar(255,255,255,125),1);



  	stringstream ss_j, display_text_j;
    ss_j << "camera/image" << idx[1] << ".jpg";
    cv::Mat image_org_j = cv::imread(ss_j.str(), 0);
	cv::Mat image_j;
	cv::resize(image_org_j,image_j,size);//resize image
	display_text_j<<"Image "<<idx[1]<<" with similarity "<<sim[idx[1]];
    cv::putText(image_j, display_text_j.str().c_str(), cv::Point(0,410),cv::FONT_HERSHEY_TRIPLEX,fontScale,cv::Scalar(255,255,255,125),1);

   	cv::hconcat(image_i, image_j, image_i);



   	cv::imshow("Matches", image_i);
    cv::waitKey(-1);

  	cout<<endl;
  }


  return 0;
}

// ----------------------------------------------------------------------------

string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

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
  //std::cout<<"added  "<<plain.rows<<" descriptors"<<std::endl;

}

void changeStructure(const vector<cv::Mat>& in, cv::Mat& out)
{

  out = in[0];

  for(int i = 1; i < in.size(); ++i)
  {
      cv::vconcat(out, in[i], out);
  }
  //std::cout<<"created a matrix of "<<out.rows<<" descriptors"<<std::endl;

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


vector<double> createHistogram(const cv::Mat& features, const cv::Mat& cluster_mean){
	
	vector<double> hist(cluster_mean.rows, 0);

	cv::flann::KMeansIndexParams indexParams(2000,11,cvflann::FLANN_CENTERS_KMEANSPP);

	cv::flann::Index kdtree(cluster_mean, indexParams);

	int maxPoints = 2;
    cv::Mat indices, dists; //(features.rows,maxPoints);
    //cv::Mat dists (features.rows,maxPoints);


    kdtree.knnSearch(features, indices, dists, maxPoints);




    //-- Filter matches using the Lowe's ratio test
    const float ratio_thresh = 0.9f;
    int num_good_features = 0;

    for (int i = 0; i < dists.rows; i++)
    {
        if ((float)dists.at<uchar>(i,0) < ratio_thresh * (float)dists.at<uchar>(i,1))
        {

        	num_good_features += 1;
			hist[(int)indices.at<uchar>(i,0)] += 1.0;


        }
    }

    cout<<num_good_features<<" distinctive features found"<<endl;

    for (size_t i = 0; i < hist.size(); i++){
        hist[i] = hist[i]/(double)num_good_features;
    }


	return hist;
}