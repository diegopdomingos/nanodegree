#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
  TODO:
    * Calculate the RMSE here.
  */
  VectorXd RMSE = VectorXd(4);
  RMSE<< 0,0,0,0;

  // check if we have enough data
  if(estimations.size() == 0){
    cout<<"The estimations are empty" << endl;
    return RMSE;
  }

  // check if estimations and ground_truth size are the same
  if(estimations.size() != ground_truth.size()){
    cout<<"Theata should have the same size"<<endl;
    return RMSE;
  }

  // Calculate the RMSE


  // Calculate the residuals
  for(int i=0; i<estimations.size(); ++i){
    VectorXd diff = estimations[i]-ground_truth[i];
    // root
    diff = diff.array()*diff.array();
    RMSE += diff;
  }

  // mean
  RMSE = RMSE/estimations.size();

  // square
  RMSE = RMSE.array().sqrt();

  return RMSE;
}
