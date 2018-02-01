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
    * Calculate the RMSE here.
  */
    VectorXd rmse(4);
    rmse << 0,0,0,0;
    VectorXd c(4);
    c << 0,0,0,0;
    VectorXd tmp(4);
    tmp << 0,0,0,0;


    if (estimations.size() == 0){
        cout << "Ooops, the estimations vector are 0 length"<<endl;
    }
    
    if (estimations.size() != ground_truth.size()){
        cout << "Ooops, different sizes of vector"<<endl;
    }

    //accumulate squared residuals
    for(int i=0; i < estimations.size(); ++i){
        tmp = (estimations[i]-ground_truth[i]).array()*(estimations[i]-ground_truth[i]).array();
        c = tmp + c;
    }

    //calculate the mean
    c = c.array()/estimations.size();
    
    //calculate the squared root
    rmse = c.array().sqrt();
    
    //return the result
    return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  /**
    * Calculate a Jacobian here.
  */

    MatrixXd Hj(3,4);

    //
    float px = x_state(0);
    float py = x_state(1);
    float vx = x_state(2);
    float vy = x_state(3);

    float c1 = (px*px+py*py);
    float c2 = pow(c1,0.5);
    float c3 = c1*c2;
    
    //check
    if (fabs(c1)<0.0001){
        cout << "CalculateJacobian() - Error - Division by Zero"<<endl;
        return Hj;
    }

    //compute the Jacobian matrix
    Hj << px/c2, py/c2, 0, 0,
          -py/c1, px/c1, 0, 0,
          py*(vx*py-vy*px)/c3,px*(vy*px-vx*py)/c3,px/c2, py/c2;
          
    return Hj;

}
