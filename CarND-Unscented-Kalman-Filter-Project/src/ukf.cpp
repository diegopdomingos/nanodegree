#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 * This is scaffolding, do not modify
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.6;
  
  //DO NOT MODIFY measurement noise values below these are provided by the sensor manufacturer.
  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  //DO NOT MODIFY measurement noise values above these are provided by the sensor manufacturer.
  
  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */
  //cout<<"Initializing some variables...\n";
  n_x_ = 5;
  n_aug_ = 7;
  lambda_ = 3 - n_aug_;
  
  for(int i=0;i<5;i++){
    P_(i,i) = 1;
  }
  
  x_ << 1,1,1,1,1;

  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  init = false;

  // Initialize the weights_ vector
  weights_ = VectorXd(2*n_aug_+1);
  double weight_0 = lambda_/(lambda_+n_aug_);
  weights_(0) = weight_0;
  for (int i=1; i<2*n_aug_+1; i++) {
    double weight = 0.5/(n_aug_+lambda_);
    weights_(i) = weight;
  }
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */

  if(!init){
    if ((meas_package.sensor_type_ == MeasurementPackage::RADAR) && (use_radar_)) {
      double x = meas_package.raw_measurements_[0]*cos(meas_package.raw_measurements_[1]);
      double y = meas_package.raw_measurements_[0]*sin(meas_package.raw_measurements_[1]);
      double vx = meas_package.raw_measurements_[2]*cos(meas_package.raw_measurements_[1]);
      double vy = meas_package.raw_measurements_[2]*sin(meas_package.raw_measurements_[1]);
      double v = sqrt(vx*vx+vy*vy);
      // avoid numbers that can originates very high/low numbers

      /*if (x<0.001){
          x = 0.001;
      }

      if (y<0.001){
          y = 0.001;
      }*/

      x_ << x, y, v, 0, 0;
    } else if ((meas_package.sensor_type_ == MeasurementPackage::LASER) && (use_laser_)){
      double px = meas_package.raw_measurements_[0];
      double py = meas_package.raw_measurements_[1];

      // avoid numbers that can originates very high/low numbers
      if (fabs(px) < 0.001){
         px=0.001;
      }

      if (fabs(py) < 0.001){
         py=0.001;
      }
      x_ << px, py, 0, 0,0;
    }
    init=true;
    time_us_ = meas_package.timestamp_;
    return;
  }  

  float dt = (meas_package.timestamp_ - time_us_) / 1000000.0;//dt - expressed in seconds
  time_us_ = meas_package.timestamp_;
  Prediction(dt);

  if ((meas_package.sensor_type_ == MeasurementPackage::RADAR) && (use_radar_)) {
      UpdateRadar(meas_package);
  } else if ((meas_package.sensor_type_ == MeasurementPackage::LASER) && (use_laser_)){
      UpdateLidar(meas_package);
  }

}


void UKF::AugmentedSigmaPoints(MatrixXd* Xsig_out) {

  //cout<<"Starting AugmentedSigmaPoints function()\n";

  //create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);
  
  //create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

  //create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  //create augmented mean state
  x_aug.head(n_x_) = x_;
  x_aug(n_x_) = 0;
  x_aug(n_x_+1) = 0;
  
  //create augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(5,5) = std_a_*std_a_;
  P_aug(6,6) = std_yawdd_*std_yawdd_;
 
  //cout<<"P_aug="<<P_aug<<endl;
 
  //create square root matrix
  MatrixXd P_aug_sqrt = P_aug.llt().matrixL();
  
  //create augmented sigma points
  Xsig_aug.col(0) = x_aug;
  
  for(int i=0;i<n_aug_;i++){
    Xsig_aug.col(i+1) = x_aug + sqrt(lambda_+n_aug_)*P_aug_sqrt.col(i);
    Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_+n_aug_)*P_aug_sqrt.col(i);
  }

  //cout<<"Xsig_aug="<<Xsig_aug<<endl;

  *Xsig_out = Xsig_aug;
  
}


/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */

  //cout<<"Starting Prediction()\n";

  //create vector for weights
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2*n_aug_+1);
  AugmentedSigmaPoints(&Xsig_aug);
   
  for(int i=0;i<2*n_aug_+1;i++){
      
      double px  = Xsig_aug(0,i);
      double py  = Xsig_aug(1,i);
      double v   = Xsig_aug(2,i);
      double phi = Xsig_aug(3,i);
      double phi_dot = Xsig_aug(4,i);
      double nu_a = Xsig_aug(5,i);
      double nu_phi = Xsig_aug(6,i);
      //cout<<"HOT_DEBUG>>>>>>>>>>>>>>>>>>>>>>>>>\n";
      //cout<<px<<","<<py<<","<<v<<","<<phi<<","<<phi_dot<<","<<nu_a<<","<<nu_phi<<","<<delta_t<<endl;
      if(fabs(phi_dot) > 0.001){
          Xsig_pred_(0,i) = px + (v/phi_dot)*(sin(phi + phi_dot*delta_t)-sin(phi)) + 0.5*(delta_t*delta_t)*cos(phi)*nu_a;
          Xsig_pred_(1,i) = py + (v/phi_dot)*(-cos(phi + phi_dot*delta_t)+cos(phi)) + 0.5*(delta_t*delta_t)*sin(phi)*nu_a;
      } else {
          Xsig_pred_(0,i) = px + v*cos(phi)*delta_t + 0.5*(delta_t*delta_t)*cos(phi)*nu_a;
          Xsig_pred_(1,i) = py + v*sin(phi)*delta_t + 0.5*(delta_t*delta_t)*sin(phi)*nu_a;
      }
      Xsig_pred_(2,i) = v + delta_t*nu_a;
      Xsig_pred_(3,i) = phi + phi_dot*delta_t + 0.5*(delta_t*delta_t)*nu_phi;
      Xsig_pred_(4,i) = phi_dot + (delta_t)*nu_phi;
  }
 
  //cout<<"Xsig_pred_="<<Xsig_pred_<<endl;
  
  x_.fill(0.0);  
  //predict state mean
  x_ = Xsig_pred_*weights_;

  //cout<<"x_="<<x_<<endl;

  P_.fill(0.0);
  //predict state covariance matrix
  for(int i=0;i<2*n_aug_+1;i++){
      VectorXd x_diff = Xsig_pred_.col(i)-x_;
      while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
      while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;
      P_ += weights_(i)*(x_diff*x_diff.transpose());
  }

  //cout<<"P_"<<P_<<endl;

}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */

  //cout<<"Starting UpdateLidar()\n";

  int n_z = 2;

  MatrixXd Zsig = Xsig_pred_.block(0,0,n_z,2*n_aug_+1);
  UpdateUKF(meas_package,Zsig,n_z);
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */

  //cout<<"Starting UpdateRadar()\n";

  int n_z = 3;

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  //transform sigma points into measurement space
  Zsig.fill(0.0);
  for(int i=0;i<2*n_aug_+1;i++){
      Zsig(0,i)=sqrt(Xsig_pred_(0,i)*Xsig_pred_(0,i)+Xsig_pred_(1,i)*Xsig_pred_(1,i));
      Zsig(1,i)=atan2(Xsig_pred_(1,i),Xsig_pred_(0,i));
      Zsig(2,i)=(Xsig_pred_(0,i)*cos(Xsig_pred_(3,i))*Xsig_pred_(2,i)+Xsig_pred_(1,i)*sin(Xsig_pred_(3,i))*Xsig_pred_(2,i));
      Zsig(2,i) /= Zsig(0,i);
  }
  
  //cout<<"Zsig="<<Zsig<<endl;

  UpdateUKF(meas_package, Zsig, n_z);

}

void UKF::UpdateUKF(MeasurementPackage meas_package, MatrixXd Zsig, int n_z){

  //cout<<"Starting UpdateUKF\n";

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);

  //raw measurment by radar
  VectorXd z = meas_package.raw_measurements_;

  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);

  //cout<<"Zsig="<<Zsig<<endl;

  z_pred.fill(0.0);
  z_pred = Zsig*weights_;
  //cout<<"z_pred="<<z_pred<<endl;

  //calculate innovation covariance matrix S
  S.fill(0.0);
  for(int i=0;i<2*n_aug_+1;i++){
      VectorXd z_diff = Zsig.col(i) - z_pred;
      while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
      while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
      S += weights_(i)*z_diff*z_diff.transpose();
  }

  if(meas_package.sensor_type_ == MeasurementPackage::LASER){
    S(0,0) += std_laspx_*std_laspx_;
    S(1,1) += std_laspy_*std_laspy_;
  } else if (meas_package.sensor_type_ == MeasurementPackage::RADAR){
    S(0,0) += std_radr_*std_radr_;
    S(1,1) += std_radphi_*std_radphi_;
    S(2,2) += std_radrd_*std_radrd_;
  }

  //cout<<"S="<<S<<endl;
  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);

  //calculate cross correlation matrix
  Tc.fill(0.0);
  for(int i=0;i<2*n_aug_+1;i++){
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    Tc+= weights_(i) * x_diff * z_diff.transpose();
  }

  //calculate Kalman gain K;
  MatrixXd K = MatrixXd(n_x_, n_z);
  //cout<<"Tc:"<<Tc<<endl;
  //cout<<"S_inv"<<S.inverse()<<endl;
  K = Tc*S.inverse();
  
  //update state mean and covariance matrix
  VectorXd z_diff = z-z_pred;
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR){
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
  }
  x_ = x_ + K*z_diff;
  P_ = P_ - K*S*K.transpose();
  //cout<<"x_"<<x_<<endl;
  //cout<<"P_"<<P_<<endl;

  // Evaluate the NIS value
  if (meas_package.sensor_type_ == MeasurementPackage::LASER){
    NIS_radar_ = z.transpose()*S.inverse()*z;
  } else if (meas_package.sensor_type_ == MeasurementPackage::RADAR){
    NIS_laser_ = z.transpose()*S.inverse()*z;
  }


}
