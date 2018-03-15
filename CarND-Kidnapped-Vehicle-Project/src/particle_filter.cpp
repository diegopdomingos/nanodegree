/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	num_particles = 10;
	is_initialized = true;

	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y,std[1]);
	normal_distribution<double> dist_theta(theta,std[2]);
	
	for (int i=0;i<num_particles;i++){
		struct Particle new_particle;
		new_particle.id = i;
		new_particle.theta = theta + dist_theta(gen);
		new_particle.x = x + dist_x(gen);
		new_particle.y = y + dist_y(gen);
		new_particle.weight = 1;
		particles.push_back(new_particle);
	}
}	

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;

	normal_distribution<double> dist_x(0,std_pos[0]);
	normal_distribution<double> dist_y(0,std_pos[1]);
	normal_distribution<double> dist_theta(0,std_pos[2]);

	for (int i=0;i<num_particles;i++){
		Particle curr_p = particles[i];

		/* avoid division by zero */
		if (fabs(yaw_rate) < 0.00001){
			particles[i].x += velocity*delta_t*cos(curr_p.theta) + dist_x(gen);
			particles[i].y += velocity*delta_t*sin(curr_p.theta) + dist_y(gen);
		} else {
			particles[i].x += (velocity/yaw_rate)*(sin(curr_p.theta + yaw_rate*delta_t)-sin(curr_p.theta)) + dist_x(gen);
			particles[i].y += (velocity/yaw_rate)*(cos(curr_p.theta)-cos(curr_p.theta + yaw_rate*delta_t)) + dist_y(gen);
			particles[i].theta += yaw_rate*delta_t + dist_theta(gen);
		}

	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.


	for (unsigned int i=0;i<observations.size();i++){

		LandmarkObs curr_obs = observations[i];

		/* set the min_dist as the maximum number available (inf) */
		double min_dist = numeric_limits<double>::max();
		int landmark_id = -1;

		/* find the closest landmark of the observation */
		for(unsigned int j=0;j<predicted.size();j++){
			LandmarkObs curr_predict = predicted[j];
			double curr_dist = dist(curr_obs.x, curr_obs.y, curr_predict.x, curr_predict.y);
			if (curr_dist < min_dist){
				min_dist = curr_dist;
				landmark_id = curr_predict.id;
			}
		}

		observations[i].id = landmark_id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html


	for (int i=0;i<num_particles;i++){

		vector<LandmarkObs> observations_map;
		vector<LandmarkObs> predictions;

		float particle_x = particles[i].x;
		float particle_y = particles[i].y;
		float particle_theta = particles[i].theta;

		// Consider only landmarks within sensor range
		for (int j=0;j<map_landmarks.landmark_list.size();j++){
			float landmark_x = map_landmarks.landmark_list[j].x_f;
			float landmark_y = map_landmarks.landmark_list[j].y_f;
			if (dist(particle_x,particle_y,landmark_x,landmark_y)<=sensor_range){
				predictions.push_back(LandmarkObs{map_landmarks.landmark_list[j].id_i,landmark_x,landmark_y});
			}
		}

		// Convert the observation to map coordinates
		for (int j=0;j<observations.size();j++){
			LandmarkObs observation_tmp;
			observation_tmp.x = particle_x+(cos(particle_theta)*observations[j].x)-(sin(particle_theta)*observations[j].y);
			observation_tmp.y = particle_y+(sin(particle_theta)*observations[j].x)+(cos(particle_theta)*observations[j].y);
			observation_tmp.id = observations[j].id;
			observations_map.push_back(observation_tmp);
		}

		// Find associations
		dataAssociation(predictions, observations_map);

		// Reset particles weight
		particles[i].weight = 1.0;
		
		// Calculate the weight for this particle

		for (unsigned int j=0;j<observations_map.size();j++){
			float partial_weight = 1.0;
			for(unsigned int n=0;n<predictions.size();n++){
				if(predictions[n].id == observations_map[j].id){
					particles[i].weight *= (1/(2*M_PI*std_landmark[0]*std_landmark[1]))*exp(-(pow(predictions[n].x-observations_map[j].x,2)/(2*pow(std_landmark[0],2)) 
									+ (pow(predictions[n].y-observations_map[j].y,2)/(2*pow(std_landmark[1],2)))));
				}
			}
		}
	}
	
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution


	vector<Particle> res_particles;	
	vector<double> weights;

	/* fill weights with weight of each particle */
	for (int i=0;i<num_particles;i++){
		weights.push_back(particles[i].weight);
	}


	uniform_int_distribution<int> dist_i(0, num_particles-1);

	int index = dist_i(gen);

	double max_weight = *max_element(weights.begin(), weights.end());

	uniform_real_distribution<double> dist_r(0.0, 2*max_weight);

	double beta = 0.0;

	/* wheel technique for choose the particle with high probability */
	for(int i=0;i<num_particles; i++){

		beta+=dist_r(gen);

		while (beta > weights[index]){
			beta-=weights[index];
			index = (index+1) % num_particles;
		}

		res_particles.push_back(particles[index]);
	}

	particles = res_particles;

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
