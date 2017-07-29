#include <random>
#include <limits>
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

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	num_particles = 100;

	// creates a Gaussian distribution for x, y, theta
	normal_distribution<double> G_x(x, std[0]);
	normal_distribution<double> G_y(y, std[1]);
	normal_distribution<double> G_theta(theta, std[2]);

	// initialize particles
	for(int i = 0; i < num_particles; i++) {
		Particle p;
		p.id = i;
		p.x = G_x(gen);
		p.y = G_y(gen);
		p.theta = G_theta(gen);
		p.weight = 1.0;
		particles.push_back(p);

		weights.push_back(1.0);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	for(int i = 0; i < num_particles; i++) {
		Particle *p = &particles[i];

		// prevent divide by zero
		if(fabs(yaw_rate) < 0.00001) {  
			p->x += velocity * delta_t * cos(p->theta);
			p->y += velocity * delta_t * sin(p->theta);
		}
		else {
			p->x += (velocity/yaw_rate) * (sin(p->theta + (yaw_rate*delta_t)) - sin(p->theta));
			p->y += (velocity/yaw_rate) * (cos(p->theta) - cos(p->theta + (yaw_rate*delta_t)));
			p->theta += yaw_rate * delta_t;

			// add Gaussian noise to yaw
			normal_distribution<double> G_theta(p->theta, std_pos[2]);
			p->theta = G_theta(gen);
		}

		// add Gaussian noise to particle position
		normal_distribution<double> G_x(p->x, std_pos[0]);
		normal_distribution<double> G_y(p->y, std_pos[1]);
		p->x = G_x(gen);
		p->y = G_y(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	for(int i = 0; i < observations.size(); i++) {
		double min = numeric_limits<double>::max();

		// find closest predicted landmark to current observed landmark obs
		for(int j = 0; j < predicted.size(); j++) {
			LandmarkObs pred = predicted[j];
			double d = dist(observations[i].x, observations[i].y, pred.x, pred.y);
			if(d < min) { 
				min = d;
				observations[i].id = j;
			}
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	for(int n = 0; n < num_particles; n++) {
		// current particle
		Particle* particle = &particles[n];

		// build list of landmarks within range of particle
		std::vector<LandmarkObs> predicted;
		std::vector<Map::single_landmark_s> landmarks = map_landmarks.landmark_list;
		for(int i = 0; i < landmarks.size(); i++) {
			double d = dist(particle->x, particle->y, landmarks[i].x_f, landmarks[i].y_f);
			// add landmark observation to predictions if within sensor range
			if(d <= sensor_range) {
				LandmarkObs l;
				l.id = landmarks[i].id_i;
				l.x = landmarks[i].x_f;
				l.y = landmarks[i].y_f;
				
				predicted.push_back(l);
			}
		}

		// build list of observations transformed to map coordinates
		std::vector<LandmarkObs> t_observations;
		for(int i = 0; i < observations.size(); i++) {
			double x = observations[i].x;
			double y = observations[i].y;

			LandmarkObs l;
			l.id = observations[i].id;
			l.x = x * cos(particle->theta) - y * sin(particle->theta) + particle->x;
			l.y = x * sin(particle->theta) + y * cos(particle->theta) + particle->y;

			t_observations.push_back(l);
		}

		// associate predicted and observed landmarks
		dataAssociation(predicted, t_observations);

		particle->weight = 1.0;

		double std_x = std_landmark[0];
		double std_y = std_landmark[1];
		for(int i = 0; i < t_observations.size(); i++) {
			// set predicted landmark matching observation
			LandmarkObs o = t_observations[i];
			LandmarkObs p = predicted[o.id];
			// calculate Multivariate-Gaussian probability for current observation
			double weight = (1/(2 * M_PI * std_x * std_y)) * exp(-(pow(o.x - p.x, 2)/(2 * pow(std_x, 2)) + (pow(o.y - p.y, 2)/(2 * pow(std_y, 2)))));
			// multiply probability into current particle's weight
			particle->weight *= weight;
		}
		weights[n] = particle->weight;
	}
}

void ParticleFilter::resample() {
	// create discrete distribution to sample indices of particles
	std::discrete_distribution<int> distribution(weights.begin(), weights.end());
	std::vector<Particle> resampled;

	for(int i = 0; i < num_particles; i++) {
		int index = distribution(gen);
		resampled.push_back(particles[index]);
	}

	particles = resampled;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
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
