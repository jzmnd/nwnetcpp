/*
agnanowire_modeling_v1.cpp
(Based on agnanowire_modeling_v3.py)
Random resitor network model for Ag nanowire network in oxide matrix
Using data from Will Scheideler

Created by Jeremy Smith on 2015-07-17
University of California, Berkeley
j-smith@eecs.berkeley.edu

Version 1.0
*/

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <unistd.h>
#include <string>
#include <vector>
#include <thread>
#include <ctime>
#include <Eigen/Dense>
#include "nwnet.h"

using namespace std;
using namespace Eigen;

void stats(std::vector<double>& v, double& average, double& stderr, double& median){
	double sum = 0;
	for(int i = 0; i < v.size(); i++){
		sum += v[i];
	}
	average = sum/v.size();
	double dev = 0;
	for(int i = 0; i < v.size(); i++){
		dev += (v[i] - average)*(v[i] - average);
	}
	stderr = std::sqrt(dev/(v.size()*(v.size()-1)));
	std::sort(v.begin(), v.end());
	median = v[v.size()/2];
}

int main(){
	/*
	Setup parameters
	Change values here for different nanowire networks
	*/
	std::srand(time(NULL) | (getpid() << 4));
	double substratesize = 50.0;
	// Points to test resistance as fraction of substrate size
	ArrayXXd testpoints(12, 2);
	testpoints << 0.250, 0.750, 0.750, 0.250,
	              0.250, 0.250, 0.750, 0.750,
	              0.375, 0.625, 0.625, 0.375,
	              0.375, 0.375, 0.625, 0.625,
	              0.323, 0.500, 0.677, 0.500,
	              0.500, 0.323, 0.500, 0.677;
	testpoints *= substratesize;
	double nwlength = 14.0;                                              // Nanowire length
	double nwlength_sd = 4.0;                                            // Standard deviation of wire lengths
	double nwdiameter = 0.033;                                           // Nanowire diameter
	double agresistivity = 1.59e-2;                                      // Ag resistivity
	
	double nwdensity[] = {0.02, 0.04, 0.05};                             // Nanowires per sq micron
	double nwinterres[] = {1.0, 10.0, 1000.0};                           // Resistance between wires

	double matrixrsheet = 1.0e8;                                         // Sheet resistance of matrix
	int runs = 4;                                                        // Number of runs per condition

	double nwresistance = 4*agresistivity/(M_PI*nwdiameter*nwdiameter);  // Nanowire resistance per unit length

	std::vector<std::string> summaryList;
	std::ofstream outfile;

	cout << "\n=================\n";
	cout << "Ag Nanowire Model\n";
	cout << "Jeremy Smith\n";
	cout << "=================\n\n";

	for(int j = 0; j < 3; j++){
		for(int k = 0; k < 3; k++){
			std::vector<WireNet> nets;
			std::vector<std::thread> procs;
			int nwnumber = nwdensity[k]*substratesize*substratesize;

			for(int i = 0; i < runs; i++){
				unsigned seed = rand()%10000;
				WireNet n(nwnumber, nwlength, nwlength_sd, substratesize, nwresistance, nwinterres[j], matrixrsheet, seed, false);
				nets.push_back(n);
			}
			for(int i = 0; i < runs; i++){
				procs.push_back(std::thread(&WireNet::solve, &nets[i]));
			}
			for(int i = 0; i < runs; i++){
				procs[i].join();
			}
			cout << endl;

			std::vector<double> rList;
			std::vector<double> tList;
			for(int i = 0; i < runs; i++){
				nets[i].parameters();
				for(int p = 0; p < testpoints.rows()/2 - 1; p++){
					Vector2d xy1 = testpoints.row(2*p);
					Vector2d xy2 = testpoints.row(2*p+1);
					int node1 = findnode(nets[i].nodeCoords, xy1);
					int node2 = findnode(nets[i].nodeCoords, xy2);
					double r = two_point_resistance(nets[i].eigenvalues, nets[i].eigenvectors, node1, node2);
					cout << "  R: " << r << "    Between nodes: " << node1 << "," << node2 << endl;
					rList.push_back(std::log10(r));
				}
				double t = 100*(1 - nets[i].areal_coverage(nwdiameter));
				cout << "  T: " << t << " " << char(37) << endl;
				tList.push_back(t);

				nets[i].output_files("data_" + std::to_string(j) + std::to_string(k) + char(65+i));    // Test conditions j and k and run number i
				cout << endl;
			}
			double r_average;
			double r_stderr;
			double r_median;
			double t_average;
			double t_stderr;
			double t_median;
			stats(rList, r_average, r_stderr, r_median);
			stats(tList, t_average, t_stderr, t_median);
			summaryList.push_back("data_" + std::to_string(j) + std::to_string(k) + '\t' 
				                  + to_string(nwlength) + '\t' + to_string(nwlength_sd) + '\t' + to_string(nwdiameter) + '\t' 
				                  + to_string(agresistivity) + '\t' + to_string(nwresistance) + '\t' 
				                  + to_string(nwdensity[k]) + '\t' + to_string(nwnumber) + '\t' 
				                  + to_string(nwinterres[j]) + '\t' + to_string(matrixrsheet) + '\t' 
				                  + to_string(r_average) + '\t' + to_string(r_median) + '\t' + to_string(r_stderr) + '\t' 
				                  + to_string(t_average) + '\t' + to_string(t_stderr) + '\n');
		}
	}

	cout << "Writing out summary file..." << endl;
    outfile.open("summary.txt");
    outfile << "filename\tnwlength\tnwstd\tnwdiameter\tagresistivity\tnwresistance\t"
            << "nwdensity\tnwnumber\tnwinterres\tmatrixrsheet\t"
            << "resistance\tmedresistance\tresistanceerr\ttransmission\ttransmissionerr\n";
    for(int i = 0; i < summaryList.size(); i++){
        outfile << summaryList[i];
    }
    outfile.close();
    cout << "DONE" << endl;
}
