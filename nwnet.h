/*
Random Nanowire Network Model Library
nwnet.h

Jeremy Smith
j-smith@eecs.berkeley.edu
EECS, University of California Berkeley

Version 1.1
*/

#ifndef NWNET_H
#define NWNET_H

#include <iostream>
#include <mutex>
#include <string>
#include <vector>
#include <list>
#include <random>
#include <thread>
#include <Eigen/Dense>
using std::cout;
using std::endl;
using Eigen::Vector2d;
using Eigen::VectorXd;
using Eigen::ArrayXd;
using Eigen::ArrayXi;
using Eigen::ArrayXXd;
using Eigen::ArrayXXi;
using Eigen::MatrixXd;

typedef std::pair<int, double> enumerate;

std::mutex m;

// Comparison function for argsort
bool argsort_comp(const enumerate& left, const enumerate& right){
    return left.second < right.second;
}

// Argsort function that returns index of the sorted Eigen array
template<typename Derived>
ArrayXi argsort(Eigen::ArrayBase<Derived>& array){
    ArrayXi indices(array.size());
    std::vector<enumerate> enum_array(array.size());

    for(int i = 0; i < array.size(); i++){
        enum_array[i].first = i;
        enum_array[i].second = array(i);
    }
    std::sort(enum_array.begin(), enum_array.end(), argsort_comp);
    for(int i = 0; i < array.size(); i++){
        indices(i) = enum_array[i].first;
    }
    return indices;
}

// Function that checks for the intersection of two line segments given four coordinates
void intersectCheck(Vector2d& start1, Vector2d& end1, Vector2d& start2, Vector2d& end2, Vector2d& output, bool& intersect){
    using std::min;
    using std::max;
    double pden = (start1(0) - end1(0))*(start2(1) - end2(1)) - (start1(1) - end1(1))*(start2(0) - end2(0));
    double px = ((start1(0)*end1(1) - start1(1)*end1(0))*(start2(0) - end2(0)) - (start1(0) - end1(0))*(start2(0)*end2(1) - start2(1)*end2(0)))/pden;
    double py = ((start1(0)*end1(1) - start1(1)*end1(0))*(start2(1) - end2(1)) - (start1(1) - end1(1))*(start2(0)*end2(1) - start2(1)*end2(0)))/pden;

    intersect = ((px >= min(start1(0), end1(0))) & (px <= max(start1(0), end1(0)))) &
                ((px >= min(start2(0), end2(0))) & (px <= max(start2(0), end2(0)))) &
                ((py >= min(start1(1), end1(1))) & (py <= max(start1(1), end1(1)))) &
                ((py >= min(start2(1), end2(1))) & (py <= max(start2(1), end2(1))));

    output << px, py;
}

// Calculates the resistance between 2 points using matrix solution
double two_point_resistance(VectorXd& val, MatrixXd& vec, int node1, int node2){
    double r12 = 0;
    int totalnodes = val.rows();

    for(int i = 0; i < totalnodes; i++){
        if(val(i) <= 0){
            continue;
        }else{
            r12 += (1.0/val(i))*std::pow((vec(node1,i) - vec(node2,i)), 2);
        }
    }
    return r12;
}

// Find the closest node to a particular (x,y) coordinate
int findnode(ArrayXXd& nodeslist, Vector2d& xycoord){
    int totalnodes = nodeslist.rows();
    ArrayXd distances(totalnodes);
    ArrayXd::Index index;

    for(int i = 0; i < totalnodes; i++){
        double dx = nodeslist(i,0) - xycoord(0);
        double dy = nodeslist(i,1) - xycoord(1);
        distances(i) = std::sqrt(std::abs(dx*dx + dy*dy));
    }
    distances.minCoeff(&index);
  return index;
}

// Function to generate random angle in range [0,2pi]
ArrayXd create_random_angles(int numwires, float& sparam, unsigned seed){
    std::default_random_engine generator(seed);
    std::uniform_real_distribution<double> distribution(0.0, 2*M_PI);
    ArrayXd d(numwires);

    for(int i = 0; i < numwires; i++){
        d(i) = distribution(generator);
    }
    sparam = (2*(d.cos().square()) - 1).sum()/numwires;
    return d;
}

// Function to generate random (x,y) positions
ArrayXXd create_random_positions(int numwires, double sampleDimension, unsigned seed){
    std::default_random_engine generator(seed);
    std::uniform_real_distribution<double> distribution(0.0, sampleDimension);
    ArrayXXd d(numwires, 2);

    for(int i = 0; i < numwires; i++){
        d(i,0) = distribution(generator);
        d(i,1) = distribution(generator);
    }
    return d;
}

// Function to generate random nanowire lengths using normal distribution
ArrayXd create_random_lengths(int numwires, double lav, double lstd, unsigned seed){
    std::default_random_engine generator(seed);
    std::normal_distribution<double> distribution(lav, lstd);
    ArrayXd d(numwires);

    for(int i = 0; i < numwires; i++){
        d(i) = std::abs(distribution(generator));
    }
    return d;
}

// Class definition of nanowire network array object
class WireNet{
public:
    WireNet(int, double, double, double, double, double, double, unsigned, bool);
    std::vector<std::string> parameterList;

    double areal_coverage(double);
    void parameters();
    void indexwires();
    void intersections();
    void solve();
    void output_files(std::string);

    VectorXd eigenvalues;
    MatrixXd eigenvectors;
    MatrixXd laplacian;
    MatrixXd cmatrix;

    ArrayXXd nodeCoords;
    ArrayXXi nodeWireIndices;
    int number_of_nodes;
    bool intersections_calculated;
    bool solve_calculated;

private:
    void conductance_matrix();
    int numwires;
    double lav;
    double lstd;
    double sampleDimension;
    double wireRes;
    double intersectRes;
    double sheetRes;
    double wireDensity;
    float sparam;
    unsigned seed;
    ArrayXd wireLengths;
    ArrayXXd startCoords;
    ArrayXXd endCoords;
    ArrayXd wireAngles;
    ArrayXXd allxCoords;
};

// Class constructor
WireNet::WireNet(int n, double l, double st, double d, double wr, double ir, double rsh, unsigned s, bool debug = false){
    numwires = n;
    lav = l;
    lstd = st;
    sampleDimension = d;

    wireRes = wr;
    intersectRes = ir;
    sheetRes = rsh;

    wireDensity = numwires/(sampleDimension*sampleDimension);

    if(debug == true){
        seed = 293423;
    }else{
        seed = s;
    }

    wireLengths = create_random_lengths(numwires, lav, lstd, seed);
    startCoords = create_random_positions(numwires, sampleDimension, seed);
    wireAngles = create_random_angles(numwires, sparam, seed);
    endCoords.resize(numwires, 2);
    endCoords << wireLengths*wireAngles.sin(), wireLengths*wireAngles.cos();
    endCoords = endCoords + startCoords;
    allxCoords.resize(numwires, 2);
    allxCoords << startCoords.col(0), endCoords.col(0);

    nodeCoords.resize(1e7*wireDensity*wireDensity + 2*numwires, 2);      // Initial node arrays are larger than the number of wires to make sure there is enough space
    nodeWireIndices.resize(1e7*wireDensity*wireDensity + 2*numwires, 2);
    intersections_calculated = false;
    solve_calculated = false;

    using std::to_string;
    parameterList.push_back("Number of wires: " + to_string(numwires));
    parameterList.push_back("Wire lengths: " + to_string(lav) + " with standard deviation: " + to_string(lstd));
    parameterList.push_back("Sample size: " + to_string(sampleDimension));
    parameterList.push_back("Wire areal density: " + to_string(wireDensity));
    parameterList.push_back("Calculated S-parameter: " + to_string(sparam));
    parameterList.push_back("Wire resistance per length: " + to_string(wireRes));
    parameterList.push_back("Wire interconnect resistance: " + to_string(intersectRes));
    parameterList.push_back("Matrix sheet resistance: " + to_string(sheetRes));
    parameterList.push_back("Random seed: " + to_string(seed));
}

// Calculates areal coverage of nanowires
double WireNet::areal_coverage(double wirediameter){
    return (wireLengths*wirediameter).sum()/(sampleDimension*sampleDimension);
}

// Displays the parameters for the WireNet
void WireNet::parameters(){
    m.lock();
    for(int i = 0; i < parameterList.size(); i++){
        cout << parameterList[i] << endl;
    }
    m.unlock();
}

// Lists all wires sorted by their wire index
void WireNet::indexwires(){
    for(int i = 0; i < numwires; i++){
        cout << "[" << i << "]  "
             << startCoords(i,0) << ", " << startCoords(i,1) << "     "
             << endCoords(i,0) << ", " << endCoords(i,1) << "     L = "
             << wireLengths(i) << endl;
    }
}

// Calculates all nodes (intersections and wire end points). Returns coordinates, wire indices and node count.
void WireNet::intersections(){
    if(intersections_calculated == true){
        m.lock();
            cout << "Intersections already calculated!";
        m.unlock();
        return;
    }
    ArrayXd xs_st = allxCoords.rowwise().minCoeff();        // Lowest x coordinate of wire i.e. start of wire
    ArrayXd xs_en = allxCoords.rowwise().maxCoeff();        // Highest x coordinate of wire i.e. end of wire
    ArrayXd xs_all(2*numwires);                             // Combined list of all x coordinates (first half of list is start coordinates, second is end coordinates)
    xs_all << xs_st, xs_en;
    ArrayXi i_xs_sort = argsort(xs_all);

    std::list<int> searchList;                    // Search list for temp storage of x indices

    m.lock();
        cout << "[" << std::this_thread::get_id() << "] Finding intersections...\n";
    m.unlock();

    int node_count = 0;

    for(int j = 0; j < 2*numwires; j++){
        int i_wireA = i_xs_sort(j);

        if(i_wireA < numwires){
            // First include wire start point as additional node
            nodeCoords.row(node_count) = startCoords.row(i_wireA);
            nodeWireIndices(node_count, 0) = i_wireA;
            nodeWireIndices(node_count, 1) = i_wireA;
            node_count++;

            // Starts of wires
            if(searchList.empty()){
                searchList.push_back(i_wireA);   // Adds current wire index to searchList if empty and moves to next wire
                continue;
            }
            for(std::list<int>::iterator i = searchList.begin(); i != searchList.end(); ++i){
                int i_wireB = *i;
                bool intersect = false;
                Vector2d st_A = startCoords.row(i_wireA);
                Vector2d en_A = endCoords.row(i_wireA);
                Vector2d st_B = startCoords.row(i_wireB);
                Vector2d en_B = endCoords.row(i_wireB);
                Vector2d intersectxy;
                intersectCheck(st_A, en_A, st_B, en_B, intersectxy, intersect);   // Checks for intersection between wireA and wireB
                if(intersect == true){
                    nodeCoords.row(node_count) = intersectxy;
                    nodeWireIndices(node_count, 0) = i_wireA;
                    nodeWireIndices(node_count, 1) = i_wireB;
                    node_count++;
                }
            }
            searchList.push_back(i_wireA);       // Adds current wire index to searchList
        }else{
            // First include wire end point as additional node
            nodeCoords.row(node_count) = endCoords.row(i_wireA - numwires);
            nodeWireIndices(node_count, 0) = i_wireA - numwires;
            nodeWireIndices(node_count, 1) = i_wireA - numwires;
            node_count++;

            // Ends of wires
            searchList.remove(i_wireA - numwires);     // Removes current wire index from searchList
        }
    }
    nodeCoords.conservativeResize(node_count, Eigen::NoChange);
    nodeWireIndices.conservativeResize(node_count, Eigen::NoChange);
    intersections_calculated = true;
    m.lock();
        cout << "[" << std::this_thread::get_id() 
             << "] There are " << node_count 
             << " nodes from " << numwires 
             << " wires of which " << node_count - 2*numwires 
             << " are intersections " << endl;
    m.unlock();

    number_of_nodes = node_count;
}

// Calculates the adjacency conductance matrix
void WireNet::conductance_matrix(){
    if(intersections_calculated == false){
        intersections();
    }

    m.lock();
        cout << "[" << std::this_thread::get_id() << "] Calculating conductance matrix..." << endl;
    m.unlock();

    cmatrix = MatrixXd::Ones(number_of_nodes, number_of_nodes) - MatrixXd::Identity(number_of_nodes, number_of_nodes);
    cmatrix *= 1.0/(sheetRes*0.5*number_of_nodes);

    for(int w = 0; w < numwires; w++){
        ArrayXi i_inter(100);            // Array of node indices that are on wire w
        ArrayXXd xy_inter(100, 2);       // Array of x,y positions of nodes on wire w
        int online_count = 0;
        // Finds nodes for wire w
        for(int i = 0; i < number_of_nodes; i++){
            if((nodeWireIndices(i, 0) == w) || (nodeWireIndices(i, 1) == w)){
                i_inter(online_count) = i;
                xy_inter.row(online_count) = nodeCoords.row(i);
                online_count++;
            }
        }
        i_inter.conservativeResize(online_count);
        xy_inter.conservativeResize(online_count, Eigen::NoChange);
        ArrayXd x = xy_inter.col(0);
        ArrayXi xy_inter_sort = argsort(x);              // Sort nodes by x-coordinate

        for(int k = 0; k < i_inter.size() - 1; k++){
            double dx = xy_inter(xy_inter_sort(k), 0) - xy_inter(xy_inter_sort(k+1), 0);
            double dy = xy_inter(xy_inter_sort(k), 1) - xy_inter(xy_inter_sort(k+1), 1);
            double intLength = std::sqrt(std::abs(dx*dx + dy*dy));     // Distance between adjacent nodes (k and k+1)
            if(intLength == 0){
                continue;
            }
            double c = 1.0/(intLength*wireRes + intersectRes);         // Conductance between node k and k+1
            // Update conductance matrix (Hermitian i.e. c_ij=c_ji)
            cmatrix(i_inter(xy_inter_sort(k)), i_inter(xy_inter_sort(k+1))) = c;
            cmatrix(i_inter(xy_inter_sort(k+1)), i_inter(xy_inter_sort(k))) = c;
        }
    }
}

// Solves the resistor network
void WireNet::solve(){
    if(solve_calculated == true){
        m.lock();
            cout << "Already solved!" << endl;
        m.unlock();
        return;
    }
    conductance_matrix();
    VectorXd c_i = cmatrix.rowwise().sum();
    laplacian = c_i.asDiagonal();
    laplacian -= cmatrix;

    m.lock();
        cout << "[" << std::this_thread::get_id() << "] Solving..." << endl;
    m.unlock();

    Eigen::SelfAdjointEigenSolver<MatrixXd> eigensolver(laplacian);
    if(eigensolver.info() != Eigen::Success) abort();

    m.lock();
        cout << "[" << std::this_thread::get_id() << "] Saving eigenvalues and eigenvctors..." << endl;
    m.unlock();

    eigenvalues = eigensolver.eigenvalues();
    eigenvectors = eigensolver.eigenvectors();

    m.lock();
        cout << "[" << std::this_thread::get_id() << "] Done" << endl;
    m.unlock();

    solve_calculated = true;
}

// Output files for resistor network
void WireNet::output_files(std::string filename){
    if(solve_calculated == false){
        m.lock();
            cout << "Network nor solved!" << endl;
        m.unlock();
        return;
    }
    m.lock();
        cout << "[" << filename << "] Writing files out..." << endl;
        std::ofstream outfile;

        outfile.open(filename + "_eigvals.dat");
        for(int i = 0; i < parameterList.size(); i++){
            outfile << parameterList[i] << '\n';
        }
        outfile << "EIGENVALUES\n";
        for(int i = 0; i < eigenvalues.size(); i++){
            outfile << eigenvalues(i) << '\n';
        }
        outfile.close();

        outfile.open(filename + "_eigvecs.dat");
        for(int i = 0; i < parameterList.size(); i++){
            outfile << parameterList[i] << '\n';
        }
        outfile << "EIGENVECTORS\n";
        for(int i = 0; i < eigenvectors.rows(); i++){
            for(int j = 0; j < eigenvectors.cols(); j++){
                outfile << eigenvectors(i,j) << '\t';
            }
            outfile << '\n';
        }
        outfile.close();

        outfile.open(filename + "_cmatrix.dat");
        for(int i = 0; i < parameterList.size(); i++){
            outfile << parameterList[i] << '\n';
        }
        outfile << "CONDUCTANCE MATRIX\n";
        for(int i = 0; i < cmatrix.rows(); i++){
            for(int j = 0; j < cmatrix.cols(); j++){
                outfile << cmatrix(i,j) << '\t';
            }
            outfile << '\n';
        }
        outfile.close();
    m.unlock();
}

#endif
