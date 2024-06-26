/*
C++ Program prepared by Ronn Vincent D. Pongot
for the University of the Philippines - Institute of Civil Engineering

CE 199 - Undergraduate Research Project (2nd Semester, AY 2023-2024)
Advised by Dr. Pher Errol Quinay
*/

/*
June 26, 2024
*/

#include <iostream>
#include <cmath>
#include <algorithm>
#include <string>
#include <sstream>
#include <thread>
#include <iomanip>

//DEVELOPER NOTE: compiled with eigen 3.4.0 for all configurations
//compiled with /bigobj (use command line in project properties C/C++)
//compiled with open MP support: yes
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>

#include <fstream>
#include <ctime>
#include <omp.h>

const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");

typedef Eigen::Triplet<double> T;
typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SpMat_Parallel;

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;

class Force {
public:
	double x_coord;
	double y_coord;
	double z_coord;
	double force_magnitude;
	int force_dof; //legend: 0x 1y 2z 3tx 4ty 5tz
};

class DistributedLoad {
public:
	double x_coord1;
	double x_coord2;
	double y_coord1;
	double y_coord2;
	double z_coord1;
	double z_coord2;
	double force_magnitude; // N/mm^3
	int force_dof;
	int triangle; // (triangle = 1: 0 to w along +x'; triangle = 2: w to 0 along +x')
};

class SelectionPoint {
public:
	double x_coord;
	double y_coord;
	double z_coord;
	int dof; //legend: 0x 1y 2z 3tx 4ty 5tz
};

class Selection {
public:
	double x_coord1;
	double x_coord2;
	double y_coord1;
	double y_coord2;
	double z_coord1;
	double z_coord2;
	int dof;
};

class Reinforcement {
public:
	double x_coord1;
	double x_coord2;
	double y_coord1;
	double y_coord2;
	double z_coord1;
	double z_coord2;
	double bar_size; //bar size is only considered along the spacing direction, you have to manually enter the other dimension (e.g. z1 -> z2) to match the bar size (overshooting the bounds works too)
	int run_direction; // 0 - x, 1 - y, 2 - z
	int spacing_direction; //direction where you traverse the spacing of bars i.e., [  O <----> O <----> O <----> O ]      ----> +x axis

	double number_of_bars;
	double spacing;

	//note: specify minx, miny, minz of the area bounded by all rebars in a layer and knowing the spacing, it will automatically populate the box with rebars
	//DEVELOPER NOTE: this treats every rebar as rectangular
};

class InterfaceElements {
public:
	vector<int> elements;

	int spring_num;
	double spring_num_per_area;

	double x_coord1;
	double x_coord2;
	double y_coord1;
	double y_coord2;
	double z_coord1;
	double z_coord2;

	int axis_of_connection;
	int spring_material_group;

	double element_bottom_coord;
	double element_top_coord;
};

void support_restrainer(Eigen::Ref<MatrixXd> element, int total_ele_num, vector<SelectionPoint> selection_description, int selection_count);

vector<int> reinforcement_tagger(Reinforcement layer_pass, MatrixXd element, int total_ele_num, MatrixXd spring, int total_spring_num, MatrixXd geometry, int choice);
vector<int> material_tagger(Selection bounds, MatrixXd element, int total_ele_num, MatrixXd spring, int total_spring_num, MatrixXd geometry);

int element_finder(MatrixXd element, int total_ele_num, double x, double y, double z);

void print_selection_point(SelectionPoint& p);
vector<SelectionPoint> selection_processor(Selection selection, MatrixXd element, int total_ele_num);
VectorXi dof_finder(vector<SelectionPoint> selection_description, int selection_count, MatrixXd element, int total_ele_num);

void print_force(Force& p);
vector<Force> distributed_load_processor(DistributedLoad distributed_load, MatrixXd geometry, MatrixXd element, int total_ele_num);
MatrixXd force_dof_finder(vector<Force> force_description, int load_count, MatrixXd element, int total_ele_num);
void force_applier(Eigen::Ref<VectorXd> F, MatrixXd dofs_to_apply, int dofs_to_apply_count, int load_increment_count, int load_increment);

VectorXd rotation_to_displacement(double Ax, double Ay, double Az, double thetax, double thetay, double thetaz);
MatrixXd transformation_matrix(double end1x, double end1y, double end1z, double end2x, double end2y, double end2z, int choice);

MatrixXd localmatrix(double Qx, double Qy, double Qz, double x1, double y1, double z1, double x2, double y2, double z2);

void element_drawer(string name, MatrixXd element, int total_ele_num, VectorXd U, VectorXd F, MatrixXd element_principal_stresses);
void spring_drawer(string name, MatrixXd spring, int total_spring_num, int choice_material, int choice_condition);

//-----------------------------------------------------------------------------

int main() {

	cout << "C++ Program prepared by Ronn Vincent D. Pongot" << endl;
	cout << "for the University of the Philippines - Institute of Civil Engineering" << endl;
	cout << "CE 199 - Undergraduate Research Project (2nd Semester, AY 2023 - 2024)" << endl;
	cout << "Advised by Dr.Pher Errol Quinay" << endl;

	//-----------------------------------------------------------------------------

	/*
	NOTICE:
	1. All forces in N (Newtons). All measurements in mm (millimeters).
	2. z axis is vertical (not a strict rule)
	3. Current build is for analysis up to nonlinear 3D static only.
	*/

	/*

	--- 1. INPUT ----

	*/

	//-----------------------------------------------------------------------------

	int time_start_master = time(NULL);

	//-----------------------------------------------------------------------------

	string default_name = "ResultsFile_ProgramID-INPUT_TimeID-" + to_string(time_start_master);

	//-----------------------------------------------------------------------------

	//READ THE MODEL PARAMETERS

	//----------------------------------------------------------------------------- 

	string read_filename = "model_parameters.txt";

	//-----------------------------------------------------------------------------

	//Consider the future feature of importing geometry from different sources?

	//-----------------------------------------------------------------------------

	//copy the model_parameters used in this run

	string line_copy1;
	ifstream ini_file{ read_filename };
	ofstream out_file{ default_name + "_" + read_filename };
	if (ini_file && out_file) {
		while (getline(ini_file, line_copy1)) {
			out_file << line_copy1 << "\n";
		}
	}
	else {
		cout << "\n\nERROR: Unable to open " << read_filename << ".\n";
		std::cout << "\n\nPress ENTER to exit the program: ";
		std::cin.get();
		return 1;
	}
	ini_file.close();
	out_file.close();

	//-----------------------------------------------------------------------------

	string read_start = "START";
	string read_end = "END";

	//-----------------------------------------------------------------------------

	//recognized commands for model_parameters
	string read_material = "MATERIAL"; // 0
	string read_geometry = "PRIMARY_GEOMETRY"; // 1
	string read_secondary_geometry = "SECONDARY_GEOMETRY"; // 2 
	string read_reinforcement_bars = "REINFORCEMENT_BARS"; // 3 
	string read_stirrups = "STIRRUPS"; // 4 
	string read_interface = "INTERFACE"; // 5
	string read_analyze = "ANALYZE"; // 6
	string read_support = "SUPPORT"; // 7
	string read_point_load = "POINT_LOAD"; // 8
	string read_load = "DIST_LOAD"; // 9
	string read_self_weight = "SELF_WEIGHT"; // 10

	vector<string> read_commands;

	read_commands.push_back(read_material);
	read_commands.push_back(read_geometry);
	read_commands.push_back(read_secondary_geometry);
	read_commands.push_back(read_reinforcement_bars);
	read_commands.push_back(read_stirrups);
	read_commands.push_back(read_interface);
	read_commands.push_back(read_analyze);
	read_commands.push_back(read_support);
	read_commands.push_back(read_point_load);
	read_commands.push_back(read_load);
	read_commands.push_back(read_self_weight);

	vector<vector<double>> input_data;
	//outer vector - lists all the entries as groups of commands (e.g. row1 MATERIAL, row2 GEOMETRY, row3 SUPPORT)
	//inner vector - lists all data gathered in the format [ state_index(which command), data1, data2, data3, ... ]

	vector<int> state_tracker; //lists in order the commands that were read, handling the duplicates and skipped entries
	VectorXi state_check = VectorXi::Zero(11); //light switch where only one entry is at "1" at any time if the pointer is inside the reading block
	VectorXi complete_check = VectorXi::Zero(11); //permanent

	if (auto file = ifstream{ read_filename }) {

		bool reading_now = false;
		bool inside_command = false;
		int current_state = 0;

		for (string line; getline(file, line);) {

			//check if the START command has been read
			if ((line.find(read_start)) != string::npos) {
				reading_now = true;
			}

			if (reading_now) {
				if (!inside_command) {
					for (int i = 0; i < read_commands.size(); i++) {
						if ((line.find(read_commands[i])) != string::npos) {
							current_state = i;
							state_check(i) = 1;
							complete_check(i) = 1;
							inside_command = true;
							//no break here, so if successive commands are called in the same line accidentally, the last one will be followed
						}
					}
				}

				//if inside a recognized command
				if (inside_command) {

					vector<double> data_collected;
					state_tracker.push_back(current_state);
					data_collected.push_back(current_state);

					for (string data_read; getline(file, data_read);) {
						if ((data_read.find('=')) != string::npos) {

							double data_point;

							//turn the string line into a stream
							istringstream iss(data_read);

							//skip all until equals sign
							getline(iss, data_read, '=');

							iss >> data_point;
							data_collected.push_back(data_point);
						}

						if ((data_read.find('}')) != string::npos) {

							if (data_collected.size() > 1) {
								//summarize the data points collected
								input_data.push_back(data_collected);
							}

							//exit the command state
							inside_command = false;

							//turn off the light
							state_check[current_state] = 0;

							//stop it from reading under the command state
							break;
						}

					}
				}

				if ((line.find(read_end)) != string::npos) {
					reading_now = false;
					break;
				}
			}
		}
	}
	else {
		cout << "\n\nERROR: Unable to open " << read_filename << ".\n";
		std::cout << "\n\nPress ENTER to exit the program: ";
		std::cin.get();
		return 1;
	}

	if (complete_check[0] == 0 || complete_check[1] == 0) {
		cout << "\n\nERROR: Not enough data given.\n\n";
		std::cin.get();
		return 1;
	}

	cout << endl << "-------------------------------------------------------------\n" << "\nINPUT DATA READ\n\n";

	//printer
	for (int i = 0; i < input_data.size(); i++) {
		for (auto it = input_data[i].begin(); it != input_data[i].end(); it++) {
			std::cout << *it << " ";
		}
		std::cout << endl;
	}
	std::cout << endl;

	//----------------------------------------------------------------------------- 

	//READ THE PROGRAM PARAMETERS

	//----------------------------------------------------------------------------- 

	string read_filename_program = "program_parameters.txt";

	//----------------------------------------------------------------------------- 

	//copy the file
	string line_copy2;
	ifstream ini_file_program{ read_filename_program };
	ofstream out_file_program{ default_name + "_" + read_filename_program };
	if (ini_file_program && out_file_program) {
		while (getline(ini_file_program, line_copy2)) {
			out_file_program << line_copy2 << "\n";
		}
	}
	else {
		cout << "\n\nERROR: Unable to open " << read_filename_program << "\n";
		std::cout << "\n\nPress ENTER to exit the program: ";
		std::cin.get();
		return 1;
	}
	ini_file_program.close();
	out_file_program.close();

	//-----------------------------------------------------------------------------

	//recognized commands
	string read_parameters = "PARAMETERS"; // 0
	string read_print = "PRINT"; // 1
	string read_automate = "AUTOMATE"; //2
	string read_special = "SPECIAL"; //3

	vector<string> read_commands_program;

	read_commands_program.push_back(read_parameters);
	read_commands_program.push_back(read_print);
	read_commands_program.push_back(read_automate);
	read_commands_program.push_back(read_special);

	vector<vector<double>> input_data_program;
	//outer vector - lists all the entries as groups of commands (e.g. row1 PARAMETERS, row2 PRINT)
	//inner vector - lists all data gathered in the format [ state_index(which command), data1, data2, data3, ... ]

	vector<int> state_tracker_program; //lists in order the commands that were read, handling the duplicates and skipped entries
	VectorXi state_check_program = VectorXi::Zero(11); //light switch where only one entry is at "1" at any time if the pointer is inside the reading block
	VectorXi complete_check_program = VectorXi::Zero(11); //permanent

	if (auto file_program = ifstream{ read_filename_program }) {

		bool reading_now = false;
		bool inside_command = false;
		int current_state = 0;

		for (string line; getline(file_program, line);) {

			//check if the START command has been read
			if ((line.find(read_start)) != string::npos) {
				reading_now = true;
			}

			if (reading_now) {
				if (!inside_command) {
					for (int i = 0; i < read_commands_program.size(); i++) {
						if ((line.find(read_commands_program[i])) != string::npos) {
							current_state = i;
							state_check_program(i) = 1;
							complete_check_program(i) = 1;
							inside_command = true;
							//no break here, so if successive commands are called in the same line accidentally, the last one will be followed
						}
					}
				}

				//if inside a recognized command
				if (inside_command) {

					vector<double> data_collected;
					state_tracker_program.push_back(current_state);
					data_collected.push_back(current_state);

					for (string data_read; getline(file_program, data_read);) {
						if ((data_read.find('=')) != string::npos) {

							double data_point;

							//turn the string line into a stream
							istringstream iss(data_read);

							//skip all until equals sign
							getline(iss, data_read, '=');

							iss >> data_point;
							data_collected.push_back(data_point);
						}

						if ((data_read.find('}')) != string::npos) {

							if (data_collected.size() > 1) {
								//summarize the data points collected
								input_data_program.push_back(data_collected);
							}

							//exit the command state
							inside_command = false;

							//turn off the light
							state_check_program[current_state] = 0;

							//stop it from reading under the command state
							break;
						}

					}
				}

				if ((line.find(read_end)) != string::npos) {
					reading_now = false;
					break;
				}
			}
		}
	}
	else {
		cout << "\n\nERROR: Unable to open " << read_filename_program << ".\n";
		std::cout << "\n\nPress ENTER to exit the program: ";
		std::cin.get();
		return 1;
	}

	if (complete_check_program[0] == 0 || complete_check_program[1] == 0) {
		cout << "\n\nERROR: Not enough data given.\n\n";
	}

	//printer
	for (int i = 0; i < input_data_program.size(); i++) {
		for (auto it = input_data_program[i].begin(); it != input_data_program[i].end(); it++) {
			std::cout << *it << " ";
		}
		std::cout << endl;
	}

	std::cout << endl;

	//-----------------------------------------------------------------------------

	//get the program parameters

	double load_increment_count_input = 1;
	double load_increment_start_input = 1;
	double num_threads_input = 1;
	double matrix_solver_input = 0;
	double allow_failure_input = 1;
	double RAM_safety_override_input = 0;

	int load_increment_count = 1;
	int load_increment_start = 1;
	int num_threads_actual = 1;
	int matrix_solver = 0;
	int allow_failure = 1;
	int RAM_safety_override = 0;

	//for automate

	int automate_check = 0;
	int automate_start = 0;
	int automate_end = 1;
	int automate_step = 1; //this will be subtracted by 1 at the end of the automate loop to truly capture the automate_step = 1 -> 1 step per loop.

	//for special
	bool peak_load_increment_check = false;
	int peak_load_increment_input = 1;
	int peak_load_increment = 1;
	bool crack_reset_check = false;
	int crack_reset_increment_input = 1;
	int crack_reset_increment = 1;

	vector<int> automate_axis{ 0,0,0 };


	vector<int> print_list;

	for (int i = 0; i < state_tracker_program.size(); i++) {
		if (state_tracker_program[i] == 0) { //PARAMETERS command
			vector<double> parameters_now = input_data_program[i];

			if (parameters_now.size() != 7) {
				parameters_now.resize(7);
			}

			/*
			user input vector legend:
			0 - current state (parameters = 0)
			1 - load increment count
			2 - load increment start
			3 - number of threads
			4 - matrix solver
			5 - allow cracks?
			6 - RAM_safety_override
			*/

			load_increment_count_input = parameters_now[1];
			load_increment_start_input = (int)fabs(parameters_now[2]);
			num_threads_input = (int)parameters_now[3];
			matrix_solver_input = (int)parameters_now[4];
			allow_failure_input = parameters_now[5];

			if (parameters_now[6] == 1) {
				RAM_safety_override_input = parameters_now[6];
			}
			if (RAM_safety_override_input == 1) {
				RAM_safety_override = RAM_safety_override_input;
			}

			load_increment_count = (int)fabs(load_increment_count_input);
			load_increment_start = (int)fabs(load_increment_start_input);

			int num_threads_max = thread::hardware_concurrency(); //get max

			if (num_threads_input > num_threads_max) {
				std::cout << "\nDesired number of threads " << num_threads_input << " is greater than maximum capable " << num_threads_max << ".";

				std::cout << "\nEnter 0 to continue with maximum capable threads, enter 1 to exit the program: ";

				int choice_thread;
				cin >> choice_thread;

				if (choice_thread == 1) {
					return 1;
				}

				num_threads_input = num_threads_max;

			}

			if (num_threads_input > 0) {
				num_threads_actual = num_threads_input;
			}

			matrix_solver = (int)fabs(matrix_solver_input);
			allow_failure = (int)fabs(allow_failure_input);
		}

		if (state_tracker_program[i] == 1) { //PRINT command
			vector<double> print_now = input_data_program[i];

			if (print_now.size() < 19) {
				print_now.resize(19);
			}
			else if (((print_now.size() - 19) % 2) == 1) { //make sure that the specific springs are in pairs
				print_now.push_back(0);
			}

			/*
			user input vector legend:
			0 - current state (print = 1)
			1 - troubleshooting.txt
			2 - pdof.csv
			3 - fdof.csv
			4 - support.csv
			5 - force.csv
			6 - force2.csv
			7 - element_initial.csv
			8 - spring.csv
			9 - Kff.txt
			10 - Kpf.txt
			11 - U.csv
			12 - F.csv
			13 - element.csv
			14 - spring.csv
			15 - spring_cracked.csv
			16 - validation.csv
			17 - element.vtk
			18 - spring.vtk
			19->n - specific_springs.vtk

			*/

			for (int print_iterant = 0; print_iterant < 19; print_iterant++) {
				if (print_now[print_iterant] != 1) { //any non-1 value becomes 0 (all non-yes will be no)
					//the for loop does not change the 19->n entries
					print_now[print_iterant] = 0;
				}
				print_list.push_back(print_now[print_iterant]);

			}
			for (int print_iterant = 19; print_iterant < print_now.size(); print_iterant++) {

				print_list.push_back(print_now[print_iterant]);
			}

		}

		if (state_tracker_program[i] == 2) { //AUTOMATE command
			vector<double> automate_now = input_data_program[i];

			if (automate_now.size() < 6) {
				automate_now.resize(6);
			}

			/*
			user input vector legend:
			0 - current state (automate = 2)
			1 - automate the program?
			2 - vary the number of elements along which axis (overrides the geometry input) ?
			3 - starting number of elements
			4 - end number of elements
			5 - step count
			*/


			if (automate_now[1] == 1) {
				automate_check = 1;
			}

			if (automate_check == 1) {

				if ((automate_now[2] >= 0) && (automate_now[2] <= 2)) {
					int axis_to_automate = (int)automate_now[2];
					automate_axis[axis_to_automate] = 1; // yes or no value
				}

				if (automate_now[3] >= 1) {
					automate_start = automate_now[3];
				}

				if (((int)automate_now[4] > automate_start)) {
					automate_end = automate_now[4];
				}

				if (automate_now[5] >= 1) {
					automate_step = automate_now[5];
				}
			}
		}

		if (state_tracker_program[i] == 3) { //SPECIAL command
			vector<double> special_now = input_data_program[i];

			if (special_now.size() < 5) {
				special_now.resize(5);
			}

			/*
			user input vector legend:
			0 - current state (special = 3)
			1 - perform load to deloading after custom load_increment (0 - no, 1 - yes) ?
			2 - peak_load_increment
			3 - perform crack reset (0 - no, 1 - yes)?
			4 - crack_reset_increment
			*/


			if (special_now[1] == 1) {
				peak_load_increment_check = true;
			}

			if (peak_load_increment_check) {
				peak_load_increment_input = special_now[2];

				if ((peak_load_increment_input > 0) && (peak_load_increment_input <= load_increment_count)) {
					peak_load_increment = peak_load_increment_input;
				}
				else {
					peak_load_increment = load_increment_count;
				}
			}

			if (special_now[3] == 1) {
				crack_reset_check = true;
			}

			if (crack_reset_check) {
				crack_reset_increment_input = special_now[4];

				if ((crack_reset_increment_input > 0) && (crack_reset_increment_input <= load_increment_count)) {
					crack_reset_increment = crack_reset_increment_input;
				}
				else {
					crack_reset_increment = load_increment_count;
				}
			}
		}
		//this line of code is very important
		if (!peak_load_increment_check) {
			peak_load_increment = load_increment_count;
		}
	}

	//note: this int automate handles the number of FULL RUNS, the load increments are further down

	if (print_list[16] == 1) { //print analyze.csv headers

		string filename_analyze = "ResultsFile_TimeID-" + to_string(time_start_master) + "_ANALYZE.csv";
		ofstream file_analyze;
		file_analyze.open(filename_analyze, ofstream::app);
		if (file_analyze.is_open()) {
			//ProgramID, LoadIncrement, element number, x, y, z, dof_num, U, F
			file_analyze << "Program_ID,Load_Increment,element_number,x,y,z,dof_number,U,F\n";
			file_analyze.close();
		}
		else {
			std::cout << "\a\n\nERROR: Failed to write to file!\n";
		}

	}

	//-----------------------------------------------------------------------------

	string filename_runtime = "ResultsFile_TimeID-" + to_string(time_start_master) + "_RUNTIME.csv";

	ofstream file_runtime;
	file_runtime.open(filename_runtime, ofstream::app);
	if (file_runtime.is_open()) {
		file_runtime << "PROGRAM_ID,LOAD_INCREMENT,RUNTIME\n";
		file_runtime.close();
	}
	else {
		std::cout << "\a\n\nERROR: Failed to write to file!\n";
	}

	//-----------------------------------------------------------------------------

	for (int automate = automate_start; automate < automate_end; automate++) {

		int time_start = time(NULL);

		std::ostringstream automate_step_string;
		automate_step_string << std::setw(5) << std::setfill('0') << automate;
		default_name = "ResultsFile_ProgramID-" + automate_step_string.str() + "_TimeID-" + to_string(time_start_master);

		//-----------------------------------------------------------------------------

		std::cout << "Preparing the materials...\n";

		int material_group_num = 0;

		for (int i = 0; i < state_tracker.size(); i++) {
			if (state_tracker[i] == 0) { //check if the data input is from command MATERIAL
				material_group_num++;
			}

		}

		MatrixXd material = MatrixXd::Zero(material_group_num, 7);

		/*
		LEGEND
		0 - E
		1 - nu
		2 - G
		3 - compressive yield or ultimate strength
		4 - tensile yield or ultimate strength
		5 - brittle or ductile
		6 - special nonlinear implementation (0 - concrete, 1 - steel, 2 - generic, 3 - support)
		*/

		int material_counter = 0;

		for (int i = 0; i < state_tracker.size(); i++) {
			if (state_tracker[i] == 0) { //check if the data input is from command MATERIAL
				vector<double> material_now = input_data[i];

				if (material_now.size() != 7) {
					material_now.resize(7);
				}

				/*
				user input vector legend:
				0 - current state (material = 0)
				1 - E
				2 - nu
				3 - comp strength
				4 - tens strength
				5 - brittle or ductile
				6 - nonlinear implementation

				*/

				material(material_counter, 0) = material_now[1]; // E
				material(material_counter, 1) = material_now[2]; // nu
				material(material_counter, 2) = material_now[1] / (2 * (1 + material_now[2])); // G
				material(material_counter, 3) = fabs(material_now[3]);
				material(material_counter, 4) = fabs(material_now[4]);
				material(material_counter, 5) = (int)material_now[5];
				material(material_counter, 6) = (int)material_now[6];

				material_counter++;
			}

		}

		//-----------------------------------------------------------------------------


		//Setting up the geometry input

		int geometry_group_num = 0;

		for (int i = 0; i < state_tracker.size(); i++) {
			if (state_tracker[i] == 1) { //check if the data input is from command PRIMARY_GEOMETRY
				geometry_group_num++;
			}

		}

		MatrixXd geometry = MatrixXd::Zero(geometry_group_num, 13);

		/*
		LEGEND
		0 - x bound 1
		1 - x bound 2
		2 - y bound 1
		3 - y bound 2
		4 - z bound 1
		5 - z bound 2
		6 - number of elements along x
		7 - number of elements along y
		8 - number of elements along z
		9 - number of springs on x face (e.g., 4 springs -> x face has 16 springs in a 4 x 4 pattern)
		10 - number of springs along y
		11 - number of springs along z
		12 - material group
		*/

		//-----------------------------------------------------------------------------

		std::cout << "Materials prepared!\n\n";

		std::cout << "Preparing the geometry...\n";

		int geometry_counter = 0;

		for (int i = 0; i < state_tracker.size(); i++) {
			if (state_tracker[i] == 1) { //check if the data input is from command PRIMARY_GEOMETRY
				vector<double> geometry_now = input_data[i];

				if (geometry_now.size() != 14) {
					geometry_now.resize(14);
				}

				/*
				user input vector legend:
				0 - current state (PRIMARY_GEOMETRY = 1)
				1 - xbound1
				2 - xbound2
				3 - ybound1
				4 - ybound2
				5 - zbound1
				6 - zbound2
				7 - number of elements along x
				8 - number of elements along y
				9 - number of elements along z
				10 - spring num x
				11 - spring num y
				12 - spring num z
				13 - material group

				*/


				double x1 = min(geometry_now[1], geometry_now[2]);
				double x2 = max(geometry_now[1], geometry_now[2]);

				geometry(geometry_counter, 0) = x1; //xbound1
				geometry(geometry_counter, 1) = x2; //xbound2

				double y1 = min(geometry_now[3], geometry_now[4]);
				double y2 = max(geometry_now[3], geometry_now[4]);

				geometry(geometry_counter, 2) = y1; //ybound1
				geometry(geometry_counter, 3) = y2; //ybound2

				double z1 = min(geometry_now[5], geometry_now[6]);
				double z2 = max(geometry_now[5], geometry_now[6]);

				geometry(geometry_counter, 4) = z1; //zbound1
				geometry(geometry_counter, 5) = z2; //zbound2

				geometry(geometry_counter, 6) = (int)fabs(geometry_now[7]); //num ele along x
				geometry(geometry_counter, 7) = (int)fabs(geometry_now[8]); //num ele along y
				geometry(geometry_counter, 8) = (int)fabs(geometry_now[9]); //num ele along z


				if (automate_check == 1) {
					for (int j = 0; j < 3; j++) {
						if (automate_axis[j] == 1) {
							geometry(geometry_counter, j + 6) = automate; //override for convergence analysis
						}
					}
				}

				geometry(geometry_counter, 9) = (int)fabs(geometry_now[10]); //num springs on x face
				geometry(geometry_counter, 10) = (int)fabs(geometry_now[11]); //num springs on y face
				geometry(geometry_counter, 11) = (int)fabs(geometry_now[12]); //num springs on z face
				geometry(geometry_counter, 12) = (int)fabs(geometry_now[13]); //material group

				geometry_counter++;
			}

		}

		std::cout << "Geometry prepared!\n\n";

		//-----------------------------------------------------------------------------

		//count the total number of elements for each mesh

		int total_ele_num = 0;

		for (int i = 0; i < geometry_group_num; i++) {
			total_ele_num += geometry(i, 6) * geometry(i, 7) * geometry(i, 8);
		}

		int total_dof_num = total_ele_num * 6;

		if ((total_dof_num > 100000) && (RAM_safety_override != 1)) {
			cout << "\n\nNOTICE: You will be solving a K sparse matrix of approximate size [" << total_dof_num << " x " << total_dof_num << "].\nFor reference, 8GB RAM machines start to lag at 100k to 150k, and risk freezing at 200k and above due to 100% RAM usage and possibly insufficient memory for eigen SparseMatrix solver.\nTo continue, set the RAM_safety_override option in program_parameters.txt to 1. ";
			std::cout << "\n\nPress ENTER to exit the program: ";
			std::cin.get();
			return 1;
		}

		//-----------------------------------------------------------------------------

		//ELEMENT MATRIX
		std::cout << "Preparing the elements...\n";

		MatrixXd element = MatrixXd::Zero(total_ele_num, 22);

		/*

		LEGEND
		0 - element number
		1 - current x coordinate
		2 - current y coordinate
		3 - current z coordinate
		4 - current thetax
		5 - current thetay
		6 - current thetaz
		7 - a
		8 - b
		9 - c
		10 - current dummy vector x distance (along x)
		11 - dummy vector y distance
		12 - dummy vector z distance
		13 - dummy2 x (along y)
		14 - dummy2 y
		15 - dummy2 z
		16 - dummy3 x (along z)
		17 - dummy3 y
		18 - dummy3 z
		19 - geometry group
		20 - interface?
		21 - support? (1 = yes)

		*/

		int counter = 0;

		for (int i = 0; i < geometry_group_num; i++) {

			int elements_x = geometry(i, 6);
			int elements_y = geometry(i, 7);
			int elements_z = geometry(i, 8);

			double xdim = geometry(i, 1) - geometry(i, 0);
			double ydim = geometry(i, 3) - geometry(i, 2);
			double zdim = geometry(i, 5) - geometry(i, 4);

			double a = xdim / elements_x;
			double b = ydim / elements_y;
			double c = zdim / elements_z;

			double origin_x = geometry(i, 0);
			double origin_y = geometry(i, 2);
			double origin_z = geometry(i, 4);

			for (int z = 0; z < elements_z; z++) {
				for (int y = 0; y < elements_y; y++) {
					for (int x = 0; x < elements_x; x++) {
						element(counter, 0) = counter;
						element(counter, 1) = (a / 2.0) + (a * x) + origin_x;
						element(counter, 2) = (b / 2.0) + (b * y) + origin_y;
						element(counter, 3) = (c / 2.0) + (c * z) + origin_z;

						//assume all elements have 0 thetax, 0 thetay, 0 thetaz
						//element(counter, 4) = 0;
						//element(counter, 5) = 0;
						//element(counter, 6) = 0;

						element(counter, 7) = a;
						element(counter, 8) = b;
						element(counter, 9) = c;

						//this assumes that all elements are aligned with the global x, y, z

						//track the normal vector of the +x face
						element(counter, 10) = (a / 2.0);
						//element(counter, 11) = 0;
						//element(counter, 12) = 0;

						//track the normal vector of the +y face
						//element(counter, 13) = 0;
						element(counter, 14) = (b / 2.0);
						//element(counter, 15) = 0;

						//track the normal vector of the +z face
						//element(counter, 16) = 0;
						//element(counter, 17) = 0;
						element(counter, 18) = (c / 2.0);

						//geometry group
						element(counter, 19) = i;

						//interface?
						//element(counter, 20) = 0; //no interface

						//support?
						//element(counter, 21) = 0;

						counter++;
					}
				}
			}
		}

		std::cout << "Interfaces prepared!\n\n";

		//-----------------------------------------------------------------------------

		//INTERFACING GEOMETRY GROUPS

		std::cout << "Preparing the interfaces...\n";

		//-----------------------------------------------------------------------------

		int interface_num = 0;

		for (int i = 0; i < state_tracker.size(); i++) {
			if (state_tracker[i] == 5) {
				interface_num++;
			}
		}

		//-----------------------------------------------------------------------------

		vector<InterfaceElements> interfaces;

		for (int i = 0; i < state_tracker.size(); i++) {
			if (state_tracker[i] == 5) {
				vector<double> interface_now = input_data[i];

				if (interface_now.size() != 4) {
					interface_now.resize(4);
				}

				/*

				user input vector legend:
				0 - current state (INTERFACE = 5)
				1 - PRIMARY_GEOMETRY Group 1
				2 - PRIMARY_GEOMETRY Group 2
				3 - MATERIAL group of springs to use

				*/

				InterfaceElements interface_object;

				int it1 = (int)interface_now[1];
				int it2 = (int)interface_now[2];

				if ((it1 < 0) || (it2 < 0)) {
					cout << "ERROR: Invalid interface data. Will not join the two desired meshes.\n";
					continue;
				}
				if ((it1 > geometry_group_num) || (it2 > geometry_group_num)) {
					cout << "ERROR: Invalid interface data. Will not join the two desired meshes.\n";
					continue;
				}

				interface_object.spring_material_group = 0; //default
				for (int spring_check = 0; spring_check < material_group_num; spring_check++) {
					if ((int)interface_now[3] == spring_check) {
						interface_object.spring_material_group = (int)interface_now[3];
					}
				}

				bool same_x_1 = geometry(it1, 1) == geometry(it2, 0);
				bool same_x_2 = geometry(it1, 0) == geometry(it2, 1);

				bool same_y_1 = geometry(it1, 3) == geometry(it2, 2);
				bool same_y_2 = geometry(it1, 2) == geometry(it2, 3);

				bool same_z_1 = geometry(it1, 5) == geometry(it2, 4);
				bool same_z_2 = geometry(it1, 4) == geometry(it2, 5);

				int axis_of_connection;
				bool connection_at_group1_top = 0;

				if (same_x_1 || same_x_2) {
					axis_of_connection = 0;
					if (same_x_1) { connection_at_group1_top = 1; }
				}
				else if (same_y_1 || same_y_2) {
					axis_of_connection = 1;
					if (same_y_1) { connection_at_group1_top = 1; }
				}
				else if (same_z_1 || same_z_2) {
					axis_of_connection = 2;
					if (same_z_1) { connection_at_group1_top = 1; }
				}
				else {
					std::cout << "\n\nUnable to find the connection of the two geometry groups: " << it1 << ", " << it2 << ".\n\n";
					continue;
				}

				vector<int> directions{ 0, 1, 2 };
				directions.erase(remove(directions.begin(), directions.end(), axis_of_connection), directions.end()); //erase-remove idiom

				int plane_direction_1 = directions[0];
				int plane_direction_2 = directions[1];

				//get range of plane

				//geometry indices for coords: x - 0, y - 2, z - 4
				int coord1_index = plane_direction_1 * 2;
				int coord2_index = plane_direction_2 * 2;

				double plane_normal;
				int group_at_top;
				int group_at_bottom;

				if (connection_at_group1_top) { //connection is at the top portion of the first mesh -> get the lower bound of the second mesh
					plane_normal = geometry(it1, axis_of_connection * 2 + 1);
					group_at_top = it2;
					group_at_bottom = it1;
				}
				else { //connection is at the bottom portion of the first mesh -> get the upper bound of the second mesh
					plane_normal = geometry(it2, axis_of_connection * 2 + 1);
					group_at_top = it1;
					group_at_bottom = it2;
				}

				//get the plane of intersection
				double plane_coord1_end1 = max(geometry(it1, coord1_index), geometry(it2, coord1_index));
				double plane_coord1_end2 = min(geometry(it1, coord1_index + 1), geometry(it2, coord1_index + 1));

				double plane_coord2_end1 = max(geometry(it1, coord2_index), geometry(it2, coord2_index));
				double plane_coord2_end2 = min(geometry(it1, coord2_index + 1), geometry(it2, coord2_index + 1));

				MatrixXd coordinates = MatrixXd::Zero(2, 3);
				//xbound1 ybound1 zbound1
				//xbound2 ybound2 zbound2

				coordinates(0, axis_of_connection) = plane_normal;
				coordinates(1, axis_of_connection) = plane_normal;

				coordinates(0, plane_direction_1) = plane_coord1_end1;
				coordinates(1, plane_direction_1) = plane_coord1_end2;

				coordinates(0, plane_direction_2) = plane_coord2_end1;
				coordinates(1, plane_direction_2) = plane_coord2_end2;

				interface_object.x_coord1 = coordinates(0, 0);
				interface_object.x_coord2 = coordinates(1, 0);

				interface_object.y_coord1 = coordinates(0, 1);
				interface_object.y_coord2 = coordinates(1, 1);

				interface_object.z_coord1 = coordinates(0, 2);
				interface_object.z_coord2 = coordinates(1, 2);

				interface_object.axis_of_connection = axis_of_connection;

				//springs: 9, 10, 11

				int spring_num_it1 = geometry(it1, axis_of_connection + 9);
				int spring_num_it2 = geometry(it2, axis_of_connection + 9);

				//get element dimensions

				MatrixXd element_dimensions = MatrixXd::Zero(2, 3);

				//element1
				element_dimensions(0, 0) = (geometry(it1, 1) - geometry(it1, 0)) / geometry(it1, 6);
				element_dimensions(0, 1) = (geometry(it1, 3) - geometry(it1, 2)) / geometry(it1, 7);
				element_dimensions(0, 2) = (geometry(it1, 5) - geometry(it1, 4)) / geometry(it1, 8);

				//element2
				element_dimensions(1, 0) = (geometry(it2, 1) - geometry(it2, 0)) / geometry(it2, 6);
				element_dimensions(1, 1) = (geometry(it2, 3) - geometry(it2, 2)) / geometry(it2, 7);
				element_dimensions(1, 2) = (geometry(it2, 5) - geometry(it2, 4)) / geometry(it2, 8);

				//divide by area of element face along plane

				double spring_num_it1_per_area = (spring_num_it1 * spring_num_it1) / (element_dimensions(0, plane_direction_1) * element_dimensions(0, plane_direction_2));
				double spring_num_it2_per_area = (spring_num_it2 * spring_num_it2) / (element_dimensions(1, plane_direction_1) * element_dimensions(1, plane_direction_2));

				interface_object.spring_num_per_area = max(spring_num_it1_per_area, spring_num_it2_per_area);


				//get number of springs per row where spring_num * spring_num = total number of springs along the whole interface

				interface_object.spring_num = ceil(sqrt(interface_object.spring_num_per_area * (plane_coord1_end2 - plane_coord1_end1) * (plane_coord2_end2 - plane_coord2_end1)));

				//this is simply getting element dimension: a, b, or c
				double bottom_mesh_axis_dimension = (geometry(group_at_bottom, axis_of_connection * 2 + 1) - geometry(group_at_bottom, axis_of_connection * 2)) / (geometry(group_at_bottom, axis_of_connection + 6));
				double top_mesh_axis_dimension = (geometry(group_at_top, axis_of_connection * 2 + 1) - geometry(group_at_top, axis_of_connection * 2)) / (geometry(group_at_top, axis_of_connection + 6));

				//offsetting the interface selection so that the selection_processor can capture elements correctly from both geometry_groups
				coordinates(0, axis_of_connection) = plane_normal - bottom_mesh_axis_dimension / 2.0;
				coordinates(1, axis_of_connection) = plane_normal + top_mesh_axis_dimension / 2.0;

				interface_object.element_bottom_coord = coordinates(0, axis_of_connection);
				interface_object.element_top_coord = coordinates(1, axis_of_connection);

				Selection input_coords;

				input_coords.x_coord1 = coordinates(0, 0);
				input_coords.x_coord2 = coordinates(1, 0);

				input_coords.y_coord1 = coordinates(0, 1);
				input_coords.y_coord2 = coordinates(1, 1);

				input_coords.z_coord1 = coordinates(0, 2);
				input_coords.z_coord2 = coordinates(1, 2);

				input_coords.dof = 0; //garbage value but initialize it just to be safe

				vector<SelectionPoint> affected_points = selection_processor(input_coords, element, total_ele_num);

				for (int ele = 0; ele < affected_points.size(); ele++) {
					double x_now = affected_points[ele].x_coord;
					double y_now = affected_points[ele].y_coord;
					double z_now = affected_points[ele].z_coord;

					interface_object.elements.push_back(element_finder(element, total_ele_num, x_now, y_now, z_now));
				}

				for (int ele = 0; ele < interface_object.elements.size(); ele++) {
					element(interface_object.elements[ele], 20) = 1;
				}

				interfaces.push_back(interface_object);
			}
		}

		//-----------------------------------------------------------------------------

		std::cout << "Interfaces prepared!\n\n";

		//-----------------------------------------------------------------------------

		std::cout << "Preparing the analysis points...\n";

		//-----------------------------------------------------------------------------

		//FOR SPECIFIC POINTS & DOFs OF INTEREST
		//gets the range and dof of elements to analyze
		//can make the range a point, 1D, 2D, or 3D by equating the bounds (i.e. x1 == x2 will be a yz plane of analysis)

		int analyze_num = 0;

		for (int i = 0; i < state_tracker.size(); i++) {
			if (state_tracker[i] == 6) {
				analyze_num++;
			}
		}
		vector<SelectionPoint> analyze_description;

		for (int i = 0; i < state_tracker.size(); i++) {
			if (state_tracker[i] == 6) { //ANALYZE - 6
				vector<double> analyze_now = input_data[i];

				bool foolproof_check = false;

				if (analyze_now.size() < 8) {
					foolproof_check = true; //in case the user forgot to add a DOF entry
				}

				if (analyze_now.size() != 8) {
					analyze_now.resize(8);
				}

				/*

				user input vector legend:
				0 - current state (ANALYZE = 6)
				1 - xcoord1
				2 - xcoord2
				3 - ycoord1
				4 - ycoord2
				5 - zcoord1
				6 - zcoord2
				7 - DOF (-1 if all)

				*/

				double x1 = min(analyze_now[1], analyze_now[2]);
				double x2 = max(analyze_now[1], analyze_now[2]);

				double y1 = min(analyze_now[3], analyze_now[4]);
				double y2 = max(analyze_now[3], analyze_now[4]);

				double z1 = min(analyze_now[5], analyze_now[6]);
				double z2 = max(analyze_now[5], analyze_now[6]);

				if (foolproof_check) {
					analyze_now[7] = -1;
				}

				for (int dof_iterant = 0; dof_iterant < 6; dof_iterant++) { //run through all 6 dofs
					if ((dof_iterant == analyze_now[7]) || (analyze_now[7] == -1)) { //get matching to input, or -1 is true for all 6
						Selection analyze_temp;

						analyze_temp.x_coord1 = x1;
						analyze_temp.x_coord2 = x2;
						analyze_temp.y_coord1 = y1;
						analyze_temp.y_coord2 = y2;
						analyze_temp.z_coord1 = z1;
						analyze_temp.z_coord2 = z2;

						analyze_temp.dof = dof_iterant;

						vector<SelectionPoint> equiv_analyze = selection_processor(analyze_temp, element, total_ele_num);
						analyze_description.insert(analyze_description.end(), equiv_analyze.begin(), equiv_analyze.end());
					}
				}
			}
		}

		int analyze_count = analyze_description.size();
		VectorXi dofs_to_analyze = dof_finder(analyze_description, analyze_count, element, total_ele_num);
		int dofs_to_analyze_count = analyze_count;

		//-----------------------------------------------------------------------------

		std::cout << "Analysis points prepared!\n\n";

		//-----------------------------------------------------------------------------

		std::cout << "Preparing the supports...\n";

		//-----------------------------------------------------------------------------

		vector<SelectionPoint> support_description;

		//-----------------------------------------------------------------------------

		int support_num = 0;

		for (int i = 0; i < state_tracker.size(); i++) {
			if (state_tracker[i] == 7) {
				support_num++;
			}
		}

		for (int i = 0; i < state_tracker.size(); i++) {
			if (state_tracker[i] == 7) { //SUPPORT - 7
				vector<double> support_now = input_data[i];

				if (support_now.size() != 13) {
					support_now.resize(13);
				}

				/*

				user input vector legend:
				0 - current state (SUPPORT - 7)
				1 - xcoord1
				2 - xcoord2
				3 - ycoord1
				4 - ycoord2
				5 - zcoord1
				6 - zcoord2
				7 - restrain ux?
				8 - restrain uy?
				9 - restrain uz?
				10 - restrain thetax?
				11 - restrain thetay?
				12 - restrain thetaz?

				*/

				double x1 = min(support_now[1], support_now[2]);
				double x2 = max(support_now[1], support_now[2]);

				double y1 = min(support_now[3], support_now[4]);
				double y2 = max(support_now[3], support_now[4]);

				double z1 = min(support_now[5], support_now[6]);
				double z2 = max(support_now[5], support_now[6]);

				for (int dof_iterant = 0; dof_iterant < 6; dof_iterant++) {
					//check if the indices 7-12 are == 1 then restrict

					int restrain_check = support_now[(size_t)(dof_iterant + 7)];

					if (restrain_check == 1) {

						Selection support_temp;

						support_temp.x_coord1 = x1;
						support_temp.x_coord2 = x2;
						support_temp.y_coord1 = y1;
						support_temp.y_coord2 = y2;
						support_temp.z_coord1 = z1;
						support_temp.z_coord2 = z2;;
						support_temp.dof = dof_iterant;

						vector<SelectionPoint> equiv_support = selection_processor(support_temp, element, total_ele_num);
						support_description.insert(support_description.end(), equiv_support.begin(), equiv_support.end());
					}
				}
			}
		}

		int support_count = support_description.size();
		VectorXi dofs_to_restrain = dof_finder(support_description, support_count, element, total_ele_num);
		support_restrainer(element, total_ele_num, support_description, support_count);
		int dofs_to_restrain_count = support_description.size();

		//-----------------------------------------------------------------------------

		//VECTORS CONTAINING THE LIST OF DOFs

		//dofs

		VectorXi dof = VectorXi::Zero(total_dof_num);

		int pdof_count = 0;

		for (int i = 0; i < dofs_to_restrain_count; i++) {

			int dof_now = dofs_to_restrain(i);
			int checker = dof(dof_now);

			if (checker != 1) { //this already handles the duplicates

				dof(dof_now) = 1; //pdof!
				pdof_count++;
			}
		}

		int fdof_count = total_dof_num - pdof_count;

		VectorXi pdof = VectorXi::Zero(pdof_count);
		VectorXi fdof = VectorXi::Zero(fdof_count);

		//-----------------------------------------------------------------------------

		//pdofs and fdofs

		int dof_counter = 0;
		int fdof_counter = 0;
		int pdof_counter = 0;

		for (int i = 0; i < total_dof_num; i++) {
			if (dof(i) == 1) {
				pdof(pdof_counter) = i;
				pdof_counter++;
			}
			else {
				fdof(fdof_counter) = i;
				fdof_counter++;
			}
		}

		//-----------------------------------------------------------------------------

		std::cout << "Supports prepared!\n\n";

		//-----------------------------------------------------------------------------

		std::cout << "Preparing the forces...\n";

		//FORCE VECTORS

		VectorXd F = VectorXd::Zero(total_dof_num);
		VectorXd U = VectorXd::Zero(total_dof_num);

		//-----------------------------------------------------------------------------

		vector<Force> force_description;
		vector<Force> constant_force_description;

		//-----------------------------------------------------------------------------

		//point_load - 8

		int point_load_num = 0;

		for (int i = 0; i < state_tracker.size(); i++) {
			if (state_tracker[i] == 8) {
				point_load_num++;
			}
		}

		for (int i = 0; i < state_tracker.size(); i++) {
			if (state_tracker[i] == 8) {
				vector<double> point_load_now = input_data[i];

				if (point_load_now.size() != 6) {
					point_load_now.resize(6);
				}

				/*
				user input vector legend:
				0 - current state (point_load - 8)
				1 - xcoord
				2 - ycoord
				3 - zcoord
				4 - force magnitude
				5 - force_dof

				*/

				Force point_load_temp;

				point_load_temp.x_coord = point_load_now[1];
				point_load_temp.y_coord = point_load_now[2];
				point_load_temp.z_coord = point_load_now[3];
				point_load_temp.force_magnitude = point_load_now[4];

				int force_dof_input = 0;
				if (((int)point_load_now[5] >= 0) && ((int)point_load_now[5] <= 5)) {
					force_dof_input = (int)point_load_now[5];
				}
				point_load_temp.force_dof = force_dof_input;

				force_description.push_back(point_load_temp);
			}
		}

		//-----------------------------------------------------------------------------

		//INPUT USER DISTRIBUTED LOADS

		//dist_load - 9

		int dist_load_num = 0;

		for (int i = 0; i < state_tracker.size(); i++) {
			if (state_tracker[i] == 9) {
				dist_load_num++;
			}
		}

		for (int i = 0; i < state_tracker.size(); i++) {
			if (state_tracker[i] == 9) {
				vector<double> dist_load_now = input_data[i];

				if (dist_load_now.size() != 10) {
					dist_load_now.resize(10);
				}

				/*
				user input vector legend:
				0 - current state (dist_load - 9)
				1 - xcoord1
				2 - xcoord2
				3 - ycoord1
				4 - ycoord2
				5 - zcoord1
				6 - zcoord2
				7 - force magnitude
				8 - force_dof
				9 - triangular load?
				*/

				double x1 = min(dist_load_now[1], dist_load_now[2]);
				double x2 = max(dist_load_now[1], dist_load_now[2]);

				double y1 = min(dist_load_now[3], dist_load_now[4]);
				double y2 = max(dist_load_now[3], dist_load_now[4]);

				double z1 = min(dist_load_now[5], dist_load_now[6]);
				double z2 = max(dist_load_now[5], dist_load_now[6]);

				DistributedLoad dist_load_temp;

				dist_load_temp.x_coord1 = x1;
				dist_load_temp.x_coord2 = x2;
				dist_load_temp.y_coord1 = y1;
				dist_load_temp.y_coord2 = y2;
				dist_load_temp.z_coord1 = z1;
				dist_load_temp.z_coord2 = z2;
				dist_load_temp.force_magnitude = dist_load_now[7]; //note: N/mm == kN/m

				int force_dof_input = 0;
				if (((int)dist_load_now[8] >= 0) && ((int)dist_load_now[8] <= 5)) {
					force_dof_input = (int)dist_load_now[8];
				}
				dist_load_temp.force_dof = force_dof_input;

				int triangle_input = 0;
				if (((int)dist_load_now[9] >= 0) && ((int)dist_load_now[9] <= 2)) {
					triangle_input = (int)dist_load_now[9];
				}
				dist_load_temp.triangle = triangle_input;

				vector<Force> equiv_load = distributed_load_processor(dist_load_temp, geometry, element, total_ele_num);
				force_description.insert(force_description.end(), equiv_load.begin(), equiv_load.end());

			}
		}

		//-----------------------------------------------------------------------------

		//self_weight - 10

		//SPECIFIY THAT UNITS ARE IN kN/m^3

		int self_weight_num = 0;

		for (int i = 0; i < state_tracker.size(); i++) {
			if (state_tracker[i] == 10) {
				self_weight_num++;
			}
		}

		for (int i = 0; i < state_tracker.size(); i++) {
			if (state_tracker[i] == 10) {
				vector<double> self_weight_now = input_data[i];

				if (self_weight_now.size() != 10) {
					self_weight_now.resize(10);
				}

				/*
				user input vector legend:
				0 - current state (self-weight - 10)
				1 - xcoord1
				2 - xcoord2
				3 - ycoord1
				4 - ycoord2
				5 - zcoord1
				6 - zcoord2
				7 - force magnitude
				8 - force_dof
				9 - triangle (garbage value)
				*/

				double x1 = min(self_weight_now[1], self_weight_now[2]);
				double x2 = max(self_weight_now[1], self_weight_now[2]);

				double y1 = min(self_weight_now[3], self_weight_now[4]);
				double y2 = max(self_weight_now[3], self_weight_now[4]);

				double z1 = min(self_weight_now[5], self_weight_now[6]);
				double z2 = max(self_weight_now[5], self_weight_now[6]);

				DistributedLoad self_weight_temp;

				self_weight_temp.x_coord1 = x1;
				self_weight_temp.x_coord2 = x2;
				self_weight_temp.y_coord1 = y1;
				self_weight_temp.y_coord2 = y2;
				self_weight_temp.z_coord1 = z1;
				self_weight_temp.z_coord2 = z2;
				self_weight_temp.force_magnitude = self_weight_now[7]; //in units N/mm^3

				int force_dof_input = 0;
				if (((int)self_weight_now[8] >= 0) && ((int)self_weight_now[8] <= 5)) {
					force_dof_input = (int)self_weight_now[8];
				}
				self_weight_temp.force_dof = force_dof_input;

				self_weight_temp.triangle = 0;

				if (self_weight_temp.force_magnitude != 0) {
					vector<Force> equiv_load = distributed_load_processor(self_weight_temp, geometry, element, total_ele_num);
					constant_force_description.insert(constant_force_description.end(), equiv_load.begin(), equiv_load.end());
				}
			}
		}

		//-----------------------------------------------------------------------------

		int load_count = force_description.size();
		int dofs_to_apply_count = load_count;

		int constant_load_count = constant_force_description.size();
		int constant_dofs_to_apply_count = constant_load_count;

		MatrixXd dofs_to_apply = force_dof_finder(force_description, load_count, element, total_ele_num);
		MatrixXd constant_dofs_to_apply = force_dof_finder(constant_force_description, constant_load_count, element, total_ele_num);

		std::cout << "Forces prepared!\n\n";

		//-----------------------------------------------------------------------------

		//SPRING MATRIX

		std::cout << "Preparing the springs...\n";

		//-----------------------------------------------------------------------------

		int total_spring_num = 0;

		//loop through geometry
		for (int i = 0; i < geometry_group_num; i++) {
			int elementsx = geometry(i, 6);
			int elementsy = geometry(i, 7);
			int elementsz = geometry(i, 8);

			int springsx = geometry(i, 9);
			int springsy = geometry(i, 10);
			int springsz = geometry(i, 11);

			int springs_along_x = springsx * springsx * (elementsx - 1) * elementsy * elementsz;
			int springs_along_y = springsy * springsy * elementsx * (elementsy - 1) * elementsz;
			int springs_along_z = springsz * springsz * elementsx * elementsy * (elementsz - 1);

			total_spring_num += springs_along_x + springs_along_y + springs_along_z;
		}

		//loop through interfaces
		for (int i = 0; i < interfaces.size(); i++) {
			total_spring_num += interfaces[i].spring_num * interfaces[i].spring_num;
		}

		//-----------------------------------------------------------------------------

		MatrixXd spring = MatrixXd::Zero(total_spring_num, 15);

		//-----------------------------------------------------------------------------

		/*

		LEGEND
		0 - spring number
		1 - spring direction
		2 - element number 1
		3 - element number 2
		4 - x coordinate end 1
		5 - y coordinate end 1
		6 - z coordinate end 1
		7 - x end 2
		8 - y end 2
		9 - z end 2
		10 - spring condition: 0 - elastic, 1 - ultimate failure, 2 - yield, 3 - plastic
		11 - spring material
		12 - interface index (-1 = not an interface)
		13 - strain of spring
		14 - stress of spring

		*/

		counter = 0;

		for (int i = 0; i < geometry_group_num; i++) {

			int elements_x = geometry(i, 6);
			int elements_y = geometry(i, 7);
			int elements_z = geometry(i, 8);

			int spring_num_x = (int)geometry(i, 9);
			int spring_num_y = (int)geometry(i, 10);
			int spring_num_z = (int)geometry(i, 11);

			//get index of first element in geometry_group

			double first_element = 0;

			for (int j = 0; j < i; j++) {
				first_element += geometry(j, 6) * geometry(j, 7) * geometry(j, 8);
			}

			for (int direction = 0; direction < 3; direction++) {

				int iterant_spring[3] = { elements_x, elements_y, elements_z };
				iterant_spring[direction] -= 1; //e.g. 20 elements along x, there will only be 19 faces to populate with springs

				vector<int> directions{ 0, 1, 2 };
				directions.erase(remove(directions.begin(), directions.end(), direction), directions.end()); //erase-remove idiom

				int spring_direction = direction;
				int other_direction_1 = directions[0];
				int other_direction_2 = directions[1];

				int spring_num[3] = { spring_num_x, spring_num_y, spring_num_z };

				int face1_iterant = spring_num[spring_direction];
				int face2_iterant = spring_num[spring_direction];

				int element2_add[3] = { 1, elements_x, elements_x * elements_y }; // to get the next element index

				//element column index for coordinates = 1, 2, 3
				int spring_index_run = spring_direction + 1;
				int spring_index_face1 = other_direction_1 + 1;
				int spring_index_face2 = other_direction_2 + 1;

				//spring column index for coordinates = 4, 5, 6
				int spring_coordinate_index_run = spring_direction + 4;
				int spring_coordinate_index_face1 = other_direction_1 + 4;
				int spring_coordinate_index_face2 = other_direction_2 + 4;

				for (int z = 0; z < iterant_spring[2]; z++) {
					for (int y = 0; y < iterant_spring[1]; y++) {
						for (int x = 0; x < iterant_spring[0]; x++) {
							for (int face1 = 0; face1 < face1_iterant; face1++) {
								for (int face2 = 0; face2 < face2_iterant; face2++) {
									spring(counter, 0) = counter;
									spring(counter, 1) = direction;

									//element1
									spring(counter, 2) = (z * elements_x * elements_y) + (y * elements_x) + x + first_element;
									int element1 = spring(counter, 2);

									//element2
									spring(counter, 3) = spring(counter, 2) + element2_add[(size_t)direction];
									int element2 = spring(counter, 3);

									//a, b, c
									double element1_dimensions[3] = { element(element1, 7) , element(element1, 8), element(element1, 9) };
									double element2_dimensions[3] = { element(element2, 7) , element(element2, 8), element(element2, 9) };

									//get tributary length on face of each spring (i.e. tributary area of spring = face1_interval * face2_interval)
									double face1_element1_interval = element1_dimensions[other_direction_1] / face1_iterant;
									double face2_element1_interval = element1_dimensions[other_direction_2] / face2_iterant;

									double face1_element2_interval = element2_dimensions[other_direction_1] / face1_iterant;
									double face2_element2_interval = element2_dimensions[other_direction_2] / face2_iterant;

									//start at the lower left corner of the face
									double edge_face1_element1 = element(element1, spring_index_face1) - element1_dimensions[other_direction_1] / 2.0; // get centroid - b /2
									double edge_face2_element1 = element(element1, spring_index_face2) - element1_dimensions[other_direction_2] / 2.0; // get centroid - c /2

									double edge_face1_element2 = element(element2, spring_index_face1) - element2_dimensions[other_direction_1] / 2.0; // get centroid - b /2
									double edge_face2_element2 = element(element2, spring_index_face2) - element2_dimensions[other_direction_2] / 2.0; // get centroid - c /2

									//end 1 coordinates
									spring(counter, spring_coordinate_index_run) = element(element1, spring_index_run) + element1_dimensions[direction] / 2.0; //from centroid to face
									spring(counter, spring_coordinate_index_face1) = edge_face1_element1 + (face1_element1_interval * (face1 + 0.5)); //springs sit at 0.5, 1.5, 2.5, 3.5, ... (n-0.5)
									spring(counter, spring_coordinate_index_face2) = edge_face2_element1 + (face2_element1_interval * (face2 + 0.5)); //face1 is face direction 1, face2 is face direction 2 (e.g., x-run, y-face1, z-face2)

									//end 2 coordinates
									spring(counter, spring_coordinate_index_run + 3) = element(element2, spring_index_run) - element2_dimensions[direction] / 2.0;
									spring(counter, spring_coordinate_index_face1 + 3) = edge_face1_element2 + (face1_element2_interval * (face1 + 0.5));
									spring(counter, spring_coordinate_index_face2 + 3) = edge_face2_element2 + (face2_element2_interval * (face2 + 0.5));

									//condition of spring: 0 - elastic, 1 - failed, 2 - yield, 3 - plastic
									//spring(counter, 10) = 0; // all start out at elastic state

									//material_group
									spring(counter, 11) = geometry(i, 12);

									//interface?
									spring(counter, 12) = -1;

									counter++;
								}
							}
						}
					}
				}
			}
		}

		for (int i = 0; i < interfaces.size(); i++) {

			int direction = interfaces[i].axis_of_connection;

			vector<int> directions{ 0, 1, 2 };
			directions.erase(remove(directions.begin(), directions.end(), direction), directions.end()); //erase-remove idiom

			int spring_direction = direction;
			int other_direction_1 = directions[0];
			int other_direction_2 = directions[1];

			//coords for iterating
			vector<double> coords1 = { interfaces[i].x_coord1,  interfaces[i].y_coord1,  interfaces[i].z_coord1 };
			vector<double> coords2 = { interfaces[i].x_coord2,  interfaces[i].y_coord2,  interfaces[i].z_coord2 };

			double origin_run = coords1[direction];
			double origin_face1 = coords1[other_direction_1];
			double origin_face2 = coords1[other_direction_2];

			int face1_iterant = interfaces[i].spring_num;
			int face2_iterant = interfaces[i].spring_num;

			//element column index for coordinates = 1, 2, 3
			int spring_index_run = spring_direction + 1;
			int spring_index_face1 = other_direction_1 + 1;
			int spring_index_face2 = other_direction_2 + 1;

			//spring column index for coordinates = 4, 5, 6
			int spring_coordinate_index_run = spring_direction + 4;
			int spring_coordinate_index_face1 = other_direction_1 + 4;
			int spring_coordinate_index_face2 = other_direction_2 + 4;

			double face1_interval = (coords2[other_direction_1] - coords1[other_direction_1]) / face1_iterant;
			double face2_interval = (coords2[other_direction_2] - coords1[other_direction_2]) / face2_iterant;

			for (int face1 = 0; face1 < face1_iterant; face1++) {
				for (int face2 = 0; face2 < face2_iterant; face2++) {

					spring(counter, 0) = counter;
					spring(counter, 1) = direction;

					//end 1 coordinates
					spring(counter, spring_coordinate_index_run) = origin_run;
					spring(counter, spring_coordinate_index_face1) = origin_face1 + (face1_interval * (face1 + 0.5));
					spring(counter, spring_coordinate_index_face2) = origin_face2 + (face2_interval * (face2 + 0.5));

					//end 2 coordinates
					spring(counter, spring_coordinate_index_run + 3) = origin_run;
					spring(counter, spring_coordinate_index_face1 + 3) = origin_face1 + (face1_interval * (face1 + 0.5));
					spring(counter, spring_coordinate_index_face2 + 3) = origin_face2 + (face2_interval * (face2 + 0.5));


					//offset coords for element finder
					vector<double> coords1_select{ 0,0,0 };
					vector<double> coords2_select{ 0,0,0 };

					coords1_select[direction] = interfaces[i].element_bottom_coord;
					coords2_select[direction] = interfaces[i].element_top_coord;

					coords1_select[other_direction_1] = spring(counter, spring_coordinate_index_face1);
					coords2_select[other_direction_1] = spring(counter, spring_coordinate_index_face1 + 3);

					coords1_select[other_direction_2] = spring(counter, spring_coordinate_index_face2);
					coords2_select[other_direction_2] = spring(counter, spring_coordinate_index_face2 + 3);

					//elements
					spring(counter, 2) = element_finder(element, total_ele_num, coords1_select[0], coords1_select[1], coords1_select[2]);
					spring(counter, 3) = element_finder(element, total_ele_num, coords2_select[0], coords2_select[1], coords2_select[2]);

					//condition of spring: 0 - elastic, 1 - failed, 2 - yield, 3 - plastic
					//spring(counter, 10) = 0; // all start out at elastic state

					//material_group
					spring(counter, 11) = interfaces[i].spring_material_group;

					//interface group number?
					spring(counter, 12) = i;

					counter++;
				}
			}
		}

		std::cout << "Springs prepared!\n\n";

		//-----------------------------------------------------------------------------

		//Secondary Geometry
		std::cout << "Preparing the secondary geometry...\n";

		//-----------------------------------------------------------------------------

		//secondary_geometry - 2

		int secondary_geometry_num = 0;

		for (int i = 0; i < state_tracker.size(); i++) {
			if (state_tracker[i] == 2) { //check if the data input is from command SECONDARY_GEOMETRY
				secondary_geometry_num++;
			}
		}

		for (int i = 0; i < state_tracker.size(); i++) {
			if (state_tracker[i] == 2) {

				vector<double> secondary_geometry_now = input_data[i];

				if (secondary_geometry_now.size() != 8) {
					secondary_geometry_now.resize(8);
				}

				/*
				user input vector legend:
				0 - current state (secondary_geometry - 2)
				1 - xcoord1
				2 - xcoord2
				3 - ycoord1
				4 - ycoord2
				5 - zcoord1
				6 - zcoord2
				7 - material group
				*/

				double x1 = min(secondary_geometry_now[1], secondary_geometry_now[2]);
				double x2 = max(secondary_geometry_now[1], secondary_geometry_now[2]);

				double y1 = min(secondary_geometry_now[3], secondary_geometry_now[4]);
				double y2 = max(secondary_geometry_now[3], secondary_geometry_now[4]);

				double z1 = min(secondary_geometry_now[5], secondary_geometry_now[6]);
				double z2 = max(secondary_geometry_now[5], secondary_geometry_now[6]);

				Selection bounds_now;

				bounds_now.x_coord1 = x1;
				bounds_now.x_coord2 = x2;
				bounds_now.y_coord1 = y1;
				bounds_now.y_coord2 = y2;
				bounds_now.z_coord1 = z1;
				bounds_now.z_coord2 = z2;

				int spring_material_group = 0; //default
				for (int spring_check = 0; spring_check < material_group_num; spring_check++) {
					if ((int)secondary_geometry_now[7] == spring_check) {
						spring_material_group = (int)secondary_geometry_now[7];
					}
				}

				vector<int> affected_springs = material_tagger(bounds_now, element, total_ele_num, spring, total_spring_num, geometry);

				if (affected_springs.size() > 0) {
					for (int j = 0; j < affected_springs.size(); j++) {
						spring(affected_springs[j], 11) = spring_material_group;
					}
				}
			}
		}

		std::cout << "Springs prepared!\n\n";

		//-----------------------------------------------------------------------------

		//Reinforcement Bars
		std::cout << "Preparing the reinforcements...\n";

		//-----------------------------------------------------------------------------

		//reinforcement_bars - 3

		int reinforcement_bars_num = 0;

		for (int i = 0; i < state_tracker.size(); i++) {
			if (state_tracker[i] == 3) { //check if the data input is from command REINFORCEMENT_BARS
				reinforcement_bars_num++;
			}
		}

		for (int i = 0; i < state_tracker.size(); i++) {
			if (state_tracker[i] == 3) {

				vector<double> reinforcement_bars_now = input_data[i];

				if (reinforcement_bars_now.size() != 12) {
					reinforcement_bars_now.resize(12);
				}

				/*
				user input vector legend:
				0 - current state (secondary_geometry - 2)
				1 - xcoord1
				2 - xcoord2
				3 - ycoord1
				4 - ycoord2
				5 - zcoord1
				6 - zcoord2
				7 - bar size
				8 - run direction
				9 - spacing direction
				10 - number of bars
				11 - material group
				*/

				double x1 = min(reinforcement_bars_now[1], reinforcement_bars_now[2]);
				double x2 = max(reinforcement_bars_now[1], reinforcement_bars_now[2]);

				double y1 = min(reinforcement_bars_now[3], reinforcement_bars_now[4]);
				double y2 = max(reinforcement_bars_now[3], reinforcement_bars_now[4]);

				double z1 = min(reinforcement_bars_now[5], reinforcement_bars_now[6]);
				double z2 = max(reinforcement_bars_now[5], reinforcement_bars_now[6]);

				Reinforcement rebar_now;

				rebar_now.x_coord1 = x1;
				rebar_now.x_coord2 = x2;
				rebar_now.y_coord1 = y1;
				rebar_now.y_coord2 = y2;
				rebar_now.z_coord1 = z1;
				rebar_now.z_coord2 = z2;

				rebar_now.bar_size = fabs(reinforcement_bars_now[7]);

				int run_direction_input = 0; //default
				if ((reinforcement_bars_now[8] >= 0) && (reinforcement_bars_now[8] <= 2)) {
					run_direction_input = reinforcement_bars_now[8];
				}

				rebar_now.run_direction = run_direction_input;

				int spacing_direction_input = 1; //default
				if ((reinforcement_bars_now[9] >= 0) && (reinforcement_bars_now[9] <= 2)) {
					spacing_direction_input = reinforcement_bars_now[9];
				}

				rebar_now.spacing_direction = spacing_direction_input;

				rebar_now.number_of_bars = (int)fabs(reinforcement_bars_now[10]);

				double spacing_coord1, spacing_coord2;

				if (rebar_now.spacing_direction == 0) {
					spacing_coord1 = rebar_now.x_coord1;
					spacing_coord2 = rebar_now.x_coord2;
				}
				else if (rebar_now.spacing_direction == 1) {
					spacing_coord1 = rebar_now.y_coord1;
					spacing_coord2 = rebar_now.y_coord2;
				}
				else if (rebar_now.spacing_direction == 2) {
					spacing_coord1 = rebar_now.z_coord1;
					spacing_coord2 = rebar_now.z_coord2;
				}

				if (rebar_now.number_of_bars > 1) {
					rebar_now.spacing = ((spacing_coord2 - spacing_coord1) - (rebar_now.number_of_bars * rebar_now.bar_size)) / (rebar_now.number_of_bars - 1.0);
				}
				else {
					rebar_now.spacing = 0;
				}

				if (rebar_now.number_of_bars > 0) {

					vector<int> affected_springs = reinforcement_tagger(rebar_now, element, total_ele_num, spring, total_spring_num, geometry, 0);

					if (affected_springs.size() > 0) {

						int spring_material_group = 0; //default
						for (int spring_check = 0; spring_check < material_group_num; spring_check++) {
							if ((int)reinforcement_bars_now[11] == spring_check) {
								spring_material_group = (int)reinforcement_bars_now[11];
							}
						}

						for (int j = 0; j < affected_springs.size(); j++) {
							spring(affected_springs[j], 11) = spring_material_group;
						}
					}
				}
			}
		}

		//-----------------------------------------------------------------------------

		//stirrups - 4

		int stirrups_num = 0;

		for (int i = 0; i < state_tracker.size(); i++) {
			if (state_tracker[i] == 4) { //check if the data input is from command STIRRUPS
				stirrups_num++;
			}
		}

		for (int i = 0; i < state_tracker.size(); i++) {
			if (state_tracker[i] == 4) {

				vector<double> stirrups_now = input_data[i];

				if (stirrups_now.size() != 11) {
					stirrups_now.resize(11);
				}

				/*
				user input vector legend:
				0 - current state (secondary_geometry - 2)
				1 - xcoord1
				2 - xcoord2
				3 - ycoord1
				4 - ycoord2
				5 - zcoord1
				6 - zcoord2
				7 - bar size
				8 - run direction
				9 - number of bars
				10 - material group
				*/

				double x1 = min(stirrups_now[1], stirrups_now[2]);
				double x2 = max(stirrups_now[1], stirrups_now[2]);

				double y1 = min(stirrups_now[3], stirrups_now[4]);
				double y2 = max(stirrups_now[3], stirrups_now[4]);

				double z1 = min(stirrups_now[5], stirrups_now[6]);
				double z2 = max(stirrups_now[5], stirrups_now[6]);

				Reinforcement stirrups_object;

				stirrups_object.x_coord1 = x1;
				stirrups_object.x_coord2 = x2;
				stirrups_object.y_coord1 = y1;
				stirrups_object.y_coord2 = y2;
				stirrups_object.z_coord1 = z1;
				stirrups_object.z_coord2 = z2;

				int spacing_direction_input = 0; //default
				if ((stirrups_now[8] >= 0) && (stirrups_now[8] <= 2)) {
					spacing_direction_input = stirrups_now[8];
				}

				stirrups_object.bar_size = fabs(stirrups_now[7]);
				stirrups_object.spacing_direction = spacing_direction_input; //not intuitive but that's how reinforcement_tagger is set-up
				stirrups_object.number_of_bars = (int)fabs(stirrups_now[9]);

				//get run direction, any of the two remaining

				vector<int> directions{ 0, 1, 2 };

				directions.erase(remove(directions.begin(), directions.end(), stirrups_now[8]), directions.end()); //erase-remove idiom

				stirrups_object.run_direction = directions[0];

				double spacing_coord1, spacing_coord2;

				if (stirrups_object.spacing_direction == 0) {
					spacing_coord1 = stirrups_object.x_coord1;
					spacing_coord2 = stirrups_object.x_coord2;
				}
				else if (stirrups_object.spacing_direction == 1) {
					spacing_coord1 = stirrups_object.y_coord1;
					spacing_coord2 = stirrups_object.y_coord2;
				}
				else if (stirrups_object.spacing_direction == 2) {
					spacing_coord1 = stirrups_object.z_coord1;
					spacing_coord2 = stirrups_object.z_coord2;
				}

				if (stirrups_object.number_of_bars > 1) {
					stirrups_object.spacing = ((spacing_coord2 - spacing_coord1) - (stirrups_object.number_of_bars * stirrups_object.bar_size)) / (stirrups_object.number_of_bars - 1.0);
				}
				else {
					stirrups_object.spacing = 0;
				}

				if (stirrups_object.number_of_bars > 0) {

					vector<int> affected_springs = reinforcement_tagger(stirrups_object, element, total_ele_num, spring, total_spring_num, geometry, 1);

					if (affected_springs.size() > 0) {

						int spring_material_group = 0; //default
						for (int spring_check = 0; spring_check < material_group_num; spring_check++) {
							if ((int)stirrups_now[10] == spring_check) {
								spring_material_group = (int)stirrups_now[10];
							}
						}

						for (int j = 0; j < affected_springs.size(); j++) {
							spring(affected_springs[j], 11) = spring_material_group;
						}
					}

				}
			}
		}

		std::cout << "\nReinforcements prepared!\n\n-------------------------------------------------------------\n\n";
		std::cout << "Writing the initial files...\n";

		MatrixXd element_copy;
		MatrixXd spring_copy;

		if (crack_reset_check) {
			//duplicate for elastic unloading-reloading
			element_copy = element;
			spring_copy = spring;
		}

		//-----------------------------------------------------------------------------

		if (print_list[2] == 1) {
			string pdof_filename = default_name + "_INITIAL_" + "pdof" + ".csv";
			ofstream pdof_file;
			pdof_file.open(pdof_filename, ofstream::app);
			if (pdof_file.is_open()) {
				pdof_file << pdof.format(CSVFormat);
				pdof_file.close();
			}
			else {
				std::cout << "\a\n\nERROR: Failed to write to file!\n";
			}
		}

		//-----------------------------------------------------------------------------

		if (print_list[3] == 1) {
			string fdof_filename = default_name + "_INITIAL_" + "fdof" + ".csv";
			ofstream fdof_file;
			fdof_file.open(fdof_filename, ofstream::app);
			if (fdof_file.is_open()) {
				fdof_file << fdof.format(CSVFormat);
				fdof_file.close();
			}
			else {
				std::cout << "\a\n\nERROR: Failed to write to file!\n";
			}
		}

		//-----------------------------------------------------------------------------

		if (print_list[4] == 1) {
			string support_filename = default_name + "_INITIAL_" + "support" + ".csv";
			ofstream support_file;
			support_file.open(support_filename, ofstream::app);
			if (support_file.is_open()) {
				support_file << "x,y,z,dof\n";
				for (int i = 0; i < support_description.size(); i++) {
					SelectionPoint support_write = support_description[i];
					support_file << support_write.x_coord << "," << support_write.y_coord << "," << support_write.z_coord << "," << support_write.dof << "\n";
				}
				support_file.close();
			}
			else {
				std::cout << "\a\n\nERROR: Failed to write to file!\n";
			}
		}

		//-----------------------------------------------------------------------------

		if (print_list[5] == 1) {
			string force_filename = default_name + "_INITIAL_" + "force" + ".csv";
			ofstream force_file;
			force_file.open(force_filename, ofstream::app);
			if (force_file.is_open()) {
				force_file << "x,y,z,force (N or Nmm),dof\n";
				for (int i = 0; i < force_description.size(); i++) {
					Force force_write = force_description[i];
					force_file << force_write.x_coord << "," << force_write.y_coord << "," << force_write.z_coord << "," << force_write.force_magnitude << "," << force_write.force_dof << "\n";
				}
				force_file.close();
			}
			else {
				std::cout << "\a\n\nERROR: Failed to write to file!\n";
			}
		}

		//-----------------------------------------------------------------------------

		if (print_list[6] == 1) {
			string force2_filename = default_name + "_INITIAL_" + "force2" + ".csv";
			ofstream force2_file;
			force2_file.open(force2_filename, ofstream::app);
			if (force2_file.is_open()) {
				force2_file << "dof,force (N or Nmm)\n";
				force2_file << dofs_to_apply.format(CSVFormat);
				force2_file.close();
			}
			else {
				std::cout << "\a\n\nERROR: Failed to write to file!\n";
			}
		}

		//-----------------------------------------------------------------------------

		if (print_list[7] == 1) {
			string filename_element_initial = default_name + "_INITIAL_" + "element_initial.csv";
			ofstream element_initial_file;
			element_initial_file.open(filename_element_initial, ofstream::app);
			if (element_initial_file.is_open()) {
				element_initial_file << "element_number,x,y,z\n";
				for (int i = 0; i < total_ele_num; i++) {
					element_initial_file << element(i, 0) << ",";
					element_initial_file << element(i, 1) << "," << element(i, 2) << "," << element(i, 3) << "\n";
				}
				element_initial_file.close();
			}
			else {
				std::cout << "\a\n\nERROR: Failed to write to file!\n";
			}
		}

		//-----------------------------------------------------------------------------

		//resource-intensive

		if (print_list[8] == 1) {
			std::cout << "Writing spring.csv...\n";
			string filename_spring = default_name + "_INITIAL_" + "spring.csv";
			ofstream spring_file;
			spring_file.open(filename_spring, ofstream::app);
			if (spring_file.is_open()) {
				spring_file << "spring_number,spring_direction,element1,element2,x_end1,y_end1,z_end1,x_end2,y_end2,z_end2,spring_condition,spring_material,interface_index,strain,stress\n";
				spring_file << spring.format(CSVFormat);
				spring_file.close();
			}
			else {
				std::cout << "\a\n\nERROR: Failed to write to file!\n";
			}
			std::cout << "spring.csv successfully written!\n\n";
		}

		//-----------------------------------------------------------------------------

		string default_name_initial = default_name + "_INITIAL_";
		MatrixXd element_principal_stresses = MatrixXd::Zero(total_ele_num, 11);

		/*
		0 - principal stress 1
		1 - principal stress 2
		2 - principal stress 3
		3 - sigma xx
		4 - sigma yy
		5 - sigma zz
		6 - tau xy
		7 - tau xz
		8 - tau yz
		9 - maximum of the three principal stresses (both positive and negative)
		10 - springs contributed
		*/

		//make a copy of F for the element_drawer

		VectorXd F_initial = F;
		force_applier(F_initial, dofs_to_apply, dofs_to_apply_count, 1, 1);
		force_applier(F_initial, constant_dofs_to_apply, constant_dofs_to_apply_count, 1, 1);

		if (print_list[17] == 1) {
			element_drawer(default_name_initial, element, total_ele_num, U, F_initial, element_principal_stresses);
		}

		if (print_list[18] == 1) {
			//too much storage space used!
			spring_drawer(default_name_initial, spring, total_spring_num, 0, -1);
		}

		//-----------------------------------------------------------------------------

		for (int i = 19; i < print_list.size(); i++) {

			int choice_of_material = print_list[i];
			i++;
			int choice_of_condition = print_list[i];

			std::ostringstream choice_of_material_string;
			choice_of_material_string << std::setw(5) << std::setfill('0') << choice_of_material;

			std::ostringstream choice_of_condition_string;
			choice_of_condition_string << std::setw(5) << std::setfill('0') << choice_of_condition;

			string material_string;
			string condition_string;

			if (choice_of_material == -1) {
				material_string = "ALL";
			}
			else {
				material_string = choice_of_material_string.str();
			}

			if (choice_of_condition == -1) {
				condition_string = "ALL";
			}
			else {
				condition_string = choice_of_condition_string.str();
			}

			string default_name_initial_special = default_name_initial + "FILTERED_Material_" + material_string + "_Condition_" + condition_string + "_";
			spring_drawer(default_name_initial_special, spring, total_spring_num, choice_of_material, choice_of_condition);
		}

		std::cout << "Files successfully written!\n\n";

		//-----------------------------------------------------------------------------

		//MAIN FOR-LOOP TO ITERATE OVER EVERY LOAD INCREMENT

		int cracked_spring_count = 0;

		//-----------------------------------------------------------------------------
		int load_increment_for_force_applier = load_increment_start;

		for (int load_increment = load_increment_start; load_increment <= load_increment_count; load_increment++) {

			if (crack_reset_check) {
				if (crack_reset_increment == load_increment) {
					//reset element and spring coordinates
					crack_reset_check = false;
					load_increment = load_increment_start;
					element = element_copy;

					for (int i = 0; i < total_spring_num; i++) {
						spring(i, 4) = spring_copy(i, 4);
						spring(i, 5) = spring_copy(i, 5);
						spring(i, 6) = spring_copy(i, 6);
						spring(i, 7) = spring_copy(i, 7);
						spring(i, 8) = spring_copy(i, 8);
						spring(i, 9) = spring_copy(i, 9);
					}

				}

			}

			std::ostringstream load_increment_string;
			load_increment_string << std::setw(5) << std::setfill('0') << load_increment;

			string default_name_increment = default_name + "_LoadIncrement-" + load_increment_string.str() + "_";
			int increment_time_start = time(NULL);

			std::cout << "-------------------------------------------------------------\n\n";
			std::cout << "Current load increment: " << load_increment << " of " << load_increment_count << endl << endl;
			std::cout << "-------------------------------------------------------------\n\n";

			//-----------------------------------------------------------------------------

			force_applier(F, dofs_to_apply, dofs_to_apply_count, load_increment_count, load_increment_for_force_applier);

			//apply self-weight
			force_applier(F, constant_dofs_to_apply, constant_dofs_to_apply_count, 1, 1);

			//-----------------------------------------------------------------------------

			/*

			--- 2 STIFFNESS MATRICES ---

			*/

			//reserve the SparseMatrix list of entries so that memory is one continuous block
			unsigned long long int reserveCount_ff = (unsigned long long)total_ele_num * 3.0 * 96.0 * (2.5 / 3.0); //six faces per element (six neighbors) divided by 2 (element1 <-> element2 == element2 <-> element1), 96 nonzero entries per element1-element2 local matrix
			// DEVELOPER NOTE: check if the 2.5/3.0 (arbitrary) is too much of an underestimation
			unsigned long long int reserveCount_pf = ((double)pdof_count / 6.0) * 3.0 * 96.0 * (1.0 / 2.0);
			// DEVELOPER NOTE: check if the 1.0/2.0 (arbitrary) is too much of an underestimation

			//-----------------------------------------------------------------------------

			vector<T> tripletList_ff;
			vector<T> tripletList_pf;

			tripletList_ff.reserve(reserveCount_ff);
			tripletList_pf.reserve(reserveCount_pf);

			vector<int> thread_ff_length;
			vector<int> thread_pf_length;

			//-----------------------------------------------------------------------------

			std::cout << "Troubleshooting" << "\ntotal_ele_num = " << total_ele_num;

			for (int i = 0; i < geometry_group_num; i++) {
				int ele_x = geometry(i, 6);
				int ele_y = geometry(i, 7);
				int ele_z = geometry(i, 8);

				int ele_num = ele_x * ele_y * ele_z;

				int spring_num_x = geometry(i, 9);
				int spring_num_y = geometry(i, 10);
				int spring_num_z = geometry(i, 11);

				double xbound1_geometry = geometry(i, 0);
				double xbound2_geometry = geometry(i, 1);
				double ybound1_geometry = geometry(i, 2);
				double ybound2_geometry = geometry(i, 3);
				double zbound1_geometry = geometry(i, 4);
				double zbound2_geometry = geometry(i, 5);

				std::cout << "\n\nFOR GROUP NUMBER " << i << ":";
				std::cout << "\ngrid_sizes: " << ele_x << ", " << ele_y << ", " << ele_z;
				std::cout << "\ngeometry bounds: (" << xbound1_geometry << ", " << ybound1_geometry << ", " << zbound1_geometry << ") to (" << xbound2_geometry << ", " << ybound2_geometry << ", " << zbound2_geometry << ")";
				std::cout << "\nele_num = " << ele_num;
				std::cout << "\nspring_nums: " << spring_num_x * spring_num_x << ", " << spring_num_y * spring_num_y << ", " << spring_num_z * spring_num_z;
			}

			std::cout << "\n\ntotal_dof_num = " << total_dof_num << "\npdof_count = " << pdof_count << "\nfdof_count = " << fdof_count << "\n\ntotal_spring_num = " << total_spring_num;
			std::cout << "\n\nreserveCount_ff = " << reserveCount_ff << "\nreserveCount_pf = " << reserveCount_pf << endl << endl;

			//-----------------------------------------------------------------------------

			std::cout << "-------------------------------------------------------------\n\n";
			std::cout << "Solving the springs... \n";
			int spring_time_start = time(NULL);

			counter = 0;

			//-----------------------------------------------------------------------------

			int num_threads = num_threads_actual;
			//typical num_threads = number of processors divided by 2
			omp_set_num_threads(num_threads);

			//-----------------------------------------------------------------------------

			double progress = 0.0;
			int hundred_percent = total_spring_num / 100;
			if (hundred_percent <= 0) {
				hundred_percent = 1;
			}

#pragma omp parallel
			{
#pragma omp master
				printf_s("\nNumber of threads: %d\n", omp_get_num_threads());

#pragma omp single
				std::cout << "Number of springs: " << total_spring_num << "\n\n";

				//-----------------------------------------------------------------------------

				//DEVELOPER NOTE: Every variable declared in this parallel section is private to each thread. Every initialization of an existing "public" variable is shared by all threads (see "race condition")

				vector<T> tripletList_ff_private;
				vector<T> tripletList_pf_private;

				tripletList_ff_private.reserve(reserveCount_ff / num_threads);
				tripletList_pf_private.reserve(reserveCount_pf);

				//-----------------------------------------------------------------------------

				MatrixXd Kglobalbuffer = MatrixXd::Zero(12, 12);
				int buffer_element1 = 0;
				int buffer_element2 = 0;

				//-----------------------------------------------------------------------------

				int parallel_counter = 0;

#pragma omp for
				for (int i = 0; i < total_spring_num; i++) {

					//loading screen
					if (counter % 1000 == 0) {
						std::cout << counter << " ";
						//this can sometimes duplicate in parallel computing (multiple threads evaluate simultaneously) but it doesn't affect the calculations, just for a "loading screen"
					}


					/*
					//loading screen from SO
					if (counter % hundred_percent == 0) {
#pragma omp critical
						{
							int barWidth = 100;
							std::cout << "[";
							int pos = barWidth * progress;
							for (int i = 0; i < barWidth; ++i) {
								if (i <= pos) std::cout << "=";
								else std::cout << " ";
							}
							std::cout << "] " << int(progress * 100.0) << "%\r";
							std::cout.flush();
							if (progress < 1) {
								progress += 0.01;
							}
						}
					}
					*/

					//-----------------------------------------------------------------------------

					// Local K of springs
					int element1 = spring(i, 2);
					int element2 = spring(i, 3);

					//-----------------------------------------------------------------------------

					if (parallel_counter == 0) { //first case for each thread
						buffer_element1 = element1;
						buffer_element2 = element2;
						parallel_counter++;
					}

					//-----------------------------------------------------------------------------

					int current_dof_1 = 6 * element1;
					int current_dof_2 = 6 * element2;

					//set-up which dofs are involved
					VectorXi current_dofs = VectorXi::Zero(12);
					for (int j = 0; j < 6; j++) {
						current_dofs(j) = current_dof_1 + j;
					}
					for (int j = 0; j < 6; j++) {
						int k = j + 6;
						current_dofs(k) = current_dof_2 + j;
					}

					//-----------------------------------------------------------------------------

					//check if new element-element interface is being analyzed or same face as previous spring iterated
					if (buffer_element1 == element1 && buffer_element2 == element2) {
						//skip
					}
					else {
						for (int j = 0; j < 12; j++) {
							for (int k = 0; k < 12; k++) {
								int Kg_dof_1 = current_dofs(j);
								int Kg_dof_2 = current_dofs(k);
								double Kg_value = Kglobalbuffer(j, k);

								if (Kg_value == 0) {
									//skip
									//save on processing time because there are some zeroes on the K stiffness matrix (12 x 12)
								}
								else {

									//check if pdof or fdof
									bool row_check = false;
									bool col_check = false;

									//DEVELOPER NOTE: We don't need to build the full K matrix, just the Kff and Kpf matrices

									//-----------------------------------------------------------------------------

									if (dof(Kg_dof_1) == 1) {
										//pdof
										row_check = true;
									}

									if (dof(Kg_dof_2) == 1) {
										//pdof
										col_check = true;
									}

									//-----------------------------------------------------------------------------

									int iterant_dof1 = 0;
									int iterant_dof2 = 0;

									//-----------------------------------------------------------------------------

									if (row_check && col_check) { //pdof both
										//skip
									}
									else if (!row_check && col_check) { //fdof - pdof
										//skip
									}
									else if (row_check && !col_check) { //pdof - fdof
										while (pdof(iterant_dof1) < Kg_dof_1) { iterant_dof1++; } //why we do this: the size of the Kpf matrix is [pdof x fdof], so we can't follow the general dof index numbering
										while (fdof(iterant_dof2) < Kg_dof_2) { iterant_dof2++; } //get the index of pdof according to the pdof numbering, then get the index of fdof according to the fdof numbering

										tripletList_pf_private.push_back(T(iterant_dof1, iterant_dof2, Kg_value));
									}
									else { //fdof both
										while (fdof(iterant_dof1) < Kg_dof_1) { iterant_dof1++; }
										while (fdof(iterant_dof2) < Kg_dof_2) { iterant_dof2++; }

										tripletList_ff_private.push_back(T(iterant_dof1, iterant_dof2, Kg_value));
									}
								}
							}
						}

						//-----------------------------------------------------------------------------

						//reset the holder
						Kglobalbuffer = MatrixXd::Zero(12, 12);
						buffer_element1 = element1;
						buffer_element2 = element2;

						//-----------------------------------------------------------------------------
					}

					double As1, As2, l1, l2, Kn1_spring, Ks1_spring, Kn2_spring, Ks2_spring;

					double difference_x = spring(i, 7) - spring(i, 4);
					double difference_y = spring(i, 8) - spring(i, 5);
					double difference_z = spring(i, 9) - spring(i, 6);

					double spring_elongated_length = sqrt(difference_x * difference_x + difference_y * difference_y + difference_z * difference_z);
					double half_sel = spring_elongated_length / 2.0;

					double Espring;
					double nu_spring;
					double Gspring;

					int dimension1, dimension2, dimension3;
					double dummyx, dummyy, dummyz;
					double sx1, sy1, sz1, sx2, sy2, sz2;
					double xp1, yp1, zp1, xp2, yp2, zp2;
					double ex1, ey1, ez1, ex2, ey2, ez2;
					int spring_num1_element1, spring_num2_element1;
					int spring_num1_element2, spring_num2_element2;

					//-----------------------------------------------------------------------------

					//Get spring E and G

					int spring_material = spring(i, 11);
					int spring_condition = spring(i, 10);
					int material_behavior = material(spring_material, 5);
					int material_nonlinearity_choice = material(spring_material, 6);

					Espring = material(spring_material, 0);
					nu_spring = material(spring_material, 1);
					Gspring = material(spring_material, 2);

					//DEVELOPER NOTE: this is one of the key things to study about AEM. How to treat cracked and yielded springs.

					if (allow_failure == 1) {
						if (material_nonlinearity_choice == 0) { // concrete
							if (spring_condition == 1) {
								Espring *= 0.001;
								Gspring *= 0.001;
								//DEVELOPER NOTE: following the treatment used in the first nonlinear AEM paper
							}
						}

						if (material_nonlinearity_choice == 1) { // steel
							if (spring_condition == 2) {
								Espring *= 0.01;
								Gspring *= 0.01;
								//DEVELOPER NOTE: this is hardcoded, a function allowing for different nonlinear material models may be considered for the future
								//Biaxial steel stress-strain with E0/100 after failure
							}

							if (spring_condition == 1) { //rupture
								Espring *= 0.001;
								Gspring *= 0.001;
							}
						}

						if (material_nonlinearity_choice == 2) { // generic
							if (material_behavior == 0) { // brittle
								if (spring_condition == 1) {
									Espring *= 0.001;
									Gspring *= 0.001;
								}
							}
							else if (material_behavior == 1) { //ductile

								if (spring_condition == 2) { //yield first
									Espring *= 0.01;
									Gspring *= 0.01;
								}

								if (spring_condition == 1) { //then rupture
									Espring *= 0.001;
									Gspring *= 0.001;
								}
							}
						}
					}

					//-----------------------------------------------------------------------------

					int geometry_group1 = element(element1, 19);
					int geometry_group2 = element(element2, 19);


					// 13 - a, 14 - b, 15 - c
					if (spring(i, 1) == 0) {
						dimension1 = 8;
						dimension2 = 9;
						dimension3 = 7;

						dummyx = element(element1, 1) + element(element1, 10);
						dummyy = element(element1, 2) + element(element1, 11);
						dummyz = element(element1, 3) + element(element1, 12);

						xp1 = element(element1, 1) + element(element1, 10);
						yp1 = element(element1, 2) + element(element1, 11);
						zp1 = element(element1, 3) + element(element1, 12);

						xp2 = element(element2, 1) - element(element2, 10);
						yp2 = element(element2, 2) - element(element2, 11);
						zp2 = element(element2, 3) - element(element2, 12);

						spring_num1_element1 = geometry(geometry_group1, 10);
						spring_num2_element1 = geometry(geometry_group1, 11);
						spring_num1_element2 = geometry(geometry_group2, 10);
						spring_num2_element2 = geometry(geometry_group2, 11);

					}
					else if (spring(i, 1) == 1) {
						dimension1 = 7;
						dimension2 = 9;
						dimension3 = 8;

						dummyx = element(element1, 1) + element(element1, 13);
						dummyy = element(element1, 2) + element(element1, 14);
						dummyz = element(element1, 3) + element(element1, 15);

						xp1 = element(element1, 1) + element(element1, 13);
						yp1 = element(element1, 2) + element(element1, 14);
						zp1 = element(element1, 3) + element(element1, 15);

						xp2 = element(element2, 1) - element(element2, 13);
						yp2 = element(element2, 2) - element(element2, 14);
						zp2 = element(element2, 3) - element(element2, 15);

						spring_num1_element1 = geometry(geometry_group1, 9);
						spring_num2_element1 = geometry(geometry_group1, 11);
						spring_num1_element2 = geometry(geometry_group2, 9);
						spring_num2_element2 = geometry(geometry_group2, 11);
					}
					else {
						dimension1 = 7;
						dimension2 = 8;
						dimension3 = 9;

						dummyx = element(element1, 1) + element(element1, 16);
						dummyy = element(element1, 2) + element(element1, 17);
						dummyz = element(element1, 3) + element(element1, 18);

						xp1 = element(element1, 1) + element(element1, 16);
						yp1 = element(element1, 2) + element(element1, 17);
						zp1 = element(element1, 3) + element(element1, 18);

						xp2 = element(element2, 1) - element(element2, 16);
						yp2 = element(element2, 2) - element(element2, 17);
						zp2 = element(element2, 3) - element(element2, 18);

						spring_num1_element1 = geometry(geometry_group1, 9);
						spring_num2_element1 = geometry(geometry_group1, 10);
						spring_num1_element2 = geometry(geometry_group2, 9);
						spring_num2_element2 = geometry(geometry_group2, 10);
					}

					//-----------------------------------------------------------------------------

					As1 = (element(element1, dimension1) / spring_num1_element1) * (element(element1, dimension2) / spring_num2_element1);
					As2 = (element(element2, dimension1) / spring_num1_element2) * (element(element2, dimension2) / spring_num2_element2);

					if (spring(i, 12) != -1) {
						//get interface data

						double x_coord1 = interfaces[spring(i, 12)].x_coord1;
						double x_coord2 = interfaces[spring(i, 12)].x_coord2;

						double y_coord1 = interfaces[spring(i, 12)].y_coord1;
						double y_coord2 = interfaces[spring(i, 12)].y_coord2;

						double z_coord1 = interfaces[spring(i, 12)].z_coord1;
						double z_coord2 = interfaces[spring(i, 12)].z_coord2;

						double differencex_interface = x_coord2 - x_coord1;
						double differencey_interface = y_coord2 - y_coord1;
						double differencez_interface = z_coord2 - z_coord1;

						vector<double> difference_interface{ differencex_interface, differencey_interface, differencez_interface };

						double spring_num_interface = interfaces[spring(i, 12)].spring_num;

						vector<int> directions{ 0, 1, 2 };
						directions.erase(remove(directions.begin(), directions.end(), spring(i, 1)), directions.end()); //erase-remove idiom

						double spring_trib_length1 = difference_interface[directions[0]] / spring_num_interface;
						double spring_trib_length2 = difference_interface[directions[1]] / spring_num_interface;

						As1 = spring_trib_length1 * spring_trib_length2;
						As2 = spring_trib_length1 * spring_trib_length2;

					}

					//DEVELOPER NOTE: Originally, I was taking into account the deformed spring length into the calculation of K, this makes the deflection-load curve exponential as there is a feedback loop of: spring_length <--> reduced Kn, Ks
					//l1 = (element(element1, dimension3) / 2) + half_sel;
					//l2 = (element(element2, dimension3) / 2) + half_sel;					

					l1 = (element(element1, dimension3) / 2);
					l2 = (element(element2, dimension3) / 2);

					Kn1_spring = Espring * (As1 / l1);
					Kn2_spring = Espring * (As2 / l2);
					Ks1_spring = Gspring * (As1 / l1);
					Ks2_spring = Gspring * (As2 / l2);

					//-----------------------------------------------------------------------------

					sx1 = spring(i, 4);
					sy1 = spring(i, 5);
					sz1 = spring(i, 6);

					sx2 = spring(i, 7);
					sy2 = spring(i, 8);
					sz2 = spring(i, 9);

					ex1 = element(element1, 1);
					ey1 = element(element1, 2);
					ez1 = element(element1, 3);

					ex2 = element(element2, 1);
					ey2 = element(element2, 2);
					ez2 = element(element2, 3);

					//-----------------------------------------------------------------------------

					//SPRING TO LOCAL (ELEMENT)

					//obtain spring local transformation matrix
					MatrixXd springlocal = MatrixXd::Zero(3, 3);
					MatrixXd springlocalfull = MatrixXd::Zero(12, 12);

					if ((fabs(sx1 - sx2) == 0) && (fabs(sy1 - sy2) == 0) && (fabs(sz1 - sz2) == 0)) { //assume that initial condition is met (function transformation_matrix fails if point1 == point2)
						springlocal = transformation_matrix(ex1, ey1, ez1, dummyx, dummyy, dummyz, 0);
						springlocalfull = transformation_matrix(ex1, ey1, ez1, dummyx, dummyy, dummyz, 1);
					}
					else {
						springlocal = transformation_matrix(sx1, sy1, sz1, sx2, sy2, sz2, 0);
						springlocalfull = transformation_matrix(sx1, sy1, sz1, sx2, sy2, sz2, 1);
					}

					//-----------------------------------------------------------------------------

					//global centroid-spring distances -> springlocal reference for K formulation
					VectorXd element1distance = VectorXd::Zero(3);
					element1distance(0) = sx1 - ex1;
					element1distance(1) = sy1 - ey1;
					element1distance(2) = sz1 - ez1;

					VectorXd element1spring = VectorXd::Zero(3);
					element1spring = springlocal * element1distance;

					//-----------------------------------------------------------------------------

					//global centroid-spring distances -> springlocal reference for K formulation
					VectorXd element2distance = VectorXd::Zero(3);
					element2distance(0) = sx2 - ex2;
					element2distance(1) = sy2 - ey2;
					element2distance(2) = sz2 - ez2;

					VectorXd element2spring = VectorXd::Zero(3);
					element2spring = springlocal * element2distance;

					element2spring(0) = -element2spring(0);

					//IMPORTANT DEVELOPER NOTE: because of formulation (see my paper on this), the local x' of element2 HAS to be negative ON PURPOSE!

					//-----------------------------------------------------------------------------

					double Kn1 = fabs(Kn1_spring);
					double Ks11 = fabs(Ks1_spring);
					double Ks12 = fabs(Ks1_spring);

					double Kn2 = fabs(Kn2_spring);
					double Ks21 = fabs(Ks2_spring);
					double Ks22 = fabs(Ks2_spring);

					//DEVELOPER NOTE: See paper on why this is a different formulation (Q Matrix)

					double Qx = (Kn1 * Kn2) / (Kn1 + Kn2);
					double Qy = (Ks11 * Ks21) / (Ks11 + Ks21);
					double Qz = (Ks12 * Ks22) / (Ks12 + Ks22);

					//-----------------------------------------------------------------------------

					//obtain Klocal
					MatrixXd Klocal = MatrixXd::Zero(12, 12);
					Klocal = localmatrix(Qx, Qy, Qz, element1spring(0), element1spring(1), element1spring(2), element2spring(0), element2spring(1), element2spring(2));

					//-----------------------------------------------------------------------------

					//convert Klocal to Kglobal (CE 151: K_global = transpose(tmat) * K_local * t_mat)
					MatrixXd Kglobal = MatrixXd::Zero(12, 12);
					Kglobal = springlocalfull.transpose() * Klocal * springlocalfull;

					//-----------------------------------------------------------------------------

					Kglobalbuffer += Kglobal;
					counter++;
				}

				//-----------------------------------------------------------------------------

#pragma omp single
				std::cout << "\n\n-------------------------------------------------------------\n\n";

				//-----------------------------------------------------------------------------

#pragma omp for ordered
				for (int threads = 0; threads < omp_get_num_threads(); threads++)
				{
#pragma omp ordered
					std::cout << "Size of tripletList_ff_private on thread " << omp_get_thread_num() << ": " << tripletList_ff_private.size() << endl;
					thread_ff_length.push_back(tripletList_ff_private.size());
				}

				//-----------------------------------------------------------------------------

#pragma omp single
				std::cout << endl;

				//-----------------------------------------------------------------------------

#pragma omp for ordered
				for (int threads = 0; threads < omp_get_num_threads(); threads++)
				{
#pragma omp ordered
					std::cout << "Size of tripletList_pf_private on thread " << omp_get_thread_num() << ": " << tripletList_pf_private.size() << endl;
					thread_pf_length.push_back(tripletList_pf_private.size());
				}

				//-----------------------------------------------------------------------------

#pragma omp critical
				{
					tripletList_ff.insert(tripletList_ff.end(), tripletList_ff_private.begin(), tripletList_ff_private.end());
					tripletList_pf.insert(tripletList_pf.end(), tripletList_pf_private.begin(), tripletList_pf_private.end());
				}
			}

			std::cout << "\nSize of tripletList_ff = " << tripletList_ff.size() << "\nSize of tripletList_pf = " << tripletList_pf.size();
			std::cout << "\n\n-------------------------------------------------------------\n\n";

			//-----------------------------------------------------------------------------

			std::cout << "Solved the springs!";
			int spring_time_end = time(NULL);
			int spring_runtime = spring_time_end - spring_time_start;
			std::cout << "\nRuntime of springs: " << spring_runtime << " seconds.\n\n";

			//-----------------------------------------------------------------------------

			string filename_troubleshoot = default_name_increment + "troubleshooting" + ".txt";
			ofstream troubleshoot;

			if (print_list[1] == 1) {
				troubleshoot.open(filename_troubleshoot, ofstream::app);
				if (troubleshoot.is_open()) {
					troubleshoot << "Troubleshooting" << "\ntotal_ele_num = " << total_ele_num;

					for (int i = 0; i < geometry_group_num; i++) {
						int ele_x = geometry(i, 6);
						int ele_y = geometry(i, 7);
						int ele_z = geometry(i, 8);

						int ele_num = ele_x * ele_y * ele_z;

						int spring_num_x = geometry(i, 9);
						int spring_num_y = geometry(i, 10);
						int spring_num_z = geometry(i, 11);

						double xbound1_geometry = geometry(i, 0);
						double xbound2_geometry = geometry(i, 1);
						double ybound1_geometry = geometry(i, 2);
						double ybound2_geometry = geometry(i, 3);
						double zbound1_geometry = geometry(i, 4);
						double zbound2_geometry = geometry(i, 5);

						troubleshoot << "\n\nFOR GROUP NUMBER " << i << ":";
						troubleshoot << "\nele_num = " << ele_num;
						troubleshoot << "\ngrid_sizes: " << ele_x << ", " << ele_y << ", " << ele_z;
						troubleshoot << "\ngeometry bounds: (" << xbound1_geometry << ", " << ybound1_geometry << ", " << zbound1_geometry << ") to (" << xbound2_geometry << ", " << ybound2_geometry << ", " << zbound2_geometry << ")";
						troubleshoot << "\nspring_nums: " << spring_num_x * spring_num_x << ", " << spring_num_y * spring_num_y << ", " << spring_num_z * spring_num_z;
					}

					troubleshoot << "\n\ntotal_dof_num = " << total_dof_num << "\npdof_count = " << pdof_count << "\nfdof_count = " << fdof_count << "\n\ntotal_spring_num = " << total_spring_num;
					troubleshoot << "\n\nreserveCount_ff = " << reserveCount_ff << "\nreserveCount_pf = " << reserveCount_pf;
					troubleshoot << "\n\nSize of tripletList_ff = " << tripletList_ff.size() << "\nSize of tripletList_pf = " << tripletList_pf.size();
					troubleshoot << "\n\nnumber of threads = " << num_threads << endl;

					int ff_stop = thread_ff_length.size();
					int pf_stop = thread_pf_length.size();

					for (int private_iterant = 0; private_iterant < ff_stop; private_iterant++) {
						troubleshoot << endl << "Size of tripletList_ff_private on thread " << private_iterant << ": " << thread_ff_length[private_iterant];
					}
					troubleshoot << endl;
					for (int private_iterant = 0; private_iterant < pf_stop; private_iterant++) {
						troubleshoot << endl << "Size of tripletList_pf_private on thread " << private_iterant << ": " << thread_pf_length[private_iterant];
					}
					troubleshoot << "\n\nRuntime of springs: " << spring_runtime << " seconds.\n";
					troubleshoot.close();
				}
				else {
					std::cout << "\a\n\nERROR: Failed to write to file!\n";
				}
			}

			//-----------------------------------------------------------------------------

			/*

			--- 3 SOLVING ---

			*/

			SpMat Kpf(pdof_count, fdof_count);
			Kpf.setFromTriplets(tripletList_pf.begin(), tripletList_pf.end());

			//-----------------------------------------------------------------------------

			//FLUSH THE U VECTOR

			for (int i = 0; i < total_dof_num; i++) {
				U(i) = 0;
			}

			VectorXd Up = VectorXd::Zero(pdof_count);

			VectorXd Ff = VectorXd::Zero(fdof_count);
			for (int i = 0; i < fdof_count; i++) {
				Ff(i) = F(fdof(i));
			}

			VectorXd Uf = VectorXd::Zero(fdof_count);
			//AEM can also be used for displacement-controlled analysis, but won't be explored here now

			//-----------------------------------------------------------------------------

			//Kpf troubleshooting (very resource-intensive at large sizes)

			if (print_list[10] == 1) {
				std::cout << "Writing the Kpf matrix file...\n";
				string filename_Kpf = default_name_increment + "Kpf" + ".txt";
				ofstream Kpf_file;
				Kpf_file.open(filename_Kpf, ofstream::app);
				if (Kpf_file.is_open()) {
					Kpf_file << MatrixXd(Kpf);
					Kpf_file.close();
				}
				else {
					std::cout << "\a\n\nERROR: Failed to write to file!\n";
				}
				std::cout << "Kpf matrix file successfully written!\n\n";
			}

			//-----------------------------------------------------------------------------

			// Solve unknown displacements

			std::cout << "Solving the matrix...";
			int matrix_time_start = time(NULL);

			Eigen::SparseLU<SpMat, Eigen::COLAMDOrdering<int>> solver;
			Eigen::BiCGSTAB<SpMat, Eigen::IncompleteLUT<double>> solver2;
			//Eigen::BiCGSTAB<SpMat_Parallel, Eigen::IncompleteLUT<double>> solver2;
			//Eigen::UmfPackLU<SpMat> solver3;

			//Eigen solves the equation Ax = b where A is the Sparse Matrix, both x and b are vectors
			//The stiffness equation is F = kU where k is the stiffness matrix, U is the displacement vector, F is the force vector
			//Since we are solving for Uf (represented by variable x in the Eigen formulation) no need to inverse the Kff matrix like in CE 151-CE 152 excel sheets

			MatrixXd Kff_to_print;

			if (matrix_solver == 0) {

				SpMat Kff(fdof_count, fdof_count);
				Kff.setFromTriplets(tripletList_ff.begin(), tripletList_ff.end());

				if (print_list[9] == 1) {

					Kff_to_print = MatrixXd(Kff);

				}

				Kff.makeCompressed();
				solver.analyzePattern(Kff);
				solver.factorize(Kff);
				std::cout << endl << "Check matrix SparseLU factorization success (1 = success): ";
				std::cout << (solver.info() == Eigen::Success) << endl;

				if (solver.info() == Eigen::Success) {
					Uf = solver.solve(Ff);
				}
				else {
					cout << "\a\n\nERROR! Solver failed to solve the matrix. (Check support conditions).\n\n";
					std::cout << "\n\nPress ENTER to exit the program: ";
					std::cin.get();
					return 1;
				}
			}

			if (matrix_solver == 1) {

				SpMat Kff(fdof_count, fdof_count);
				//SpMat_Parallel Kff(fdof_count, fdof_count);
				Kff.setFromTriplets(tripletList_ff.begin(), tripletList_ff.end());


				if (print_list[9] == 1) {

					Kff_to_print = MatrixXd(Kff);

				}

				Kff.makeCompressed();
				solver2.compute(Kff);
				Uf = solver2.solve(Ff);

				std::cout << endl << "Using BiCGSTAB:" << endl;
				std::cout << "#iterations:     " << solver2.iterations() << endl;
				std::cout << "estimated error: " << solver2.error() << endl;

				if (isnan(solver2.error())) {
					cout << "\a\n\nERROR! Solver failed to solve the matrix. (Check support conditions).\n\n";
					std::cout << "\n\nPress ENTER to exit the program: ";
					std::cin.get();
					return 1;
				}
			}

			std::cout << "Solved the matrix!";

			//-----------------------------------------------------------------------------

			//Kff troubleshooting (very resource-intensive at large sizes)

			if (print_list[9] == 1) {

				std::cout << "Writing the Kff matrix file...\n";

				string filename_Kff = default_name_increment + "Kff" + ".txt";
				ofstream Kff_file;
				Kff_file.open(filename_Kff, ofstream::app);
				if (Kff_file.is_open()) {
					Kff_file << Kff_to_print;
					Kff_file.close();
				}
				else {
					std::cout << "\a\n\nERROR: Failed to write to file!\n";
				}
				std::cout << "Kff matrix file successfully written!\n\n";
			}

			//-----------------------------------------------------------------------------

			int matrix_time_end = time(NULL);
			int matrix_runtime = matrix_time_end - matrix_time_start;
			std::cout << "\nRuntime of matrix: " << matrix_runtime << " seconds.\n\n";

			if (print_list[1] == 1) {
				troubleshoot.open(filename_troubleshoot, ofstream::app);
				if (troubleshoot.is_open()) {

					if (matrix_solver == 0) {
						troubleshoot << endl << "Check matrix SparseLU factorization success (1 = success): ";
						troubleshoot << (solver.info() == Eigen::Success) << endl;
					}

					if (matrix_solver == 1) {
						troubleshoot << endl << "Using BiCGSTAB:" << endl;
						troubleshoot << "#iterations:     " << solver2.iterations() << endl;
						troubleshoot << "estimated error: " << solver2.error() << endl;
					}

					troubleshoot << "\nRuntime of matrix: " << matrix_runtime << " seconds.";
					troubleshoot.close();
				}
				else {
					std::cout << "\a\n\nERROR: Failed to write to file!\n";
				}
			}

			//-----------------------------------------------------------------------------

			// Solve support reactions

			MatrixXd Fp = Kpf * Uf;

			//-----------------------------------------------------------------------------

			// Returning the values to full U and F vectors

			int j = 0;
			int k = 0;
			for (int i = 0; i < total_dof_num; i++) {
				if (i == pdof(j)) {
					U(i) = Up(j);
					F(i) = Fp(j);
					if (j + 1 < pdof_count) { j++; }
				}
				else if (i == fdof(k)) {
					U(i) = Uf(k);
					if (k + 1 < fdof_count) { k++; }
				}
				else {
					cout << "error!";
				}
			}

			//-----------------------------------------------------------------------------

			if (print_list[11] == 1) {

				j = 0;
				k = 0;
				string U_filename = default_name_increment + "U" + ".csv";
				ofstream U_file;
				U_file.open(U_filename, ofstream::app);
				if (U_file.is_open()) {
					U_file << "dof,tag,deflection (mmm or radians)\n";
					for (int i = 0; i < total_dof_num; i++) {

						U_file << i << ",";

						if (i == pdof(j)) {
							U_file << "pdof";
							if (j + 1 < pdof_count) { j++; }
						}
						else if (i == fdof(k)) {
							U_file << "fdof";
							if (k + 1 < fdof_count) { k++; }
						}
						else {
							cout << "error!";
						}
						U_file << "," << U(i) << "\n";
					}
					U_file.close();
				}
				else {
					std::cout << "\a\n\nERROR: Failed to write to file!\n";
				}

			}

			//-----------------------------------------------------------------------------

			if (print_list[12] == 1) {

				j = 0;
				k = 0;
				string F_filename = default_name_increment + "F" + ".csv";
				ofstream F_file;
				F_file.open(F_filename, ofstream::app);
				if (F_file.is_open()) {
					F_file << "dof,tag,force (N or Nmm)\n";
					for (int i = 0; i < total_dof_num; i++) {

						F_file << i << ",";

						if (i == pdof(j)) {
							F_file << "pdof";
							if (j + 1 < pdof_count) { j++; }
						}
						else if (i == fdof(k)) {
							F_file << "fdof";
							if (k + 1 < fdof_count) { k++; }
						}
						else {
							cout << "error!";
						}
						F_file << "," << F(i) << "\n";
					}
					F_file.close();
				}
				else {
					std::cout << "\a\n\nERROR: Failed to write to file!\n";
				}

			}

			//-----------------------------------------------------------------------------

			//check if -nan(ind) exists in the U vector: probably means that the Kff matrix was TOO sparse (i.e. too many zeroes due to lack of working springs) and the solver succeeded but gave garbage values

			for (int i = 0; i < total_dof_num; i++) {
				if (isnan(U(i))) {
					std::cout << "\a\n\nNOTICE: A nan(ind) value was found in the U vector (from Ff = Kff * Uf) .\nLikely cause: too many springs have cracked and the system cannot be solved numerically (too many zeroes).\n\n";
					std::cout << "\n\nPress ENTER to exit the program: ";
					std::cin.get();
					return 0;
				}
			}

			//-----------------------------------------------------------------------------

			/*

			--- 4 STRAINS AND STRESSES ---

			*/

			MatrixXd element_stresses = MatrixXd::Zero(total_ele_num, 12);

			/*
			LEGEND

			(in their local coordinates)
			0 - x face cumulative stresses global x (but the -x face has negated values to be positive accdng to sign convention)
			1 - x face cumulative stresses global y
			2 - x face cumulative stresses global z

			3 - y face cumulative stresses global x (but the -x face has negated values to be positive accdng to sign convention)
			4 - y face cumulative stresses global y
			5 - y face cumulative stresses global z

			6 - z face cumulative stresses global x (but the -x face has negated values to be positive accdng to sign convention)
			7 - z face cumulative stresses global y
			8 - z face cumulative stresses global z

			9 - number of contributing springs x face
			10 - number of contributing springs y face
			11 - number of contributing springs z face

			*/

			//update spring, dummy vector, and element coordinates due to new deflections
			//get new spring coordinates

			for (int i = 0; i < total_spring_num; i++) {

				//get elements

				int element1 = spring(i, 2);
				int element2 = spring(i, 3);

				//-----------------------------------------------------------------------------

				//get dofs
				int current_dof_1 = 6 * element1;
				int current_dof_2 = 6 * element2;

				//set-up which dofs are involved
				VectorXi current_dofs = VectorXi::Zero(12);
				for (int j = 0; j < 6; j++) {
					current_dofs(j) = current_dof_1 + j;
				}
				for (int j = 0; j < 6; j++) {
					int k = j + 6;
					current_dofs(k) = current_dof_2 + j;
				}

				//-----------------------------------------------------------------------------

				//get current deflections from matrix solution

				double deltax_1 = U(current_dofs(0));
				double deltay_1 = U(current_dofs(1));
				double deltaz_1 = U(current_dofs(2));

				double thetax_1 = U(current_dofs(3));
				double thetay_1 = U(current_dofs(4));
				double thetaz_1 = U(current_dofs(5));

				double deltax_2 = U(current_dofs(6));
				double deltay_2 = U(current_dofs(7));
				double deltaz_2 = U(current_dofs(8));

				double thetax_2 = U(current_dofs(9));
				double thetay_2 = U(current_dofs(10));
				double thetaz_2 = U(current_dofs(11));

				//setup coordinates of spring ends setting element centroid -> (0,0,0)
				double x1_from_centroid = spring(i, 4) - element(element1, 1);
				double y1_from_centroid = spring(i, 5) - element(element1, 2);
				double z1_from_centroid = spring(i, 6) - element(element1, 3);

				double x2_from_centroid = spring(i, 7) - element(element2, 1);
				double y2_from_centroid = spring(i, 8) - element(element2, 2);
				double z2_from_centroid = spring(i, 9) - element(element2, 3);

				//-----------------------------------------------------------------------------

				VectorXd rotational_deflections_end1 = rotation_to_displacement(x1_from_centroid, y1_from_centroid, z1_from_centroid, thetax_1, thetay_1, thetaz_1);
				VectorXd rotational_deflections_end2 = rotation_to_displacement(x2_from_centroid, y2_from_centroid, z2_from_centroid, thetax_2, thetay_2, thetaz_2);

				deltax_1 += rotational_deflections_end1(0);
				deltay_1 += rotational_deflections_end1(1);
				deltaz_1 += rotational_deflections_end1(2);

				deltax_2 += rotational_deflections_end2(0);
				deltay_2 += rotational_deflections_end2(1);
				deltaz_2 += rotational_deflections_end2(2);

				if (isnan(deltax_1) || isnan(deltay_1) || isnan(deltaz_1) || isnan(deltax_2) || isnan(deltay_2) || isnan(deltaz_2)) {
					std::cout << "\a\n\nERROR! At strains and stresses: one of the rotation adjustments evaluated as NaN.\n\n";
					std::cout << "Input at error call: " << x1_from_centroid << " " << y1_from_centroid << " " << z1_from_centroid << " " << thetax_1 << " " << thetay_1 << " " << thetaz_1 << endl;
					std::cout << "Input at error call: " << x2_from_centroid << " " << y2_from_centroid << " " << z2_from_centroid << " " << thetax_2 << " " << thetay_2 << " " << thetaz_2 << endl << endl;
					std::cout << "Output at error call: " << deltax_1 << " " << deltay_1 << " " << deltaz_1 << " " << deltax_2 << " " << deltay_2 << " " << deltaz_2 << endl;
					std::cout << "\n\nPress ENTER to exit the program: ";
					std::cin.get();
					return 1;
				}

				//-----------------------------------------------------------------------------

				if (element(element1, 21) == 0) {
					spring(i, 4) = spring(i, 4) + deltax_1;
					spring(i, 5) = spring(i, 5) + deltay_1;
					spring(i, 6) = spring(i, 6) + deltaz_1;
				}

				if (element(element2, 21) == 0) {
					spring(i, 7) = spring(i, 7) + deltax_2;
					spring(i, 8) = spring(i, 8) + deltay_2;
					spring(i, 9) = spring(i, 9) + deltaz_2;
				}

				//-----------------------------------------------------------------------------
			}

			//get new element dummy vector coordinates

			for (int i = 0; i < total_ele_num; i++) {

				if (element(i, 21) == 1) {
					//skip
				}
				else
				{
					int current_dof_0 = i * 6;
					int current_dof_1 = current_dof_0 + 1;
					int current_dof_2 = current_dof_0 + 2;
					int current_dof_3 = current_dof_0 + 3;
					int current_dof_4 = current_dof_0 + 4;
					int current_dof_5 = current_dof_0 + 5;

					//-----------------------------------------------------------------------------

					double deltax = U(current_dof_0);
					double deltay = U(current_dof_1);
					double deltaz = U(current_dof_2);

					double thetax = U(current_dof_3);
					double thetay = U(current_dof_4);
					double thetaz = U(current_dof_5);

					//treat the dummy vectors

					VectorXd rotational_deflections_dummyalongx = rotation_to_displacement(element(i, 10), element(i, 11), element(i, 12), thetax, thetay, thetaz);
					VectorXd rotational_deflections_dummyalongy = rotation_to_displacement(element(i, 13), element(i, 14), element(i, 15), thetax, thetay, thetaz);
					VectorXd rotational_deflections_dummyalongz = rotation_to_displacement(element(i, 16), element(i, 17), element(i, 18), thetax, thetay, thetaz);

					//note: The dummy vectors are defined as distance from centroid, not as spatial coordinates. Therefore, no need to add the translational deflections

					element(i, 10) += rotational_deflections_dummyalongx(0);
					element(i, 11) += rotational_deflections_dummyalongx(1);
					element(i, 12) += rotational_deflections_dummyalongx(2);

					element(i, 13) += rotational_deflections_dummyalongy(0);
					element(i, 14) += rotational_deflections_dummyalongy(1);
					element(i, 15) += rotational_deflections_dummyalongy(2);

					element(i, 16) += rotational_deflections_dummyalongz(0);
					element(i, 17) += rotational_deflections_dummyalongz(1);
					element(i, 18) += rotational_deflections_dummyalongz(2);

					//-----------------------------------------------------------------------------

					//get new element coordinates
					//note: element centroids don't rotate

					element(i, 1) += deltax;
					element(i, 2) += deltay;
					element(i, 3) += deltaz;
					element(i, 4) += thetax;
					element(i, 5) += thetay;
					element(i, 6) += thetaz;
				}
			}

			//-----------------------------------------------------------------------------

			if (print_list[13] == 1) {

				string filename_element = default_name_increment + "element" + ".csv";
				ofstream element_file;
				element_file.open(filename_element, ofstream::app);
				if (element_file.is_open()) {
					element_file << "element_number,x,y,z,thetax,thetay,thetaz,x_dimension,y_dimension,z_dimension,x_face_x,x_face_y,x_face_z,y_face_x,y_face_y,y_face_z,z_face_x,z_face_y,z_face_z,geometry_group,interface_index,support_element\n";
					element_file << element.format(CSVFormat);
					element_file.close();
				}
				else {
					std::cout << "\a\n\nERROR: Failed to write to file!\n";
				}
			}

			//make a copy of the F vector before flushing to use on element_drawer further on in the increment

			VectorXd F_copy = F;

			//-----------------------------------------------------------------------------

			//FLUSH THE F VECTOR - RESET FROM CRACKED SPRINGS REDISTRIBUTION

			for (int i = 0; i < total_dof_num; i++) {
				F(i) = 0;
			}

			//-----------------------------------------------------------------------------

			std::cout << "-------------------------------------------------------------\n\n";
			std::cout << "Solving the strains and stresses... \n";
			int stress_time_start = time(NULL);

			//-----------------------------------------------------------------------------

			for (int i = 0; i < total_spring_num; i++) {

				int element1 = spring(i, 2);
				int element2 = spring(i, 3);

				//-----------------------------------------------------------------------------
				//OBTAIN SPRING STRAINS v3: SINGLE SPRING METHOD (SSM), a simplification of the original AEM methodology
				//-----------------------------------------------------------------------------

				double trib_length1, trib_length2;

				//get trib_length which was initially centroid -> spring location accdng to global=local axes
				if (spring(i, 1) == 0) {
					trib_length1 = element(element1, 7) / 2.0;
					trib_length2 = element(element2, 7) / 2.0;
				}
				else if (spring(i, 1) == 1) {
					trib_length1 = element(element1, 8) / 2.0;
					trib_length2 = element(element2, 8) / 2.0;
				}
				else {
					trib_length1 = element(element1, 9) / 2.0;
					trib_length2 = element(element2, 9) / 2.0;
				}

				double excess_length = trib_length1 + trib_length2;

				//-----------------------------------------------------------------------------

				//This part is similar to the stiffness matrix construction part earlier

				int dimension1, dimension2, dimension3;
				double dummyx1, dummyy1, dummyz1, dummyx2, dummyy2, dummyz2;
				double sx1, sy1, sz1, sx2, sy2, sz2;
				double ex1, ey1, ez1, ex2, ey2, ez2;
				int spring_num1_element1, spring_num2_element1;
				int spring_num1_element2, spring_num2_element2;

				int geometry_group1 = element(element1, 19);
				int geometry_group2 = element(element2, 19);

				// 13 - a, 14 - b, 15 - c
				if (spring(i, 1) == 0) {
					dimension1 = 8;
					dimension2 = 9;
					dimension3 = 7;

					dummyx1 = element(element1, 10);
					dummyy1 = element(element1, 11);
					dummyz1 = element(element1, 12);

					dummyx2 = -element(element2, 10); //negative because pointing backwards (element2 -> element1)
					dummyy2 = -element(element2, 11);
					dummyz2 = -element(element2, 12);

					spring_num1_element1 = geometry(geometry_group1, 10);
					spring_num2_element1 = geometry(geometry_group1, 11);
					spring_num1_element2 = geometry(geometry_group2, 10);
					spring_num2_element2 = geometry(geometry_group2, 11);

				}
				else if (spring(i, 1) == 1) {
					dimension1 = 7;
					dimension2 = 9;
					dimension3 = 8;

					dummyx1 = element(element1, 13);
					dummyy1 = element(element1, 14);
					dummyz1 = element(element1, 15);

					dummyx2 = -element(element2, 13);
					dummyy2 = -element(element2, 14);
					dummyz2 = -element(element2, 15);

					spring_num1_element1 = geometry(geometry_group1, 9);
					spring_num2_element1 = geometry(geometry_group1, 11);
					spring_num1_element2 = geometry(geometry_group2, 9);
					spring_num2_element2 = geometry(geometry_group2, 11);
				}
				else {
					dimension1 = 7;
					dimension2 = 8;
					dimension3 = 9;

					dummyx1 = element(element1, 16);
					dummyy1 = element(element1, 17);
					dummyz1 = element(element1, 18);

					dummyx2 = -element(element2, 16);
					dummyy2 = -element(element2, 17);
					dummyz2 = -element(element2, 18);

					spring_num1_element1 = geometry(geometry_group1, 9);
					spring_num2_element1 = geometry(geometry_group1, 10);
					spring_num1_element2 = geometry(geometry_group2, 9);
					spring_num2_element2 = geometry(geometry_group2, 10);
				}

				//-----------------------------------------------------------------------------

				sx1 = spring(i, 4);
				sy1 = spring(i, 5);
				sz1 = spring(i, 6);

				sx2 = spring(i, 7);
				sy2 = spring(i, 8);
				sz2 = spring(i, 9); //these are different from the K stiffness for-loop (we are using the current coordinates now after deflections)

				ex1 = element(element1, 1);
				ey1 = element(element1, 2);
				ez1 = element(element1, 3);

				ex2 = element(element2, 1);
				ey2 = element(element2, 2);
				ez2 = element(element2, 3); //also using the new ones

				//obtain spring local transformation matrix
				MatrixXd springlocal = MatrixXd::Zero(3, 3);
				MatrixXd springlocal_full = MatrixXd::Zero(12, 12);

				if ((fabs(sx1 - sx2) == 0) && (fabs(sy1 - sy2) == 0) && (fabs(sz1 - sz2) == 0)) { //function transformation_matrix fails if point1 == point2
					springlocal = transformation_matrix(0, 0, 0, dummyx1, dummyy1, dummyz1, 0);
					springlocal_full = transformation_matrix(0, 0, 0, dummyx1, dummyy1, dummyz1, 1);
				}
				else {
					springlocal = transformation_matrix(sx1, sy1, sz1, sx2, sy2, sz2, 0);
					springlocal_full = transformation_matrix(sx1, sy1, sz1, sx2, sy2, sz2, 1);
				}

				MatrixXd initial_direction_of_spring = transformation_matrix(0, 0, 0, dummyx1, dummyy1, dummyz1, 0);

				//get global vector of step n (both ends)
				double difference_x = sx2 - sx1;
				double difference_y = sy2 - sy1;
				double difference_z = sz2 - sz1;

				VectorXd spring_n_global = VectorXd::Zero(3);

				spring_n_global(0) = difference_x;
				spring_n_global(1) = difference_y;
				spring_n_global(2) = difference_z;

				//check if tension or compression
				VectorXd check_tens_or_comp = VectorXd::Zero(3);
				check_tens_or_comp = initial_direction_of_spring * spring_n_global; //compare spring now to its initial state, which was along the x face, or y face, or z face

				double sign_of_end = 0;
				if (check_tens_or_comp(0) != 0) {
					sign_of_end = check_tens_or_comp(0) / fabs(check_tens_or_comp(0)); //get only the sign, positive if tension, negative if compression
				}

				// get magnitude
				double magnitude = sqrt(difference_x * difference_x + difference_y * difference_y + difference_z * difference_z);

				//-----------------------------------------------------------------------------

				double strain_xprime = sign_of_end * magnitude / excess_length;

				spring(i, 13) = strain_xprime;

				//-----------------------------------------------------------------------------

				if (spring(i, 10) == 1) { //cracked springs don't carry any stress in the program for now
					spring(i, 14) = 0;
				}
				else {
					double stress_normal = 0, stress_comp = 0, stress_tens = 0, stress_shear_y = 0, stress_shear_z = 0;

					//get material nonlinearity setting

					int spring_material = spring(i, 11);
					int material_behavior = material(spring_material, 5);
					int material_nonlinearity_choice = material(spring_material, 6);

					//note: we use the material E (not tangent stiffness) because that's how the Maekawa model is calculated
					double Espring = material(spring_material, 0);
					double compressive_strength = material(spring_material, 3);
					double tensile_strength = material(spring_material, 4);

					if (material_nonlinearity_choice == 0) { //concrete

						double epsilon = strain_xprime;
						double epsilon_c = 2.0 * -compressive_strength / Espring;
						double ratio = epsilon / epsilon_c;
						double epsilon_p = epsilon_c * (ratio - (20.0 / 7.0) * (1 - exp(-0.35 * ratio)));
						double K = exp(-0.73 * (ratio) * (1.0 - exp(-1.25 * (ratio)))); //reduction factor from Maekawa

						//Maekawa - compressive loading part
						if (epsilon < 0) {
							stress_comp = K * Espring * (epsilon - epsilon_p);

							if (fabs(stress_comp) > compressive_strength) {
								stress_comp = -compressive_strength;
							}
						}
						//Maekawa - tensile loading part
						else if (epsilon > 0) {
							double Rf = K;
							//Can change this to 1, or K, or K^3 as other options according to https://manuals.dianafea.com/d104/Theory/Theoryse300.html
							double f_TSC = Espring; //supposed to be a function to take into account the tension softening/stiffening
							//therefore, the program is not yet suited to do analysis of recontact, crack closing, and element collision, but AEM is capable of it according to literature
							stress_tens = Rf * f_TSC * (epsilon - epsilon_p);
							if (stress_tens > tensile_strength) {
								stress_tens = tensile_strength;
							}
						}

						//-----------------------------------------------------------------------------

					}
					else if (material_nonlinearity_choice == 1) { //steel
						if (spring(i, 10) == 0) { // elastic steel
							if (strain_xprime < 0) {
								stress_comp = strain_xprime * Espring;

								if (fabs(stress_comp) > compressive_strength) {
									stress_comp = -compressive_strength;
								}
							}
							else if (strain_xprime > 0) {
								stress_tens = strain_xprime * Espring;

								if (stress_tens > tensile_strength) {
									stress_tens = tensile_strength;
								}
							}
						}

						else if (spring(i, 10) == 2) { // yielded steel

							//get stress at yield:
							double stress_yield_tens = tensile_strength;
							double stress_yield_comp = -compressive_strength;

							//get strain at yield
							double strain_yield_tens = stress_yield_tens / Espring;
							double strain_yield_comp = stress_yield_comp / Espring;

							if (strain_xprime < 0) { //compression
								//get strain_current - strain_yield
								double difference_strain = strain_xprime - strain_yield_comp; //this will be negative

								//get stress from increase in strain with reduced E
								double stress_add = difference_strain * Espring / 100.0; //this will be negative
								//reduced E as per Meguro & Tagel-Din 2000

								stress_comp = -compressive_strength + stress_add;
							}
							else if (strain_xprime > 0) { //tension
								//get strain_current - strain_yield
								double difference_strain = strain_xprime - strain_yield_tens; //this will be positive

								//get stress from increase in strain with reduced E
								double stress_add = difference_strain * Espring / 100.0; //this will be positive
								//reduced E as per Meguro & Tagel-Din 2000

								stress_tens = tensile_strength + stress_add;
							}
						}
						else if (spring(i, 10) == 3) { // strain hardening steel	
							//strain hardening region
							//no implementation yet but would like to study more
						}
					}
					else if (material_nonlinearity_choice == 2) { // generic case (linear all the way)
						stress_comp = strain_xprime * Espring;

						if (fabs(stress_comp) > compressive_strength) {
							stress_comp = -compressive_strength;
						}

						stress_tens = strain_xprime * Espring;

						if (stress_tens > tensile_strength) {
							stress_tens = tensile_strength;
						}
					}

					//-----------------------------------------------------------------------------

					if (strain_xprime < 0) {
						stress_normal = stress_comp;
					}
					else if (strain_xprime > 0) {
						stress_normal = stress_tens;
					}

					spring(i, 14) = stress_normal;

					//-----------------------------------------------------------------------------

					/*
					--- 5 FAILURE CRITERIA ---
					*/

					//In single spring model, we only have to worry about elongation -> x' strain -> uniaxial behavior of spring

					double principal_1 = max(stress_normal, 0.0);
					double principal_2 = min(stress_normal, 0.0);

					//-----------------------------------------------------------------------------

					double strength_check_1, strength_check_2;

					//DEVELOPER NOTE: temporary set-up due to input .txt configuration: concrete checks for ultimate failure, steel checks for yield (no feature for ultimate steel failure yet)
					strength_check_1 = tensile_strength;
					strength_check_2 = compressive_strength;


					//DEVELOPER NOTE: if allow_failure == 0, then the springs will only get tagged (e.g. failed state), but they will all still contribute as before to the Kff matrix

					//check if ends failed - concrete
					/*
					Reference used:
					https://www.pci.org/PCI_Docs/Design_Resources/Guides_and_manuals/references/bridge_design_manual/JL-88-July-August_Cracks_and_Crack_Control_in_Concrete_Structures.pdf
					"Concrete cracks when the tensile strain exceeds 0.010 to 0.012 percent, this limiting strain is essentially independent of concrete strength"
					use 0.012%
					*/
					//however, include also equivalent strain for UHPC cases

					if (material_nonlinearity_choice == 0) { // concrete
						if (strain_xprime > max(0.00012, tensile_strength / Espring)) {
							spring(i, 10) = 1; //cracked
						}
						if (fabs(principal_2) >= strength_check_2) {
							spring(i, 10) = 1; //cracked
						}
					}

					if (material_nonlinearity_choice == 1) { // steel
						if ((principal_1 >= strength_check_1) || (fabs(principal_2) >= strength_check_2)) {
							spring(i, 10) = 2; //yield
						}

						//DEVELOPER NOTE: use rupture strain from here:
						//https://www.researchgate.net/publication/273405702_Flexural_Behavior_of_Concrete_Beams_Strengthened_with_New_Prestressed_Carbon-Basalt_Hybrid_Fiber_Sheets

						if (fabs(strain_xprime) > 0.15) {
							spring(i, 10) = 1; //rupture

							//DEVELOPER NOTE: since only the yielding where stress-strain curve is upward by E/100, and ultimate strength -> necking strain with reducing stress curve is not yet modeled by program, give a rough estimate for practical purposes
							if (strain_xprime > 0) { //tensile strain
								stress_normal = tensile_strength + 200; //https://www.structuralguide.com/tensile-strength-of-rebar/
							}
							else {
								stress_normal = -(compressive_strength)-200;
							}
						}
					}

					if (material_nonlinearity_choice == 2) { //generic
						if ((principal_1 >= strength_check_1) || (fabs(principal_2) >= strength_check_2)) {
							if (material_behavior == 0) { // brittle
								spring(i, 10) = 1; //failed
							}
							else if (material_behavior == 1) { //ductile
								spring(i, 10) = 2; //yield

								//DEVELOPER NOTE: this is hardcoded for now, but maybe we can make it so that the user inputs the rupture strain of any ductile material
								//https://www.engineeringarchives.com/les_mom_brittleductile.html

								if (fabs(strain_xprime) > 0.05) {
									spring(i, 10) = 1; //rupture
								}
							}
						}
					}

					//-----------------------------------------------------------------------------
					// Crack Force and Moment Redistribution (imagine snapping a pencil in two, you get forces from the crack because of the released energy)
					//-----------------------------------------------------------------------------

					//get element local transformation matrices
					MatrixXd element_1_local = transformation_matrix(0, 0, 0, dummyx1, dummyy1, dummyz1, 0);
					MatrixXd element_2_local = transformation_matrix(0, 0, 0, dummyx2, dummyy2, dummyz2, 0);

					bool force_redistribution = true;

					if (spring(i, 10) == 1) {
						cracked_spring_count++;

						if (force_redistribution) {

							// 13 - a, 14 - b, 15 - c
							if (spring(i, 1) == 0) {
								spring_num1_element1 = geometry(geometry_group1, 10);
								spring_num2_element1 = geometry(geometry_group1, 11);
								spring_num1_element2 = geometry(geometry_group2, 10);
								spring_num2_element2 = geometry(geometry_group2, 11);

							}
							else if (spring(i, 1) == 1) {
								spring_num1_element1 = geometry(geometry_group1, 9);
								spring_num2_element1 = geometry(geometry_group1, 11);
								spring_num1_element2 = geometry(geometry_group2, 9);
								spring_num2_element2 = geometry(geometry_group2, 11);
							}
							else {
								spring_num1_element1 = geometry(geometry_group1, 9);
								spring_num2_element1 = geometry(geometry_group1, 10);
								spring_num1_element2 = geometry(geometry_group2, 9);
								spring_num2_element2 = geometry(geometry_group2, 10);
							}

							//-----------------------------------------------------------------------------

							double As1 = (element(element1, dimension1) / spring_num1_element1) * (element(element1, dimension2) / spring_num2_element1);
							double As2 = (element(element2, dimension1) / spring_num1_element2) * (element(element2, dimension2) / spring_num2_element2);

							if (spring(i, 12) != -1) {
								//get interface data

								double x_coord1 = interfaces[spring(i, 12)].x_coord1;
								double x_coord2 = interfaces[spring(i, 12)].x_coord2;

								double y_coord1 = interfaces[spring(i, 12)].y_coord1;
								double y_coord2 = interfaces[spring(i, 12)].y_coord2;

								double z_coord1 = interfaces[spring(i, 12)].z_coord1;
								double z_coord2 = interfaces[spring(i, 12)].z_coord2;

								double differencex_interface = x_coord2 - x_coord1;
								double differencey_interface = y_coord2 - y_coord1;
								double differencez_interface = z_coord2 - z_coord1;

								vector<double> difference_interface{ differencex_interface, differencey_interface, differencez_interface };

								double spring_num_interface = interfaces[spring(i, 12)].spring_num;

								vector<int> directions{ 0, 1, 2 };
								directions.erase(remove(directions.begin(), directions.end(), spring(i, 1)), directions.end()); //erase-remove idiom

								double spring_trib_length1 = difference_interface[directions[0]] / spring_num_interface;
								double spring_trib_length2 = difference_interface[directions[1]] / spring_num_interface;

								As1 = spring_trib_length1 * spring_trib_length2;
								As2 = spring_trib_length1 * spring_trib_length2;

							}

							//get forces from stresses

							double force_element1 = stress_normal * As1; //unsigned for now, will add correct sign in next step
							double force_element2 = stress_normal * As2;


							//get forces global
							/*
							sign convention :
							spring: -x 0---------> +x
							tension:  (negative value) <=== 0---------> ===> (positive value)
							compression:  (positive value) ===> 0---------> <=== (negative value)
							*/

							VectorXd force_element1_spring = VectorXd::Zero(3);
							force_element1_spring(0) = -force_element1;

							VectorXd force_element2_spring = VectorXd::Zero(3);
							force_element2_spring(0) = force_element2;

							VectorXd force_element1_global = springlocal.transpose() * force_element1_spring;
							VectorXd force_element2_global = springlocal.transpose() * force_element2_spring;

							//get lever arms global

							VectorXd element1distance = VectorXd::Zero(3);
							element1distance(0) = sx1 - ex1; //x1
							element1distance(1) = sy1 - ey1; //y1
							element1distance(2) = sz1 - ez1; //z1

							VectorXd element2distance = VectorXd::Zero(3);
							element2distance(0) = sx2 - ex2; //x2
							element2distance(1) = sy2 - ey2; //y2
							element2distance(2) = sz2 - ez2; //z2

							//complete all 12 forces/moments to apply

							VectorXd redistribution = VectorXd::Zero(12);

							redistribution(0) = force_element1_global(0); //Fx
							redistribution(1) = force_element1_global(1); //Fy
							redistribution(2) = force_element1_global(2); //Fz

							//Mx = y Fz - z Fy
							//My = z Fx - x Fz
							//Mz = x Fy - y Fx

							redistribution(3) = element1distance(1) * force_element1_global(2) - element1distance(2) * force_element1_global(1);
							redistribution(4) = element1distance(2) * force_element1_global(0) - element1distance(0) * force_element1_global(2);
							redistribution(5) = element1distance(0) * force_element1_global(1) - element1distance(1) * force_element1_global(0);

							redistribution(6) = force_element2_global(0);
							redistribution(7) = force_element2_global(1);
							redistribution(8) = force_element2_global(2);

							redistribution(9) = element2distance(1) * force_element2_global(2) - element2distance(2) * force_element2_global(1);
							redistribution(10) = element2distance(2) * force_element2_global(0) - element2distance(0) * force_element2_global(2);
							redistribution(11) = element2distance(0) * force_element2_global(1) - element2distance(1) * force_element2_global(0);

							//apply on next increment

							//get dofs
							int current_dof_1 = 6 * element1;
							int current_dof_2 = 6 * element2;

							//set-up which dofs are involved
							VectorXi current_dofs = VectorXi::Zero(12);
							for (int j = 0; j < 6; j++) {
								current_dofs(j) = current_dof_1 + j;
							}
							for (int j = 0; j < 6; j++) {
								int k = j + 6;
								current_dofs(k) = current_dof_2 + j;
							}
							for (int j = 0; j < 12; j++) {
								F(current_dofs(j)) += redistribution(j);
							}

						}
					}


					//-----------------------------------------------------------------------------
					// Element Stresses and Principal Stresses
					//-----------------------------------------------------------------------------

					//END1
					VectorXd stress_local_end1 = VectorXd::Zero(3);
					stress_local_end1(0) = stress_normal;
					stress_local_end1(1) = 0;
					stress_local_end1(2) = 0;

					//END2
					VectorXd stress_local_end2 = VectorXd::Zero(3);
					stress_local_end2(0) = stress_normal;
					stress_local_end2(1) = 0;
					stress_local_end2(2) = 0;

					//summarize spring stresses to element stresses

					//convert from spring local to global
					VectorXd element_stresses_element1_global = springlocal.transpose() * stress_local_end1;
					VectorXd element_stresses_element2_global = (-1.0) * springlocal.transpose() * stress_local_end2;

					//global to element local
					VectorXd element_stresses_element1 = element_1_local * stress_local_end1;
					VectorXd element_stresses_element2 = (-1.0) * element_2_local * stress_local_end2;

					//indices of element_stresses according to spring direction
					//x - 0, 3, 6
					//y - 1, 4, 7
					//z - 2, 5, 8

					//contributing - 9, 10, 11

					int spring_index_x = spring(i, 1) * 3;
					int spring_index_y = spring_index_x + 1;
					int spring_index_z = spring_index_y + 1;

					element_stresses(element1, spring_index_x) += element_stresses_element1(0);
					element_stresses(element1, spring_index_y) += element_stresses_element1(1);
					element_stresses(element1, spring_index_z) += element_stresses_element1(2);

					element_stresses(element2, spring_index_x) += element_stresses_element2(0);
					element_stresses(element2, spring_index_y) += element_stresses_element2(1);
					element_stresses(element2, spring_index_z) += element_stresses_element2(2);

					int spring_index_contrib = spring(i, 1) + 9;

					element_stresses(element1, spring_index_contrib) += 1;
					element_stresses(element2, spring_index_contrib) += 1;
				}
			}

			//compute average normal stresses and principal stresses

			std::cout << "Computing the principal stresses... \n";

			for (int i = 0; i < total_ele_num; i++) {
				double average_normal_x = element_stresses(i, 0) / element_stresses(i, 9);
				double average_normal_y = element_stresses(i, 4) / element_stresses(i, 10);
				double average_normal_z = element_stresses(i, 8) / element_stresses(i, 11);

				if (isnan(average_normal_x)) {
					average_normal_x = 0;
				}
				if (isnan(average_normal_y)) {
					average_normal_y = 0;
				}
				if (isnan(average_normal_z)) {
					average_normal_z = 0;
				}

				//DEVELOPER NOTE:
				//theoretically, the shear xy is equal to the shear yx
				//numerically, the numbers don't match but are somewhat close
				//in this code, get the average numerically for practical purposes
				double average_shear_xy = (element_stresses(i, 1) + element_stresses(i, 3)) / (element_stresses(i, 9) + element_stresses(i, 10));
				double average_shear_xz = (element_stresses(i, 2) + element_stresses(i, 6)) / (element_stresses(i, 9) + element_stresses(i, 11));
				double average_shear_yz = (element_stresses(i, 5) + element_stresses(i, 7)) / (element_stresses(i, 10) + element_stresses(i, 11));

				if (isnan(average_shear_xy)) {
					average_shear_xy = 0;
				}
				if (isnan(average_shear_xz)) {
					average_shear_xz = 0;
				}
				if (isnan(average_shear_yz)) {
					average_shear_yz = 0;
				}

				MatrixXd stress_tensor = MatrixXd::Zero(3, 3);

				stress_tensor(0, 0) = average_normal_x;
				stress_tensor(1, 1) = average_normal_y;
				stress_tensor(2, 2) = average_normal_z;

				stress_tensor(0, 1) = average_shear_xy;
				stress_tensor(0, 2) = average_shear_xz;
				stress_tensor(1, 2) = average_shear_yz;

				stress_tensor(1, 0) = average_shear_xy;
				stress_tensor(2, 0) = average_shear_xz;
				stress_tensor(2, 1) = average_shear_yz;

				//get eigenvalues -> those are the principal stresses
				Eigen::SelfAdjointEigenSolver<MatrixXd> principal_eigens;
				principal_eigens.compute(stress_tensor, false); //false, meaning don't get the eigenvectors, just the eigenvalues
				VectorXd eigenvalues = principal_eigens.eigenvalues();

				element_principal_stresses(i, 0) = eigenvalues(2); //highest value
				element_principal_stresses(i, 1) = eigenvalues(1); //middle value
				element_principal_stresses(i, 2) = eigenvalues(0); //lowest value

				element_principal_stresses(i, 3) = average_normal_x;
				element_principal_stresses(i, 4) = average_normal_y;
				element_principal_stresses(i, 5) = average_normal_z;

				element_principal_stresses(i, 6) = average_shear_xy;
				element_principal_stresses(i, 7) = average_shear_xz;
				element_principal_stresses(i, 8) = average_shear_yz;

				vector<double> eigenvalues_abs{ fabs(eigenvalues(0)), fabs(eigenvalues(1)), fabs(eigenvalues(2)) };

				auto maximum_index = max_element(eigenvalues_abs.begin(), eigenvalues_abs.end());
				int maximum_index_int = distance(eigenvalues_abs.begin(), maximum_index);

				element_principal_stresses(i, 9) = eigenvalues(maximum_index_int);
				element_principal_stresses(i, 10) = element_stresses(i, 9) + element_stresses(i, 10) + element_stresses(i, 11);
			}

			std::cout << "Principal stresses computed!\n";

			//-----------------------------------------------------------------------------

			//using F_copy because it was flushed before getting the cracked forces for the next increment
			if (print_list[17] == 1) {
				element_drawer(default_name_increment, element, total_ele_num, U, F_copy, element_principal_stresses);
			}

			//-----------------------------------------------------------------------------

			if (print_list[18] == 1) {
				//too much storage space consumed!
				spring_drawer(default_name_increment, spring, total_spring_num, 0, -1);
			}

			//-----------------------------------------------------------------------------

			if (print_list[14] == 1) {
				//very resource-intensive
				string filename_spring = default_name_increment + "spring" + ".csv";
				ofstream spring_file;
				spring_file.open(filename_spring, ofstream::app);
				if (spring_file.is_open()) {
					spring_file << "spring_number,spring_direction,element1,element2,x_end1,y_end1,z_end1,x_end2,y_end2,z_end2,spring_condition,spring_material,interface_index,strain,stress\n";
					spring_file << spring.format(CSVFormat);
					spring_file.close();
				}
				else {
					std::cout << "\a\n\nERROR: Failed to write to file!\n";
				}
			}

			//-----------------------------------------------------------------------------

			//END OF CURRENT LOAD INCREMENT

			//-----------------------------------------------------------------------------

			int stress_time_end = time(NULL);
			int stress_runtime = stress_time_end - stress_time_start;
			std::cout << "\nRuntime of stresses: " << stress_runtime << " seconds.\n";
			std::cout << "\nCumulative number of cracked springs: " << cracked_spring_count << " of " << total_spring_num << endl;

			//-----------------------------------------------------------------------------

			int increment_time_end = time(NULL);
			int increment_runtime = increment_time_end - increment_time_start;
			std::cout << "\n-------------------------------------------------------------\n";
			std::cout << "\nRuntime of load increment " << load_increment << ": " << increment_runtime << " seconds.\n" << "\n-------------------------------------------------------------\n\n";

			std::cout << "Writing the load increment files...\n";

			//-----------------------------------------------------------------------------

			if (print_list[1] == 1) {
				troubleshoot.open(filename_troubleshoot, ofstream::app);
				if (troubleshoot.is_open()) {
					troubleshoot << "\nRuntime of stresses: " << stress_runtime << " seconds.\n";
					troubleshoot << "\nCumulative number of cracked springs: " << cracked_spring_count << " of " << total_spring_num << endl;
					troubleshoot << "\nRuntime of increment: " << increment_runtime << " seconds.";
					troubleshoot.close();
				}
				else {
					std::cout << "\a\n\nERROR: Failed to write to file!\n";
				}
			}

			//-----------------------------------------------------------------------------

			if (print_list[15] == 1) {
				//too much storage space consumed!
				string filename_spring_cracked = default_name_increment + "spring_cracked.csv";
				ofstream spring_cracked_file;
				spring_cracked_file.open(filename_spring_cracked, ofstream::app);
				if (spring_cracked_file.is_open()) {
					spring_cracked_file << "spring_num,x_end1,y_end1,z_end1,x_end2,y_end2,z_end2,spring_direction,length_of_crack_x,length_of_crack_y,length_of_crack_z\n";
					for (int i = 0; i < total_spring_num; i++) {
						if (spring(i, 10) == 1) {

							double sx1 = spring(i, 4);
							double sy1 = spring(i, 5);
							double sz1 = spring(i, 6);

							double sx2 = spring(i, 7);
							double sy2 = spring(i, 8);
							double sz2 = spring(i, 9);

							double differencex = sx2 - sx1;
							double differencey = sy2 - sy1;
							double differencez = sz2 - sz1;

							spring_cracked_file << spring(i, 0) << ",";
							spring_cracked_file << spring(i, 4) << "," << spring(i, 5) << "," << spring(i, 6) << ",";
							spring_cracked_file << spring(i, 7) << "," << spring(i, 8) << "," << spring(i, 9) << "," << spring(i, 1) << "," << differencex << "," << differencey << "," << differencez << "\n";
						}
					}
					spring_cracked_file.close();
				}
				else {
					std::cout << "\a\n\nERROR: Failed to write to file!\n";
				}
			}

			//-----------------------------------------------------------------------------

			for (int i = 19; i < print_list.size(); i++) {

				int choice_of_material = print_list[i];
				i++;
				int choice_of_condition = print_list[i];

				std::ostringstream load_increment_string_special;
				load_increment_string_special << std::setw(5) << std::setfill('0') << load_increment;

				std::ostringstream choice_of_material_string;
				choice_of_material_string << std::setw(5) << std::setfill('0') << choice_of_material;

				std::ostringstream choice_of_condition_string;
				choice_of_condition_string << std::setw(5) << std::setfill('0') << choice_of_condition;

				string material_string;
				string condition_string;

				if (choice_of_material == -1) {
					material_string = "ALL";
				}
				else {
					material_string = choice_of_material_string.str();
				}

				if (choice_of_condition == -1) {
					condition_string = "ALL";
				}
				else {
					condition_string = choice_of_condition_string.str();
				}
				string default_name_special = default_name + "_FILTERED_Material-" + material_string + "_Condition-" + condition_string + "_LoadIncrement-" + load_increment_string_special.str() + "_";
				spring_drawer(default_name_special, spring, total_spring_num, choice_of_material, choice_of_condition);

			}

			if (print_list[16] == 1) {
				string filename_analyze = "ResultsFile_TimeID-" + to_string(time_start_master) + "_ANALYZE.csv";
				ofstream file_analyze;
				file_analyze.open(filename_analyze, ofstream::app);
				if (file_analyze.is_open()) {
					for (int i = 0; i < dofs_to_analyze.size(); i++) {
						//ProgramID, LoadIncrement, element number,x,y,z,dof_num,U,F
						file_analyze << automate << "," << load_increment << "," << dofs_to_analyze(i) / 6 << ",";
						file_analyze << element((int)(dofs_to_analyze(i) / 6.0), 1) << "," << element((int)(dofs_to_analyze(i) / 6.0), 2) << "," << element((int)(dofs_to_analyze(i) / 6.0), 3) << ",";
						file_analyze << dofs_to_analyze(i) << ",";
						file_analyze << U(dofs_to_analyze(i)) << ",";
						file_analyze << F(dofs_to_analyze(i)) << "\n";
					}
					file_analyze.close();
				}
				else {
					std::cout << "\a\n\nERROR: Failed to write to file!\n";
				}
			}

			//-----------------------------------------------------------------------------

			file_runtime.open(filename_runtime, ofstream::app);
			if (file_runtime.is_open()) {
				file_runtime << automate << "," << load_increment << "," << increment_runtime << "\n";
				file_runtime.close();
			}
			else {
				std::cout << "\a\n\nERROR: Failed to write to file!\n";
			}

			//-----------------------------------------------------------------------------

			std::cout << "Files successfully written!\n\n";

			//I have added a manual "deloader" functionality

			if (load_increment < peak_load_increment) {
				load_increment_for_force_applier++;
			}
			else {
				load_increment_for_force_applier--;
			}
		}

		int time_end = time(NULL);
		int runtime = time_end - time_start;
		std::cout << "-------------------------------------------------------------\n\nRUNTIME OF PROGRAM STEP " << automate << ": " << runtime << " seconds.\n\n-------------------------------------------------------------\n\n";

		//-----------------------------------------------------------------------------

		file_runtime.open(filename_runtime, ofstream::app);
		if (file_runtime.is_open()) {
			file_runtime << automate << ",ALL," << runtime << "\n";
			file_runtime.close();
		}
		else {
			std::cout << "\a\n\nERROR: Failed to write to file!\n";
		}

		//-----------------------------------------------------------------------------

		automate += automate_step - 1;
	}

	int time_end_master = time(NULL);
	int runtime_master = time_end_master - time_start_master;
	std::cout << "-------------------------------------------------------------\n\nRUNTIME OF FULL PROGRAM: " << runtime_master << " seconds.\n\n-------------------------------------------------------------";

	file_runtime.open(filename_runtime, ofstream::app);
	if (file_runtime.is_open()) {
		file_runtime << "ALL,ALL," << runtime_master;
		file_runtime.close();
	}
	else {
		std::cout << "\a\n\nERROR: Failed to write to file!\n";
	}

	//-----------------------------------------------------------------------------

	//END

	std::cout << "\n\nPress ENTER to exit the program: ";
	std::cin.get();
	return 0;
}

void print_force(Force& p) {
	std::cout << p.x_coord << " " << p.y_coord << " " << p.z_coord << " " << p.force_magnitude << " " << p.force_dof << endl;
}

void print_selection_point(SelectionPoint& p) {
	std::cout << p.x_coord << " " << p.y_coord << " " << p.z_coord << " " << p.dof << endl;
}

vector<int> reinforcement_tagger(Reinforcement layer_pass, MatrixXd element, int total_ele_num, MatrixXd spring, int total_spring_num, MatrixXd geometry, int choice) {

	//choice = 0 for rebar, 1 for stirrups

	std::cout << "\nFunction reinforcement_tagger running...";

	Reinforcement layer = layer_pass;
	vector<int> affected_springs;

	double x1 = layer.x_coord1;
	double x2 = layer.x_coord2;
	double y1 = layer.y_coord1;
	double y2 = layer.y_coord2;
	double z1 = layer.z_coord1;
	double z2 = layer.z_coord2;

	double bar_size = layer.bar_size;
	int run_direction = layer.run_direction;
	int spacing_direction = layer.spacing_direction;

	if ((spacing_direction < 0) || (spacing_direction > 3)) {
		std::cout << "\a\n\nERROR: Rebars not initialized properly!\n" << endl << endl;
		std::cout << "\n\nPress ENTER to exit the program: ";
		std::cin.get();
		exit(1);
	}

	double spacing = layer.spacing;

	if (spacing < 0) {
		std::cout << "\a\n\nERROR: Rebars not initialized properly!\n" << endl << endl;
		std::cout << "\n\nPress ENTER to exit the program: ";
		std::cin.get();
		exit(1);
	}

	double spacing_direction_total = 0;

	double current_bound_1 = 0; //primary - spacing
	double current_bound_2 = 0;

	double run_bound_1 = 0; //secondary - run
	double run_bound_2 = 0;

	double remain_bound_1 = 0; //secondary - remaining
	double remain_bound_2 = 0;

	//figure out other axis (not spacing, not running)
	vector<int> directions{ 0, 1, 2 };

	directions.erase(remove(directions.begin(), directions.end(), spacing_direction), directions.end()); //erase-remove idiom
	directions.erase(remove(directions.begin(), directions.end(), run_direction), directions.end());

	int remaining_direction = directions[0];

	if (spacing_direction == run_direction) {
		std::cout << "\a\n\nERROR: Rebars not initialized properly!\n" << endl << endl;
		std::cout << "\n\nPress ENTER to exit the program: ";
		std::cin.get();
		exit(1);
	}

	MatrixXd total_bounds = MatrixXd::Zero(3, 2);

	total_bounds(0, 0) = x1;
	total_bounds(0, 1) = x2;

	total_bounds(1, 0) = y1;
	total_bounds(1, 1) = y2;

	total_bounds(2, 0) = z1;
	total_bounds(2, 1) = z2;

	run_bound_1 = total_bounds(run_direction, 0);
	run_bound_2 = total_bounds(run_direction, 1);

	remain_bound_1 = total_bounds(remaining_direction, 0);
	remain_bound_2 = total_bounds(remaining_direction, 1);

	int number_of_bars = 1;
	double spacing_direction_cumulative = bar_size;

	spacing_direction_total = total_bounds(spacing_direction, 1) - total_bounds(spacing_direction, 0);

	if ((spacing_direction_cumulative + spacing + bar_size) <= spacing_direction_total) {
		while (spacing_direction_cumulative + spacing + bar_size <= spacing_direction_total) {
			number_of_bars += 1;
			spacing_direction_cumulative += (spacing + bar_size);
		}
	}

	std::cout << "\nNumber of reinforcement along " << run_direction << " and across " << spacing_direction << ": " << number_of_bars << endl;

	current_bound_1 = total_bounds(spacing_direction, 0);
	current_bound_2 = current_bound_1 + bar_size;

	MatrixXd space_bounds = MatrixXd::Zero(number_of_bars, 2);

	for (int i = 0; i < number_of_bars; i++) {
		space_bounds(i, 0) = current_bound_1;
		space_bounds(i, 1) = current_bound_2;

		current_bound_1 += (bar_size + spacing);
		current_bound_2 += (spacing + bar_size);
	}

	MatrixXd run_bounds = MatrixXd::Zero(2, 2);

	if (choice == 0) {
		run_bounds(0, 0) = run_bound_1;
		run_bounds(0, 1) = run_bound_2;

		run_bounds(1, 0) = run_bound_1;
		run_bounds(1, 1) = run_bound_2;
	}
	else if (choice == 1) {
		run_bounds(0, 0) = run_bound_1;
		run_bounds(0, 1) = run_bound_1 + bar_size;

		run_bounds(1, 0) = run_bound_2 - bar_size;
		run_bounds(1, 1) = run_bound_2;
	}

	MatrixXd remain_bounds = MatrixXd::Zero(2, 2);

	if (choice == 0) {
		remain_bounds(0, 0) = remain_bound_1;
		remain_bounds(0, 1) = remain_bound_2;

		remain_bounds(1, 0) = remain_bound_1;
		remain_bounds(1, 1) = remain_bound_2;
	}
	else if (choice == 1) {
		remain_bounds(0, 0) = remain_bound_1;
		remain_bounds(0, 1) = remain_bound_1 + bar_size;

		remain_bounds(1, 0) = remain_bound_2 - bar_size;
		remain_bounds(1, 1) = remain_bound_2;
	}

	for (int i = 0; i < total_spring_num; i++) {

		int element1 = spring(i, 2);
		int element2 = spring(i, 3);

		double element_space_bound_1 = 0;
		double element_space_bound_2 = 0;

		double element_run_bound_1 = 0;
		double element_run_bound_2 = 0;

		double element_remain_bound_1 = 0;
		double element_remain_bound_2 = 0;

		int geometry_group1 = element(element1, 19);
		int geometry_group2 = element(element2, 19);

		//spring_num indices 9, 10, 11

		int spring_space_element1 = geometry(geometry_group1, spacing_direction + 9);
		int spring_space_element2 = geometry(geometry_group2, spacing_direction + 9);

		int spring_run_element1 = geometry(geometry_group1, run_direction + 9);
		int spring_run_element2 = geometry(geometry_group2, run_direction + 9);

		int spring_remain_element1 = geometry(geometry_group1, remaining_direction + 9);
		int spring_remain_element2 = geometry(geometry_group2, remaining_direction + 9);

		//indices 
		//element coordinate 1 2 3
		//element dimension 7 8 9
		//spring coordinate 4 5 6

		int index_space_coordinate = spacing_direction + 1;
		int index_run_coordinate = run_direction + 1;
		int index_remain_coordinate = remaining_direction + 1;

		int index_space_element_dimension = spacing_direction + 7;
		int index_run_element_dimension = run_direction + 7;
		int index_remain_element_dimension = remaining_direction + 7;

		int index_space = spacing_direction + 4;
		int index_run = run_direction + 4;
		int index_remain = remaining_direction + 4;

		//get the boundaries of the spring tributary volume (the tributary area along the element face, and the element centroid as the tributary length)
		if (spring(i, 1) == spacing_direction) {
			element_space_bound_1 = element(element1, index_space_coordinate);
			element_space_bound_2 = element(element2, index_space_coordinate);

			element_run_bound_1 = min(spring(i, index_run) - element(element1, index_run_element_dimension) / spring_run_element1 / 2.0, spring(i, index_run + 3) - element(element2, index_run_element_dimension) / spring_run_element2 / 2.0);
			element_run_bound_2 = max(spring(i, index_run) + element(element1, index_run_element_dimension) / spring_run_element1 / 2.0, spring(i, index_run + 3) + element(element2, index_run_element_dimension) / spring_run_element2 / 2.0);

			element_remain_bound_1 = min(spring(i, index_remain) - element(element1, index_remain_element_dimension) / spring_remain_element1 / 2.0, spring(i, index_remain + 3) - element(element2, index_remain_element_dimension) / spring_remain_element2 / 2.0);
			element_remain_bound_2 = max(spring(i, index_remain) + element(element1, index_remain_element_dimension) / spring_remain_element1 / 2.0, spring(i, index_remain + 3) + element(element2, index_remain_element_dimension) / spring_remain_element2 / 2.0);
		}
		else if (spring(i, 1) == run_direction) {

			element_space_bound_1 = min(spring(i, index_space) - element(element1, index_space_element_dimension) / spring_space_element1 / 2.0, spring(i, index_space + 3) - element(element2, index_space_element_dimension) / spring_space_element2 / 2.0);
			element_space_bound_2 = max(spring(i, index_space) + element(element1, index_space_element_dimension) / spring_space_element1 / 2.0, spring(i, index_space + 3) + element(element2, index_space_element_dimension) / spring_space_element2 / 2.0);

			element_run_bound_1 = element(element1, index_run_coordinate);
			element_run_bound_2 = element(element2, index_run_coordinate);

			element_remain_bound_1 = min(spring(i, index_remain) - element(element1, index_remain_element_dimension) / spring_remain_element1 / 2.0, spring(i, index_remain + 3) - element(element2, index_remain_element_dimension) / spring_remain_element2 / 2.0);
			element_remain_bound_2 = max(spring(i, index_remain) + element(element1, index_remain_element_dimension) / spring_remain_element1 / 2.0, spring(i, index_remain + 3) + element(element2, index_remain_element_dimension) / spring_remain_element2 / 2.0);

		}
		else if (spring(i, 1) == remaining_direction) {
			element_space_bound_1 = min(spring(i, index_space) - element(element1, index_space_element_dimension) / spring_space_element1 / 2.0, spring(i, index_space + 3) - element(element2, index_space_element_dimension) / spring_space_element2 / 2.0);
			element_space_bound_2 = max(spring(i, index_space) + element(element1, index_space_element_dimension) / spring_space_element1 / 2.0, spring(i, index_space + 3) + element(element2, index_space_element_dimension) / spring_space_element2 / 2.0);

			element_run_bound_1 = min(spring(i, index_run) - element(element1, index_run_element_dimension) / spring_run_element1 / 2.0, spring(i, index_run + 3) - element(element2, index_run_element_dimension) / spring_run_element2 / 2.0);
			element_run_bound_2 = max(spring(i, index_run) + element(element1, index_run_element_dimension) / spring_run_element1 / 2.0, spring(i, index_run + 3) + element(element2, index_run_element_dimension) / spring_run_element2 / 2.0);

			element_remain_bound_1 = element(element1, index_remain_coordinate);
			element_remain_bound_2 = element(element2, index_remain_coordinate);
		}

		//spring 14 15 16
		//element 13 14 15

		bool bar_inside_1 = false;
		bool bar_inside_2 = false;
		bool bar_inside_3 = false;

		for (int j = 0; j < 2; j++) {
			double temp_bound_1 = run_bounds(j, 0);
			double temp_bound_2 = run_bounds(j, 1);

			bool temp3 = (temp_bound_1 < element_run_bound_1) && (temp_bound_2 < element_run_bound_1); // bar is wholly to the left of the area under investigation i.e.    O   []
			bool temp4 = (temp_bound_1 > element_run_bound_2) && (temp_bound_2 > element_run_bound_2); // bar is wholly to the right of the area under investigation i.e.    []   O

			if ((!temp3) && (!temp4)) {
				bar_inside_2 = true;
				break;
			}
		}

		for (int j = 0; j < 2; j++) {
			double temp_bound_1 = remain_bounds(j, 0);
			double temp_bound_2 = remain_bounds(j, 1);

			bool temp5 = (temp_bound_1 < element_remain_bound_1) && (temp_bound_2 < element_remain_bound_1); // bar is wholly to the left of the area under investigation i.e.    O   []
			bool temp6 = (temp_bound_1 > element_remain_bound_2) && (temp_bound_2 > element_remain_bound_2); // bar is wholly to the right of the area under investigation i.e.    []   O

			if ((!temp5) && (!temp6)) {
				bar_inside_3 = true;
				break;
			}
		}

		for (int j = 0; j < number_of_bars; j++) {
			double temp_bound_1 = space_bounds(j, 0);
			double temp_bound_2 = space_bounds(j, 1);

			bool temp1 = (temp_bound_1 < element_space_bound_1) && (temp_bound_2 < element_space_bound_1); // bar is wholly to the left of the area under investigation i.e.    O   []
			bool temp2 = (temp_bound_1 > element_space_bound_2) && (temp_bound_2 > element_space_bound_2); // bar is wholly to the right of the area under investigation i.e.    []   O

			if ((!temp1) && (!temp2)) { //if the bar is NEITHER to the left fully or the right fully of the area -> it intersects it.
				bar_inside_1 = true;
				break;
			}
		}

		if (choice == 0) {
			if (bar_inside_1 && bar_inside_2 && bar_inside_3) {
				affected_springs.push_back(i);
			}
		}
		else if (choice == 1) {
			bool stirrup_check = bar_inside_2 || bar_inside_3;

			if (bar_inside_1 && stirrup_check) {
				bool temp_stir_1 = (run_bound_1 < element_run_bound_1) && (run_bound_2 < element_run_bound_1); // bar is wholly to the left of the area under investigation i.e.    O   []
				bool temp_stir_2 = (run_bound_1 > element_run_bound_2) && (run_bound_2 > element_run_bound_2); // bar is wholly to the right of the area under investigation i.e.    []   O
				bool temp_stir_3 = (remain_bound_1 < element_remain_bound_1) && (remain_bound_2 < element_remain_bound_1); // bar is wholly to the left of the area under investigation i.e.    O   []
				bool temp_stir_4 = (remain_bound_1 > element_remain_bound_2) && (remain_bound_2 > element_remain_bound_2); // bar is wholly to the right of the area under investigation i.e.    []   O

				bool basic_stir_check = !temp_stir_1 && !temp_stir_2 && !temp_stir_3 && !temp_stir_4;

				if (basic_stir_check) {
					affected_springs.push_back(i);
				}
			}
		}
	}
	std::cout << "Function reinforcement_tagger completed!\n";
	return affected_springs;
}

vector<int> material_tagger(Selection bounds, MatrixXd element, int total_ele_num, MatrixXd spring, int total_spring_num, MatrixXd geometry) {

	double x1 = bounds.x_coord1;
	double x2 = bounds.x_coord2;
	double y1 = bounds.y_coord1;
	double y2 = bounds.y_coord2;
	double z1 = bounds.z_coord1;
	double z2 = bounds.z_coord2;

	vector<int> affected_springs;

	for (int i = 0; i < total_spring_num; i++) {

		int element1 = spring(i, 2);
		int element2 = spring(i, 3);

		int geometry_group1 = element(element1, 19);
		int geometry_group2 = element(element2, 19);

		double spring_x1 = 0, spring_x2 = 0, spring_y1 = 0, spring_y2 = 0, spring_z1 = 0, spring_z2 = 0;

		if (spring(i, 1) == 0) {
			spring_x1 = element(element1, 1);
			spring_x2 = element(element2, 1);

			spring_y1 = min(spring(i, 5) - element(element1, 8) / geometry(geometry_group1, 7) / 2.0, spring(i, 5 + 3) - element(element2, 8) / geometry(geometry_group2, 7) / 2.0);
			spring_y2 = max(spring(i, 5) + element(element1, 8) / geometry(geometry_group1, 7) / 2.0, spring(i, 5 + 3) + element(element2, 8) / geometry(geometry_group2, 7) / 2.0);

			spring_z1 = min(spring(i, 6) - element(element1, 9) / geometry(geometry_group1, 8) / 2.0, spring(i, 6 + 3) - element(element2, 9) / geometry(geometry_group2, 8) / 2.0);
			spring_z2 = max(spring(i, 6) + element(element1, 9) / geometry(geometry_group1, 8) / 2.0, spring(i, 6 + 3) + element(element2, 9) / geometry(geometry_group2, 8) / 2.0);
		}
		else if (spring(i, 1) == 1) {

			spring_x1 = min(spring(i, 4) - element(element1, 7) / geometry(geometry_group1, 6) / 2.0, spring(i, 4 + 3) - element(element2, 7) / geometry(geometry_group2, 6) / 2.0);
			spring_x2 = max(spring(i, 4) + element(element1, 7) / geometry(geometry_group1, 6) / 2.0, spring(i, 4 + 3) + element(element2, 7) / geometry(geometry_group2, 6) / 2.0);

			spring_y1 = element(element1, 2);
			spring_y2 = element(element2, 2);

			spring_z1 = min(spring(i, 6) - element(element1, 9) / geometry(geometry_group1, 8) / 2.0, spring(i, 6 + 3) - element(element2, 9) / geometry(geometry_group2, 8) / 2.0);
			spring_z2 = max(spring(i, 6) + element(element1, 9) / geometry(geometry_group1, 8) / 2.0, spring(i, 6 + 3) + element(element2, 9) / geometry(geometry_group2, 8) / 2.0);

		}
		else if (spring(i, 1) == 2) {
			spring_x1 = min(spring(i, 4) - element(element1, 7) / geometry(geometry_group1, 6) / 2.0, spring(i, 5 + 3) - element(element2, 7) / geometry(geometry_group2, 6) / 2.0);
			spring_x2 = max(spring(i, 4) + element(element1, 7) / geometry(geometry_group1, 6) / 2.0, spring(i, 5 + 3) + element(element2, 7) / geometry(geometry_group2, 6) / 2.0);

			spring_y1 = min(spring(i, 5) - element(element1, 8) / geometry(geometry_group1, 7) / 2.0, spring(i, 6 + 3) - element(element2, 8) / geometry(geometry_group2, 7) / 2.0);
			spring_y2 = max(spring(i, 5) + element(element1, 8) / geometry(geometry_group1, 7) / 2.0, spring(i, 6 + 3) + element(element2, 8) / geometry(geometry_group2, 7) / 2.0);

			spring_z1 = element(element1, 3);
			spring_z2 = element(element2, 3);
		}

		bool spring_inside_1 = false;
		bool spring_inside_2 = false;
		bool spring_inside_3 = false;

		bool temp1 = (x1 < spring_x1) && (x2 < spring_x1); // bar is wholly to the left of the area under investigation i.e.    O   []
		bool temp2 = (x1 > spring_x2) && (x2 > spring_x2); // bar is wholly to the right of the area under investigation i.e.    []   O

		if ((!temp1) && (!temp2)) {
			spring_inside_1 = true;
		}
		else {
			continue;
		}

		bool temp3 = (y1 < spring_y1) && (y2 < spring_y1); // bar is wholly to the left of the area under investigation i.e.    O   []
		bool temp4 = (y1 > spring_y2) && (y2 > spring_y2); // bar is wholly to the right of the area under investigation i.e.    []   O

		if ((!temp3) && (!temp4)) {
			spring_inside_2 = true;
		}
		else {
			continue;
		}

		bool temp5 = (z1 < spring_z1) && (z2 < spring_z1); // bar is wholly to the left of the area under investigation i.e.    O   []
		bool temp6 = (z1 > spring_z2) && (z2 > spring_z2); // bar is wholly to the right of the area under investigation i.e.    []   O

		if ((!temp5) && (!temp6)) {
			spring_inside_3 = true;
		}
		else {
			continue;
		}
		if (spring_inside_1 && spring_inside_2 && spring_inside_3) {
			affected_springs.push_back(i);
		}
	}
	return affected_springs;
}


int element_finder(MatrixXd element, int total_ele_num, double x, double y, double z) {

	double distance_lowest;
	double distance_current = 0;
	int nearest_element = 0; //default: apply load at first element

	for (int i = 0; i < total_ele_num; i++) {
		double elementx = element(i, 1);
		double elementy = element(i, 2);
		double elementz = element(i, 3);

		double difference_x = x - elementx;
		double difference_y = y - elementy;
		double difference_z = z - elementz;

		distance_current = sqrt(difference_x * difference_x + difference_y * difference_y + difference_z * difference_z);

		if (i == 0) { //first element found
			distance_lowest = distance_current;
		}

		if (distance_current <= distance_lowest) {
			nearest_element = i;
			distance_lowest = distance_current;
		}
	}
	return nearest_element;
}

vector<SelectionPoint> selection_processor(Selection selection, MatrixXd element, int total_ele_num) {
	double x1 = selection.x_coord1;
	double x2 = selection.x_coord2;
	double y1 = selection.y_coord1;
	double y2 = selection.y_coord2;
	double z1 = selection.z_coord1;
	double z2 = selection.z_coord2;

	int dof = selection.dof;

	int element_corner1 = element_finder(element, total_ele_num, x1, y1, z1);
	int element_corner2 = element_finder(element, total_ele_num, x2, y2, z2);

	vector <SelectionPoint> equivalent_points;

	double bound_x1 = min(element(element_corner1, 1), element(element_corner2, 1));
	double bound_y1 = min(element(element_corner1, 2), element(element_corner2, 2));
	double bound_z1 = min(element(element_corner1, 3), element(element_corner2, 3));

	double bound_x2 = max(element(element_corner1, 1), element(element_corner2, 1));
	double bound_y2 = max(element(element_corner1, 2), element(element_corner2, 2));
	double bound_z2 = max(element(element_corner1, 3), element(element_corner2, 3));

	for (int i = 0; i < total_ele_num; i++) {
		double elementx = element(i, 1);
		double elementy = element(i, 2);
		double elementz = element(i, 3);

		bool x_bool = (elementx >= bound_x1) && (elementx <= bound_x2);
		bool y_bool = (elementy >= bound_y1) && (elementy <= bound_y2);
		bool z_bool = (elementz >= bound_z1) && (elementz <= bound_z2);

		if (x_bool && y_bool && z_bool) {

			SelectionPoint point_equiv;

			point_equiv.x_coord = elementx;
			point_equiv.y_coord = elementy;
			point_equiv.z_coord = elementz;
			point_equiv.dof = dof;

			equivalent_points.push_back(point_equiv);
		}
	}
	return equivalent_points;
}

VectorXi dof_finder(vector<SelectionPoint> selection_description, int selection_count, MatrixXd element, int total_ele_num) {

	VectorXi dofs_to_select = VectorXi::Zero(selection_count);

	for (int i = 0; i < selection_count; i++) {

		SelectionPoint current_support = selection_description[i];

		double x = current_support.x_coord;
		double y = current_support.y_coord;
		double z = current_support.z_coord;

		int element_to_select = element_finder(element, total_ele_num, x, y, z);
		int dof_now = element_to_select * 6 + current_support.dof;
		dofs_to_select(i) = dof_now;
	}
	return dofs_to_select;
}

void support_restrainer(Eigen::Ref<MatrixXd> element, int total_ele_num, vector<SelectionPoint> selection_description, int selection_count) {
	for (int i = 0; i < selection_count; i++) {
		SelectionPoint current_support = selection_description[i];

		double x = current_support.x_coord;
		double y = current_support.y_coord;
		double z = current_support.z_coord;

		int element_to_select = element_finder(element, total_ele_num, x, y, z);

		element(element_to_select, 21) = 1;
	}
}

vector<Force> distributed_load_processor(DistributedLoad distributed_load, MatrixXd geometry, MatrixXd element, int total_ele_num) {
	double x1 = distributed_load.x_coord1;
	double x2 = distributed_load.x_coord2;
	double y1 = distributed_load.y_coord1;
	double y2 = distributed_load.y_coord2;
	double z1 = distributed_load.z_coord1;
	double z2 = distributed_load.z_coord2;
	double current_force = distributed_load.force_magnitude;
	int dof = distributed_load.force_dof;

	double distance_lowest1;
	double distance_lowest2;
	double distance_current1 = 0;
	double distance_current2 = 0;

	int element_corner1 = element_finder(element, total_ele_num, x1, y1, z1);
	int element_corner2 = element_finder(element, total_ele_num, x2, y2, z2);

	vector<Force> equivalent_point_loads;

	double bound_x1 = element(element_corner1, 1);
	double bound_y1 = element(element_corner1, 2);
	double bound_z1 = element(element_corner1, 3);

	double bound_x2 = element(element_corner2, 1);
	double bound_y2 = element(element_corner2, 2);
	double bound_z2 = element(element_corner2, 3);

	for (int i = 0; i < total_ele_num; i++) {

		double elementx = element(i, 1);
		double elementy = element(i, 2);
		double elementz = element(i, 3);

		double element_a = element(i, 7);
		double element_b = element(i, 8);
		double element_c = element(i, 9);

		bool x_bool = (elementx >= bound_x1) && (elementx <= bound_x2);
		bool y_bool = (elementy >= bound_y1) && (elementy <= bound_y2);
		bool z_bool = (elementz >= bound_z1) && (elementz <= bound_z2);

		if (x_bool && y_bool && z_bool) {

			double element_far_edge_x1 = elementx - element_a / 2.0;
			double element_far_edge_y1 = elementy - element_b / 2.0;
			double element_far_edge_z1 = elementz - element_c / 2.0;

			double element_far_edge_x2 = elementx + element_a / 2.0;
			double element_far_edge_y2 = elementy + element_b / 2.0;
			double element_far_edge_z2 = elementz + element_c / 2.0;

			//check for any excess or lacking only at boundary elements

			//technically: if the resultant force on the element is off-center (there is excess or lacking)
			//there should also be an applied moment on the element
			//but I didn't implement this here since it's too complicated and I won't be using that consideration too much in the analysis

			int geometry_group = element(i, 19);

			double xdim = geometry(geometry_group, 1) - geometry(geometry_group, 0);
			double ydim = geometry(geometry_group, 3) - geometry(geometry_group, 2);
			double zdim = geometry(geometry_group, 5) - geometry(geometry_group, 4);

			double origin_x = geometry(geometry_group, 0);
			double origin_y = geometry(geometry_group, 2);
			double origin_z = geometry(geometry_group, 4);

			bool edge_x_min = (elementx - element(i, 7) / 2.0) <= origin_x;
			bool edge_x_max = (elementx + element(i, 7) / 2.0) >= (origin_x + xdim);

			bool edge_y_min = (elementy - element(i, 8) / 2.0) <= origin_y;
			bool edge_y_max = (elementy + element(i, 8) / 2.0) >= (origin_y + ydim);

			bool edge_z_min = (elementz - element(i, 9) / 2.0) <= origin_z;
			bool edge_z_max = (elementz + element(i, 9) / 2.0) >= (origin_z + zdim);

			if (elementx == bound_x1) {
				double check = element_far_edge_x1 - x1;
				//if positive: lacking, the force is being applied beyond the element but there are no other elements to "catch" the force (have to add more tributary area to compensate)
				//if negative: excess, the force distribution is on the element but does not reach the element edge (have to reduce the force applied because only part of the element is covered by the load)

				if ((check > 0) && (edge_x_min)) { //catches 1 - min edge and 3 - both
					//skip if distload bounds overshoot the element but there are no x-adjacent elements
				}
				else {
					element_a += check;
				}
			}

			if (elementx == bound_x2) {
				double check = x2 - element_far_edge_x2;
				if ((check > 0) && (edge_x_max)) { //catches 2 - max edge and 3 - both
					//skip
				}
				else {
					element_a += check;
				}
			}

			if (elementy == bound_y1) {
				double check = element_far_edge_y1 - y1;
				if ((check > 0) && (edge_y_min)) {
					//skip
				}
				else {
					element_b += check;
				}
			}
			if (elementy == bound_y2) {
				double check = y2 - element_far_edge_y2;
				if ((check > 0) && (edge_y_max)) {
					//skip
				}
				else {
					element_b += check;
				}
			}
			if (elementz == bound_z1) {
				double check = element_far_edge_z1 - z1;
				if ((check > 0) && (edge_z_min)) {
					//skip
				}
				else {
					element_c += check;
				}
			}
			if (elementz == bound_z2) {
				double check = z2 - element_far_edge_z2;
				if ((check > 0) && (edge_z_max)) {
					//skip
				}
				else {
					element_c += check;
				}
			}

			if ((int)fabs(element_a) == 0) { element_a = 1; } //convert to unit dimensions for cases where there is no x, y, or z width
			if ((int)fabs(element_b) == 0) { element_b = 1; }
			if ((int)fabs(element_c) == 0) { element_c = 1; }

			double loading;
			Force point_equiv;

			if (distributed_load.triangle == 1 || distributed_load.triangle == 2) {
				double runx = fabs(x2 - x1);
				double runy = fabs(y2 - y1);
				double runz = fabs(z2 - z1);

				//DEVELOPER NOTE: shortcut, assume that only line loads varying along 1 axis will be used

				double run_max = max({ runx, runy, runz });

				double slope = current_force / run_max;

				double end1 = max(element_far_edge_x1, x1);
				double end2 = min(element_far_edge_x2, x2);

				//y = mx + b
				double b = 0 - slope * x1;

				double force1 = 0;
				double force2 = 0;

				if (distributed_load.triangle == 1) { // 0 -> w
					force1 = slope * end1 + b;
					force2 = slope * end2 + b;
				}
				else if (distributed_load.triangle == 2) { // w -> 0
					force1 = slope * (-end1) - b + run_max * slope; // double-check if this is valid for all cases
					force2 = slope * (-end2) - b + run_max * slope;
				}
				loading = (force1 + force2) / 2;
			}
			else {
				loading = current_force;
			}

			point_equiv.x_coord = elementx;
			point_equiv.y_coord = elementy;
			point_equiv.z_coord = elementz;
			point_equiv.force_magnitude = loading * element_a * element_b * element_c;
			point_equiv.force_dof = dof;

			equivalent_point_loads.push_back(point_equiv);
		}
	}
	return equivalent_point_loads;
}

MatrixXd force_dof_finder(vector<Force> force_description, int load_count, MatrixXd element, int total_ele_num) {

	MatrixXd dofs_to_apply = MatrixXd::Zero(load_count, 2);

	for (int i = 0; i < load_count; i++) {

		Force current_force = force_description[i];

		double x = current_force.x_coord;
		double y = current_force.y_coord;
		double z = current_force.z_coord;

		double distance_lowest;
		double distance_current = 0;
		int element_to_apply = element_finder(element, total_ele_num, x, y, z);

		int dof_to_apply = element_to_apply * 6 + current_force.force_dof;

		dofs_to_apply(i, 0) = dof_to_apply;
		dofs_to_apply(i, 1) = current_force.force_magnitude;

	}
	return dofs_to_apply;
}


void force_applier(Eigen::Ref<VectorXd> F, MatrixXd dofs_to_apply, int dofs_to_apply_count, int load_increment_count, int load_increment) {
	for (int i = 0; i < dofs_to_apply_count; i++) {
		int current_dof = dofs_to_apply(i, 0); //0 - dof, 1 - force
		double current_force = dofs_to_apply(i, 1);

		F(current_dof) += (current_force / load_increment_count) * (load_increment);
	}
}

VectorXd rotation_to_displacement(double Ax, double Ay, double Az, double thetax, double thetay, double thetaz) {
	VectorXd delta_from_rotation = VectorXd::Zero(3);
	//Ax - x_from_centroid
	//Ay - y_from_centroid
	//Az - z_from_centroid
	//B - new location global

	//Mz
	double Bx_tz = Ax * cos(thetaz) - Ay * sin(thetaz);
	double By_tz = Ax * sin(thetaz) + Ay * cos(thetaz);

	double delta_x_tz = Bx_tz - Ax;
	double delta_y_tz = By_tz - Ay;
	double delta_z_tz = 0;

	//My
	double Bx_ty = Ax * cos(-thetay) - Az * sin(-thetay);
	double Bz_ty = Ax * sin(-thetay) + Az * cos(-thetay);

	double delta_x_ty = Bx_ty - Ax;
	double delta_y_ty = 0;
	double delta_z_ty = Bz_ty - Az;

	//Mx
	double By_tx = Ay * cos(thetax) - Az * sin(thetax);
	double Bz_tx = Ay * sin(thetax) + Az * cos(thetax);

	double delta_x_tx = 0;
	double delta_y_tx = By_tx - Ay;
	double delta_z_tx = Bz_tx - Az;

	delta_from_rotation(0) = delta_x_tx + delta_x_ty + delta_x_tz;
	delta_from_rotation(1) = delta_y_tx + delta_y_ty + delta_y_tz;
	delta_from_rotation(2) = delta_z_tx + delta_z_ty + delta_z_tz;

	return delta_from_rotation;
}

MatrixXd transformation_matrix(double end1x, double end1y, double end1z, double end2x, double end2y, double end2z, int choice) {

	//check if there is already a separation
	//this function does not work for the initial condition: 0 0 0 gap between spring ends
	double deltax = end2x - end1x;
	double deltay = end2y - end1y;
	double deltaz = end2z - end1z;

	if (deltax == 0 && deltay == 0 && deltaz == 0) {
		cout << endl << "\a\n\nERROR! transformation_matrix: Point1 == Point2\n\n" << endl;
	}

	//-----------------------------------------------------------------------------

	VectorXd xprime = VectorXd::Zero(3);
	VectorXd yprime = VectorXd::Zero(3);
	VectorXd zprime = VectorXd::Zero(3);

	xprime(0) = deltax;
	xprime(1) = deltay;
	xprime(2) = deltaz;

	if (deltax == 0 && deltay == 0) {
		yprime(0) = 0;
		yprime(1) = -deltaz;
		yprime(2) = deltay;
	}
	else {
		yprime(0) = -deltay;
		yprime(1) = deltax;
		yprime(2) = 0;
	}

	//-----------------------------------------------------------------------------

	//the zprimes follow the right hand rule

	zprime(0) = xprime(1) * yprime(2) - xprime(2) * yprime(1);
	zprime(1) = -(xprime(0) * yprime(2) - xprime(2) * yprime(0));
	zprime(2) = xprime(0) * yprime(1) - xprime(1) * yprime(0);

	double lx = sqrt((xprime(0) * xprime(0)) + (xprime(1) * xprime(1)) + (xprime(2) * xprime(2)));
	double ly = sqrt((yprime(0) * yprime(0)) + (yprime(1) * yprime(1)) + (yprime(2) * yprime(2)));
	double lz = sqrt((zprime(0) * zprime(0)) + (zprime(1) * zprime(1)) + (zprime(2) * zprime(2)));

	//-----------------------------------------------------------------------------

	double cosax, cosbx, coscx, cosay, cosby, coscy, cosaz, cosbz, coscz;

	cosax = xprime(0) / lx;
	cosbx = xprime(1) / lx;
	coscx = xprime(2) / lx;

	cosay = yprime(0) / ly;
	cosby = yprime(1) / ly;
	coscy = yprime(2) / ly;

	cosaz = zprime(0) / lz;
	cosbz = zprime(1) / lz;
	coscz = zprime(2) / lz;

	if (isnan(cosax) || isnan(cosbx) || isnan(coscx) || isnan(cosay) || isnan(cosby) || isnan(coscy) || isnan(cosaz) || isnan(cosbz) || isnan(coscz)) {
		std::cout << "\a\n\nERROR! At transformation_matrix: one of the cosines evaluated as NaN (the two points are close enough that a divide by zero occured).\n\n";
		std::cout << "Input at error call: " << end1x << " " << end1y << " " << end1z << " " << end2x << " " << end2y << " " << end2z << endl;
		std::cout << "\n\nPress ENTER to exit the program: ";
		exit(1);
	}

	//-----------------------------------------------------------------------------

	MatrixXd basic = MatrixXd::Zero(3, 3);

	basic << cosax, cosbx, coscx,
		cosay, cosby, coscy,
		cosaz, cosbz, coscz;

	MatrixXd full = MatrixXd::Zero(12, 12);

	//making the 12x12 matrix of cosines
	int counter = 0;
	for (int i = 0; i < 4; i++) {

		counter = i * 3;

		for (int j = 0; j < 3; j++) {
			for (int k = 0; k < 3; k++) {
				int fullx = counter + j;
				int fully = counter + k;

				full(fullx, fully) = basic(j, k);
			}
		}
	}

	//-----------------------------------------------------------------------------

	if (choice == 0) {
		return basic;
	}
	if (choice == 1) {
		return full;
	}

	return basic;
}

MatrixXd localmatrix(double Qx, double Qy, double Qz, double x1, double y1, double z1, double x2, double y2, double z2) {
	MatrixXd Klocal(12, 12);
	Klocal << Qx, 0, 0, 0, Qx* z1, -Qx * y1, -Qx, 0, 0, 0, -Qx * z2, Qx* y2,
		0, Qy, 0, -Qy * z1, 0, Qy* x1, 0, -Qy, 0, Qy* z2, 0, Qy* x2,
		0, 0, Qz, Qz* y1, -Qz * x1, 0, 0, 0, -Qz, -Qz * y2, -Qz * x2, 0,
		0, -Qy * z1, Qz* y1, Qy* (z1 * z1) + Qz * (y1 * y1), -Qz * x1 * y1, -Qy * x1 * z1, 0, Qy* z1, -Qz * y1, -Qy * z1 * z2 - Qz * y1 * y2, -Qz * x2 * y1, -Qy * x2 * z1,
		Qx* z1, 0, -Qz * x1, -Qz * x1 * y1, Qx* (z1 * z1) + Qz * (x1 * x1), -Qx * y1 * z1, -Qx * z1, 0, Qz* x1, Qz* x1* y2, -Qx * z1 * z2 + Qz * x1 * x2, Qx* y2* z1,
		-Qx * y1, Qy* x1, 0, -Qy * x1 * z1, -Qx * y1 * z1, Qx* (y1 * y1) + Qy * (x1 * x1), Qx* y1, -Qy * x1, 0, Qy* x1* z2, Qx* y1* z2, -Qx * y1 * y2 + Qy * x1 * x2,
		-Qx, 0, 0, 0, -Qx * z1, Qx* y1, Qx, 0, 0, 0, Qx* z2, -Qx * y2,
		0, -Qy, 0, Qy* z1, 0, -Qy * x1, 0, Qy, 0, -Qy * z2, 0, -Qy * x2,
		0, 0, -Qz, -Qz * y1, Qz* x1, 0, 0, 0, Qz, Qz* y2, Qz* x2, 0,
		0, Qy* z2, -Qz * y2, -Qy * z1 * z2 - Qz * y1 * y2, Qz* x1* y2, Qy* x1* z2, 0, -Qy * z2, Qz* y2, Qy* (z2 * z2) + Qz * (y2 * y2), Qz* x2* y2, Qy* x2* z2,
		-Qx * z2, 0, -Qz * x2, -Qz * x2 * y1, -Qx * z1 * z2 + Qz * x1 * x2, Qx* y1* z2, Qx* z2, 0, Qz* x2, Qz* x2* y2, Qx* (z2 * z2) + Qz * (x2 * x2), -Qx * y2 * z2,
		Qx* y2, Qy* x2, 0, -Qy * x2 * z1, Qx* y2* z1, -Qx * y1 * y2 + Qy * x1 * x2, -Qx * y2, -Qy * x2, 0, Qy* x2* z2, -Qx * y2 * z2, Qx* (y2 * y2) + Qy * (x2 * x2);
	return Klocal;
}


void element_drawer(string name, MatrixXd element, int total_ele_num, VectorXd U, VectorXd F, MatrixXd element_principal_stresses) {

	string filename_element_vtk = name + "PARAVIEW_" + "element.vtk";
	ofstream element_vtk_file;
	element_vtk_file.open(filename_element_vtk, ofstream::app);
	if (element_vtk_file.is_open()) {

		std::cout << "\nDrawing the mesh...\n";
		int draw_time_start = time(NULL);

		MatrixXd vtk_vertices = MatrixXd::Zero(total_ele_num * 8, 3);
		Eigen::MatrixXi vtk_polygons = Eigen::MatrixXi::Zero(total_ele_num * 6, 5);

		for (int polygon_iterant = 0; polygon_iterant < (total_ele_num * 6); polygon_iterant++) {
			vtk_polygons(polygon_iterant, 0) = 4;
		}

		element_vtk_file << "# vtk DataFile Version 2.0\n";
		element_vtk_file << "Elements\n";
		element_vtk_file << "ASCII\n";
		element_vtk_file << "DATASET POLYDATA\n";
		element_vtk_file << "POINTS " << total_ele_num * 8 << " double\n";
		for (int i = 0; i < total_ele_num; i++) {

			//initialize changing variables
			double x_temp = 0;
			double y_temp = 0;
			double z_temp = 0;

			VectorXd distance_temp = VectorXd::Zero(3);

			//get element coords
			double x_centroid = element(i, 1);
			double y_centroid = element(i, 2);
			double z_centroid = element(i, 3);

			//get rotations
			double thetax = element(i, 4);
			double thetay = element(i, 5);
			double thetaz = element(i, 6);

			double half_a = element(i, 7) / 2;
			double half_b = element(i, 8) / 2;
			double half_c = element(i, 9) / 2;

			VectorXd distance_temp_add = VectorXd::Zero(3);

			int point1 = i * 8;
			int point2 = point1 + 1;
			int point3 = point2 + 1;
			int point4 = point3 + 1;
			int point5 = point4 + 1;
			int point6 = point5 + 1;
			int point7 = point6 + 1;
			int point8 = point7 + 1;

			//point1 minx miny minz
			distance_temp(0) = -half_a;
			distance_temp(1) = -half_b;
			distance_temp(2) = -half_c;

			distance_temp_add = rotation_to_displacement(distance_temp(0), distance_temp(1), distance_temp(2), thetax, thetay, thetaz);

			vtk_vertices(point1, 0) = x_centroid + distance_temp(0) + distance_temp_add(0);
			vtk_vertices(point1, 1) = y_centroid + distance_temp(1) + distance_temp_add(1);
			vtk_vertices(point1, 2) = z_centroid + distance_temp(2) + distance_temp_add(2);

			//-----------------------------------------------------------------------------
			//point2 maxx miny minz
			distance_temp(0) = half_a;
			distance_temp(1) = -half_b;
			distance_temp(2) = -half_c;

			distance_temp_add = rotation_to_displacement(distance_temp(0), distance_temp(1), distance_temp(2), thetax, thetay, thetaz);

			vtk_vertices(point2, 0) = x_centroid + distance_temp(0) + distance_temp_add(0);
			vtk_vertices(point2, 1) = y_centroid + distance_temp(1) + distance_temp_add(1);
			vtk_vertices(point2, 2) = z_centroid + distance_temp(2) + distance_temp_add(2);

			//-----------------------------------------------------------------------------
			//point3 maxx miny maxz
			distance_temp(0) = half_a;
			distance_temp(1) = -half_b;
			distance_temp(2) = half_c;

			distance_temp_add = rotation_to_displacement(distance_temp(0), distance_temp(1), distance_temp(2), thetax, thetay, thetaz);

			vtk_vertices(point3, 0) = x_centroid + distance_temp(0) + distance_temp_add(0);
			vtk_vertices(point3, 1) = y_centroid + distance_temp(1) + distance_temp_add(1);
			vtk_vertices(point3, 2) = z_centroid + distance_temp(2) + distance_temp_add(2);

			//-----------------------------------------------------------------------------
			//point4 minx miny maxz
			distance_temp(0) = -half_a;
			distance_temp(1) = -half_b;
			distance_temp(2) = half_c;

			distance_temp_add = rotation_to_displacement(distance_temp(0), distance_temp(1), distance_temp(2), thetax, thetay, thetaz);

			vtk_vertices(point4, 0) = x_centroid + distance_temp(0) + distance_temp_add(0);
			vtk_vertices(point4, 1) = y_centroid + distance_temp(1) + distance_temp_add(1);
			vtk_vertices(point4, 2) = z_centroid + distance_temp(2) + distance_temp_add(2);

			//-----------------------------------------------------------------------------
			//point5 minx maxy minz
			distance_temp(0) = -half_a;
			distance_temp(1) = half_b;
			distance_temp(2) = -half_c;

			distance_temp_add = rotation_to_displacement(distance_temp(0), distance_temp(1), distance_temp(2), thetax, thetay, thetaz);

			vtk_vertices(point5, 0) = x_centroid + distance_temp(0) + distance_temp_add(0);
			vtk_vertices(point5, 1) = y_centroid + distance_temp(1) + distance_temp_add(1);
			vtk_vertices(point5, 2) = z_centroid + distance_temp(2) + distance_temp_add(2);

			//-----------------------------------------------------------------------------
			//point6 maxx maxy minz
			distance_temp(0) = half_a;
			distance_temp(1) = half_b;
			distance_temp(2) = -half_c;

			distance_temp_add = rotation_to_displacement(distance_temp(0), distance_temp(1), distance_temp(2), thetax, thetay, thetaz);

			vtk_vertices(point6, 0) = x_centroid + distance_temp(0) + distance_temp_add(0);
			vtk_vertices(point6, 1) = y_centroid + distance_temp(1) + distance_temp_add(1);
			vtk_vertices(point6, 2) = z_centroid + distance_temp(2) + distance_temp_add(2);

			//-----------------------------------------------------------------------------
			//point7 maxx maxy maxz
			distance_temp(0) = half_a;
			distance_temp(1) = half_b;
			distance_temp(2) = half_c;

			distance_temp_add = rotation_to_displacement(distance_temp(0), distance_temp(1), distance_temp(2), thetax, thetay, thetaz);

			vtk_vertices(point7, 0) = x_centroid + distance_temp(0) + distance_temp_add(0);
			vtk_vertices(point7, 1) = y_centroid + distance_temp(1) + distance_temp_add(1);
			vtk_vertices(point7, 2) = z_centroid + distance_temp(2) + distance_temp_add(2);

			//-----------------------------------------------------------------------------
			//point8 minx maxy maxz
			distance_temp(0) = -half_a;
			distance_temp(1) = half_b;
			distance_temp(2) = half_c;

			distance_temp_add = rotation_to_displacement(distance_temp(0), distance_temp(1), distance_temp(2), thetax, thetay, thetaz);

			vtk_vertices(point8, 0) = x_centroid + distance_temp(0) + distance_temp_add(0);
			vtk_vertices(point8, 1) = y_centroid + distance_temp(1) + distance_temp_add(1);
			vtk_vertices(point8, 2) = z_centroid + distance_temp(2) + distance_temp_add(2);

			int polygon_count1 = i * 6;
			int polygon_count2 = polygon_count1 + 1;
			int polygon_count3 = polygon_count2 + 1;
			int polygon_count4 = polygon_count3 + 1;
			int polygon_count5 = polygon_count4 + 1;
			int polygon_count6 = polygon_count5 + 1;

			//polygons
			//bottom face
			// 1 - 5 - 6 - 2
			vtk_polygons(polygon_count1, 1) = point1;
			vtk_polygons(polygon_count1, 2) = point5;
			vtk_polygons(polygon_count1, 3) = point6;
			vtk_polygons(polygon_count1, 4) = point2;

			//top face
			//3 - 7 - 8 - 4
			vtk_polygons(polygon_count2, 1) = point3;
			vtk_polygons(polygon_count2, 2) = point7;
			vtk_polygons(polygon_count2, 3) = point8;
			vtk_polygons(polygon_count2, 4) = point4;

			//miny face
			//1 - 2 - 3 - 4
			vtk_polygons(polygon_count3, 1) = point1;
			vtk_polygons(polygon_count3, 2) = point2;
			vtk_polygons(polygon_count3, 3) = point3;
			vtk_polygons(polygon_count3, 4) = point4;

			//maxy face
			//6 - 5 - 8 - 7
			vtk_polygons(polygon_count4, 1) = point6;
			vtk_polygons(polygon_count4, 2) = point5;
			vtk_polygons(polygon_count4, 3) = point8;
			vtk_polygons(polygon_count4, 4) = point7;

			//minx face
			//5 - 1 - 4 - 8
			vtk_polygons(polygon_count5, 1) = point5;
			vtk_polygons(polygon_count5, 2) = point1;
			vtk_polygons(polygon_count5, 3) = point4;
			vtk_polygons(polygon_count5, 4) = point8;

			//maxx face
			//2 - 6 - 7 - 3
			vtk_polygons(polygon_count6, 1) = point2;
			vtk_polygons(polygon_count6, 2) = point6;
			vtk_polygons(polygon_count6, 3) = point7;
			vtk_polygons(polygon_count6, 4) = point3;

		}

		element_vtk_file << vtk_vertices << "\n";
		element_vtk_file << "POLYGONS " << total_ele_num * 6 << " " << total_ele_num * 6 * 5 << "\n";
		element_vtk_file << vtk_polygons;

		element_vtk_file << "CELL_DATA " << total_ele_num * 6 << "\n";
		element_vtk_file << "VECTORS Force_Vectors double\n";
		for (int i = 0; i < total_ele_num; i++) {
			int current_dof = i * 6;

			element_vtk_file << 0 << " " << 0 << " " << 0 << "\n";
			element_vtk_file << 0 << " " << 0 << " " << F(current_dof + 2) / 1000.00 << "\n";

			element_vtk_file << 0 << " " << 0 << " " << 0 << "\n";
			element_vtk_file << 0 << " " << F(current_dof + 1) / 1000.00 << " " << 0 << "\n";

			element_vtk_file << 0 << " " << 0 << " " << 0 << "\n";
			element_vtk_file << F(current_dof) / 1000.00 << " " << 0 << " " << 0 << "\n";
		}
		element_vtk_file << "VECTORS Moment_Vectors double\n";
		for (int i = 0; i < total_ele_num; i++) {
			int current_dof = i * 6 + 3;

			element_vtk_file << 0 << " " << 0 << " " << 0 << "\n";
			element_vtk_file << 0 << " " << 0 << " " << F(current_dof + 2) / 1000000.00 << "\n";

			element_vtk_file << 0 << " " << 0 << " " << 0 << "\n";
			element_vtk_file << 0 << " " << F(current_dof + 1) / 1000000.00 << " " << 0 << "\n";

			element_vtk_file << 0 << " " << 0 << " " << 0 << "\n";
			element_vtk_file << F(current_dof) / 1000000.00 << " " << 0 << " " << 0 << "\n";
		}

		element_vtk_file << "VECTORS Displacement_Vectors double\n";
		for (int i = 0; i < total_ele_num; i++) {
			int current_dof = i * 6;

			element_vtk_file << 0 << " " << 0 << " " << 0 << "\n";
			element_vtk_file << 0 << " " << 0 << " " << U(current_dof + 2) << "\n";

			element_vtk_file << 0 << " " << 0 << " " << 0 << "\n";
			element_vtk_file << 0 << " " << U(current_dof + 1) << " " << 0 << "\n";

			element_vtk_file << 0 << " " << 0 << " " << 0 << "\n";
			element_vtk_file << U(current_dof) << " " << 0 << " " << 0 << "\n";
		}

		element_vtk_file << "VECTORS Rotation_Vectors double\n";
		for (int i = 0; i < total_ele_num; i++) {
			int current_dof = i * 6 + 3;

			element_vtk_file << 0 << " " << 0 << " " << 0 << "\n";
			element_vtk_file << 0 << " " << 0 << " " << U(current_dof + 2) << "\n";

			element_vtk_file << 0 << " " << 0 << " " << 0 << "\n";
			element_vtk_file << 0 << " " << U(current_dof + 1) << " " << 0 << "\n";

			element_vtk_file << 0 << " " << 0 << " " << 0 << "\n";
			element_vtk_file << U(current_dof) << " " << 0 << " " << 0 << "\n";
		}

		element_vtk_file << "SCALARS Force_Scalars double 3\n";
		element_vtk_file << "LOOKUP_TABLE Force\n";
		for (int i = 0; i < total_ele_num; i++) {
			int current_dof = i * 6;

			element_vtk_file << F(current_dof) / 1000.00 << " " << F(current_dof + 1) / 1000.00 << " " << F(current_dof + 2) / 1000.00 << "\n";
			element_vtk_file << F(current_dof) / 1000.00 << " " << F(current_dof + 1) / 1000.00 << " " << F(current_dof + 2) / 1000.00 << "\n";
			element_vtk_file << F(current_dof) / 1000.00 << " " << F(current_dof + 1) / 1000.00 << " " << F(current_dof + 2) / 1000.00 << "\n";
			element_vtk_file << F(current_dof) / 1000.00 << " " << F(current_dof + 1) / 1000.00 << " " << F(current_dof + 2) / 1000.00 << "\n";
			element_vtk_file << F(current_dof) / 1000.00 << " " << F(current_dof + 1) / 1000.00 << " " << F(current_dof + 2) / 1000.00 << "\n";
			element_vtk_file << F(current_dof) / 1000.00 << " " << F(current_dof + 1) / 1000.00 << " " << F(current_dof + 2) / 1000.00 << "\n";
		}
		element_vtk_file << "SCALARS Moment_Scalars double 3\n";
		element_vtk_file << "LOOKUP_TABLE Moment\n";
		for (int i = 0; i < total_ele_num; i++) {
			int current_dof = i * 6 + 3;

			element_vtk_file << F(current_dof) / 1000000.00 << " " << F(current_dof + 1) / 1000000.00 << " " << F(current_dof + 2) / 1000000.00 << "\n";
			element_vtk_file << F(current_dof) / 1000000.00 << " " << F(current_dof + 1) / 1000000.00 << " " << F(current_dof + 2) / 1000000.00 << "\n";
			element_vtk_file << F(current_dof) / 1000000.00 << " " << F(current_dof + 1) / 1000000.00 << " " << F(current_dof + 2) / 1000000.00 << "\n";
			element_vtk_file << F(current_dof) / 1000000.00 << " " << F(current_dof + 1) / 1000000.00 << " " << F(current_dof + 2) / 1000000.00 << "\n";
			element_vtk_file << F(current_dof) / 1000000.00 << " " << F(current_dof + 1) / 1000000.00 << " " << F(current_dof + 2) / 1000000.00 << "\n";
			element_vtk_file << F(current_dof) / 1000000.00 << " " << F(current_dof + 1) / 1000000.00 << " " << F(current_dof + 2) / 1000000.00 << "\n";
		}

		element_vtk_file << "SCALARS Displacement_Scalars double 3\n";
		element_vtk_file << "LOOKUP_TABLE Displacement\n";
		for (int i = 0; i < total_ele_num; i++) {
			int current_dof = i * 6;

			element_vtk_file << U(current_dof) << " " << U(current_dof + 1) << " " << U(current_dof + 2) << "\n";
			element_vtk_file << U(current_dof) << " " << U(current_dof + 1) << " " << U(current_dof + 2) << "\n";
			element_vtk_file << U(current_dof) << " " << U(current_dof + 1) << " " << U(current_dof + 2) << "\n";
			element_vtk_file << U(current_dof) << " " << U(current_dof + 1) << " " << U(current_dof + 2) << "\n";
			element_vtk_file << U(current_dof) << " " << U(current_dof + 1) << " " << U(current_dof + 2) << "\n";
			element_vtk_file << U(current_dof) << " " << U(current_dof + 1) << " " << U(current_dof + 2) << "\n";
		}
		element_vtk_file << "SCALARS Rotation_Scalars double 3\n";
		element_vtk_file << "LOOKUP_TABLE Rotation\n";
		for (int i = 0; i < total_ele_num; i++) {
			int current_dof = i * 6 + 3;

			element_vtk_file << U(current_dof) << " " << U(current_dof + 1) << " " << U(current_dof + 2) << "\n";
			element_vtk_file << U(current_dof) << " " << U(current_dof + 1) << " " << U(current_dof + 2) << "\n";
			element_vtk_file << U(current_dof) << " " << U(current_dof + 1) << " " << U(current_dof + 2) << "\n";
			element_vtk_file << U(current_dof) << " " << U(current_dof + 1) << " " << U(current_dof + 2) << "\n";
			element_vtk_file << U(current_dof) << " " << U(current_dof + 1) << " " << U(current_dof + 2) << "\n";
			element_vtk_file << U(current_dof) << " " << U(current_dof + 1) << " " << U(current_dof + 2) << "\n";
		}

		element_vtk_file << "SCALARS Principal_Stress_1 double 1\n";
		element_vtk_file << "LOOKUP_TABLE PrinStress1\n";
		for (int i = 0; i < total_ele_num; i++) {

			element_vtk_file << element_principal_stresses(i, 0) << "\n";
			element_vtk_file << element_principal_stresses(i, 0) << "\n";
			element_vtk_file << element_principal_stresses(i, 0) << "\n";
			element_vtk_file << element_principal_stresses(i, 0) << "\n";
			element_vtk_file << element_principal_stresses(i, 0) << "\n";
			element_vtk_file << element_principal_stresses(i, 0) << "\n";
		}

		element_vtk_file << "SCALARS Principal_Stress_2 double 1\n";
		element_vtk_file << "LOOKUP_TABLE PrinStress2\n";
		for (int i = 0; i < total_ele_num; i++) {

			element_vtk_file << element_principal_stresses(i, 1) << "\n";
			element_vtk_file << element_principal_stresses(i, 1) << "\n";
			element_vtk_file << element_principal_stresses(i, 1) << "\n";
			element_vtk_file << element_principal_stresses(i, 1) << "\n";
			element_vtk_file << element_principal_stresses(i, 1) << "\n";
			element_vtk_file << element_principal_stresses(i, 1) << "\n";
		}

		element_vtk_file << "SCALARS Principal_Stress_3 double 1\n";
		element_vtk_file << "LOOKUP_TABLE PrinStress3\n";
		for (int i = 0; i < total_ele_num; i++) {

			element_vtk_file << element_principal_stresses(i, 2) << "\n";
			element_vtk_file << element_principal_stresses(i, 2) << "\n";
			element_vtk_file << element_principal_stresses(i, 2) << "\n";
			element_vtk_file << element_principal_stresses(i, 2) << "\n";
			element_vtk_file << element_principal_stresses(i, 2) << "\n";
			element_vtk_file << element_principal_stresses(i, 2) << "\n";
		}

		element_vtk_file << "SCALARS Principal_Stress_Max double 1\n";
		element_vtk_file << "LOOKUP_TABLE PrinStressMax\n";
		for (int i = 0; i < total_ele_num; i++) {

			element_vtk_file << element_principal_stresses(i, 9) << "\n";
			element_vtk_file << element_principal_stresses(i, 9) << "\n";
			element_vtk_file << element_principal_stresses(i, 9) << "\n";
			element_vtk_file << element_principal_stresses(i, 9) << "\n";
			element_vtk_file << element_principal_stresses(i, 9) << "\n";
			element_vtk_file << element_principal_stresses(i, 9) << "\n";
		}

		element_vtk_file << "SCALARS sigma_xx double 1\n";
		element_vtk_file << "LOOKUP_TABLE sigma_xx\n";
		for (int i = 0; i < total_ele_num; i++) {

			element_vtk_file << element_principal_stresses(i, 3) << "\n";
			element_vtk_file << element_principal_stresses(i, 3) << "\n";
			element_vtk_file << element_principal_stresses(i, 3) << "\n";
			element_vtk_file << element_principal_stresses(i, 3) << "\n";
			element_vtk_file << element_principal_stresses(i, 3) << "\n";
			element_vtk_file << element_principal_stresses(i, 3) << "\n";
		}

		element_vtk_file << "SCALARS sigma_yy double 1\n";
		element_vtk_file << "LOOKUP_TABLE sigma_yy\n";
		for (int i = 0; i < total_ele_num; i++) {

			element_vtk_file << element_principal_stresses(i, 4) << "\n";
			element_vtk_file << element_principal_stresses(i, 4) << "\n";
			element_vtk_file << element_principal_stresses(i, 4) << "\n";
			element_vtk_file << element_principal_stresses(i, 4) << "\n";
			element_vtk_file << element_principal_stresses(i, 4) << "\n";
			element_vtk_file << element_principal_stresses(i, 4) << "\n";
		}

		element_vtk_file << "SCALARS sigma_zz double 1\n";
		element_vtk_file << "LOOKUP_TABLE sigma_zz\n";
		for (int i = 0; i < total_ele_num; i++) {

			element_vtk_file << element_principal_stresses(i, 5) << "\n";
			element_vtk_file << element_principal_stresses(i, 5) << "\n";
			element_vtk_file << element_principal_stresses(i, 5) << "\n";
			element_vtk_file << element_principal_stresses(i, 5) << "\n";
			element_vtk_file << element_principal_stresses(i, 5) << "\n";
			element_vtk_file << element_principal_stresses(i, 5) << "\n";
		}

		element_vtk_file << "SCALARS tau_xy double 1\n";
		element_vtk_file << "LOOKUP_TABLE tau_xy\n";
		for (int i = 0; i < total_ele_num; i++) {

			element_vtk_file << element_principal_stresses(i, 6) << "\n";
			element_vtk_file << element_principal_stresses(i, 6) << "\n";
			element_vtk_file << element_principal_stresses(i, 6) << "\n";
			element_vtk_file << element_principal_stresses(i, 6) << "\n";
			element_vtk_file << element_principal_stresses(i, 6) << "\n";
			element_vtk_file << element_principal_stresses(i, 6) << "\n";
		}

		element_vtk_file << "SCALARS tau_xz double 1\n";
		element_vtk_file << "LOOKUP_TABLE tau_xz\n";
		for (int i = 0; i < total_ele_num; i++) {

			element_vtk_file << element_principal_stresses(i, 7) << "\n";
			element_vtk_file << element_principal_stresses(i, 7) << "\n";
			element_vtk_file << element_principal_stresses(i, 7) << "\n";
			element_vtk_file << element_principal_stresses(i, 7) << "\n";
			element_vtk_file << element_principal_stresses(i, 7) << "\n";
			element_vtk_file << element_principal_stresses(i, 7) << "\n";
		}

		element_vtk_file << "SCALARS tau_yz double 1\n";
		element_vtk_file << "LOOKUP_TABLE tau_yz\n";
		for (int i = 0; i < total_ele_num; i++) {

			element_vtk_file << element_principal_stresses(i, 8) << "\n";
			element_vtk_file << element_principal_stresses(i, 8) << "\n";
			element_vtk_file << element_principal_stresses(i, 8) << "\n";
			element_vtk_file << element_principal_stresses(i, 8) << "\n";
			element_vtk_file << element_principal_stresses(i, 8) << "\n";
			element_vtk_file << element_principal_stresses(i, 8) << "\n";
		}

		element_vtk_file << "SCALARS active_springs double 1\n";
		element_vtk_file << "LOOKUP_TABLE active_springs\n";
		for (int i = 0; i < total_ele_num; i++) {

			element_vtk_file << element_principal_stresses(i, 10) << "\n";
			element_vtk_file << element_principal_stresses(i, 10) << "\n";
			element_vtk_file << element_principal_stresses(i, 10) << "\n";
			element_vtk_file << element_principal_stresses(i, 10) << "\n";
			element_vtk_file << element_principal_stresses(i, 10) << "\n";
			element_vtk_file << element_principal_stresses(i, 10) << "\n";
		}

		element_vtk_file.close();
		int draw_time_end = time(NULL);
		int draw_runtime = draw_time_end - draw_time_start;
		std::cout << "Mesh drawn! (Runtime: " << draw_runtime << " seconds)\n";
	}
	else {
		std::cout << "\a\n\nERROR: Failed to write to file!\n";
	}
}

void spring_drawer(string name, MatrixXd spring, int total_spring_num, int choice_material, int choice_condition) {
	int spring_printed_counter = 0;
	VectorXi springs_to_draw = VectorXi::Zero(total_spring_num);

	for (int i = 0; i < total_spring_num; i++) {
		if (choice_material == -1) { //all the materials
			if (choice_condition == spring(i, 10)) {  //but only a specific condition
				spring_printed_counter++;
				springs_to_draw(i) = 1;
			}
			else if (choice_condition == -1) { //and all the conditions
				spring_printed_counter++;
				springs_to_draw(i) = 1;
			}
		}
		else if (choice_material == spring(i, 11)) { // a specific material
			if (choice_condition == spring(i, 10)) { // with a specific condition
				spring_printed_counter++;
				springs_to_draw(i) = 1;
			}
			else if (choice_condition == -1) { //or all the conditions
				spring_printed_counter++;
				springs_to_draw(i) = 1;
			}
		}
	}
	string filename_spring_vtk = name + "PARAVIEW_" + "spring.vtk";
	ofstream spring_vtk_file;
	spring_vtk_file.open(filename_spring_vtk, ofstream::app);
	if (spring_vtk_file.is_open()) {

		std::cout << "Drawing the mesh springs...\n";
		int draw_time_start = time(NULL);

		spring_vtk_file << "# vtk DataFile Version 2.0\n";
		spring_vtk_file << "Springs\n";
		spring_vtk_file << "ASCII\n";
		spring_vtk_file << "DATASET POLYDATA\n";
		spring_vtk_file << "POINTS " << spring_printed_counter * 2 << " double\n";

		if (spring_printed_counter > 0) {

			for (int i = 0; i < total_spring_num; i++) {
				int checker = springs_to_draw(i);
				if (checker == 1) {
					//cout << "test";
					spring_vtk_file << spring(i, 4) << " " << spring(i, 5) << " " << spring(i, 6) << "\n";
					spring_vtk_file << spring(i, 7) << " " << spring(i, 8) << " " << spring(i, 9) << "\n";
				}
			}

			spring_vtk_file << "LINES " << spring_printed_counter << " " << (spring_printed_counter * 3) << "\n";

			for (int i = 0; i < spring_printed_counter; i++) {
				spring_vtk_file << 2 << " " << (i * 2) << " " << (i * 2 + 1) << "\n";
			}

			spring_vtk_file << "CELL_DATA " << spring_printed_counter << "\n";

			if (choice_condition != -1) {

				//calculate spring lengths and put as data
				spring_vtk_file << "SCALARS Length double 3\n";
				spring_vtk_file << "LOOKUP_TABLE Length\n";

				for (int i = 0; i < total_spring_num; i++) {

					int checker = springs_to_draw(i);
					if (checker == 1) {
						int current_spring = i;

						double sx1 = spring(current_spring, 4);
						double sy1 = spring(current_spring, 5);
						double sz1 = spring(current_spring, 6);

						double sx2 = spring(current_spring, 7);
						double sy2 = spring(current_spring, 8);
						double sz2 = spring(current_spring, 9);

						double differencex = sx2 - sx1;
						double differencey = sy2 - sy1;
						double differencez = sz2 - sz1;

						spring_vtk_file << differencex << " " << differencey << " " << differencez << "\n";
					}
				}
			}

			spring_vtk_file << "SCALARS Strain double 1\n";
			spring_vtk_file << "LOOKUP_TABLE Strain\n";

			for (int i = 0; i < total_spring_num; i++) {

				int checker = springs_to_draw(i);
				if (checker == 1) {
					spring_vtk_file << spring(i, 13) << "\n";
				}
			}


			spring_vtk_file << "SCALARS Stress double 1\n";
			spring_vtk_file << "LOOKUP_TABLE Stress\n";

			for (int i = 0; i < total_spring_num; i++) {

				int checker = springs_to_draw(i);
				if (checker == 1) {
					spring_vtk_file << spring(i, 14) << "\n";
				}
			}

		}

		spring_vtk_file.close();

		int draw_time_end = time(NULL);
		int draw_runtime = draw_time_end - draw_time_start;

		std::cout << "Mesh springs drawn! (Runtime: " << draw_runtime << " seconds)\n\n";
	}
	else {
		std::cout << "\a\n\nERROR: Failed to write to file!\n";
	}
}

