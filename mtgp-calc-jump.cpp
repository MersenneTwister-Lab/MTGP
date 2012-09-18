/**
 * @file calc-jump.cpp
 *
 * @brief calc jump function.
 *
 * @author Mutsuo Saito (Hiroshima University)
 * @author Makoto Matsumoto (The University of Tokyo)
 *
 * Copyright (C) 2012 Mutsuo Saito, Makoto Matsumoto,
 * Hiroshima University and The University of Tokyo.
 * All rights reserved.
 *
 * The 3-clause BSD License is applied to this software, see
 * LICENSE.txt
 */
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <inttypes.h>
#include <stdint.h>
#include <time.h>
#include <NTL/GF2X.h>
#include <NTL/vec_GF2.h>
#include <NTL/ZZ.h>
#include "mtgp-calc-jump.hpp"

using namespace NTL;
using namespace std;

static void read_file(GF2X& lcmpoly, long line_no, const string& file);

static void usage(char * name) {
    cout << name << " [-a] jump-step poly-file" << endl;
    cout << "    -a       : output array format" << endl;
    cout << "    jump-step: a number between zero and 2^{MTGP_MEXP}-1.\n"
	 << "               large decimal number is allowed." << endl;
    cout << "    poly-file: output of calc-poly"
	 << "file" << endl;
}
int main(int argc, char * argv[]) {
    if (argc <= 2) {
	usage(argv[0]);
	return -1;
    }
    int offset = 0;
    if (strcmp(argv[1],"-a") == 0) {
	if (argc <= 3) {
	    usage(argv[0]);
	    return -1;
	}
	offset = 1;
    }
    string step_string = argv[offset + 1];
    string filename = argv[offset + 2];
    long no = 0;
    GF2X lcmpoly;
    read_file(lcmpoly, no, filename);
    long degree = deg(lcmpoly);
    ZZ step;
    stringstream ss(step_string);
    ss >> step;
    if (offset == 0) {
	string jump_str;
	calc_jump(jump_str, step, lcmpoly);
#if defined(DEBUG)
	cout << "deg lcmpoly:" << dec << deg(lcmpoly) << endl;
#endif
	cout << "jump polynomial:" << endl;
	cout << jump_str << endl;
    } else {
	int size = degree / 32 + 2;
	uint32_t array[size];
	calc_jump(array, size, step, lcmpoly);
	cout << "/* jump step:" << step_string << " */" << endl;
	cout << "uint32_t jump_array[] = {";
	for (int i = 0; i < size; i++) {
	    if (i % 5 == 0) {
		cout << endl;
	    }
	    cout << "0x" << setfill('0') << setw(8) << hex << array[i]
		 << ",";
	}
	cout << endl;
	cout << "};" << endl;
    }
    return 0;
}

static void read_file(GF2X& lcmpoly, long line_no, const string& file)
{
    ifstream ifs(file.c_str());
    string line;
    for (int i = 0; i < line_no; i++) {
	getline(ifs,line);
	getline(ifs,line);
    }
    if (ifs) {
	getline(ifs,line);
	line = "";
	getline(ifs,line);
    }
    stringtopoly(lcmpoly, line);
}
