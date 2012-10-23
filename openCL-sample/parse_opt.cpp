#include <unistd.h>
#include <getopt.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include "parse_opt.h"

using namespace std;

bool parse_opt(options& opt, int argc, char **argv) {
    errno = 0;
    string pgm = argv[0];
    bool error = false;
#if 0
    int c;
    bool has_group = false;
    bool has_data = false;
    static struct option longopts[] = {
	{"group-number", required_argument, NULL, 'g'},
	{"data-cout", required_argument, NULL, 'd'},
	{NULL, 0, NULL, 0}};
    for (;;) {
	c = getopt_long(argc, argv, "g:d:", longopts, NULL);
	if (error) {
	    break;
	}
	if (c == -1) {
	    break;
	}
	switch (c) {
	case 'g':
	    opt.group_num = strtol(optarg, NULL, 10);
	    if (errno) {
		error = true;
		cerr << "group num error!" << endl;
		cerr << strerror(errno) << endl;
	    }
	    has_group = true;
	    break;
	case 'd':
	    opt.data_count = strtol(optarg, NULL, 10);
	    if (errno) {
		error = true;
		cerr << "data count error!" << endl;
		cerr << strerror(errno) << endl;
	    }
	    has_data = true;
	    break;
	case '?':
	default:
	    error = true;
	    break;
	}
    }
    argc -= optind;
    argv += optind;
    if (!has_group || !has_data) {
	error = true;
    }
#endif
    if (argc <= 2) {
	error = true;
    }
    do {
	if (error) {
	    break;
	}
	opt.group_num = strtol(argv[1], NULL, 10);
	if (errno) {
	    error = true;
	    cerr << "group num error!" << endl;
	    cerr << strerror(errno) << endl;
	    break;
	}
	opt.data_count = strtol(argv[2], NULL, 10);
	if (errno) {
	    error = true;
	    cerr << "data count error!" << endl;
	    cerr << strerror(errno) << endl;
	    break;
	}
    } while (0);
    if (error) {
	cerr << pgm
	     << " group-num data-count" << endl;
	cerr << "group-num  group number of kernel call."
	     << endl;
	cerr << "data-count number of generate random numbers."
	     << endl;
	return false;
    }
    return true;
}
