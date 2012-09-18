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
    int c;
    bool error = false;
    bool has_group = false;
    bool has_data = false;
    string pgm = argv[0];
    errno = 0;
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
    if (error) {
	cerr << pgm
	     << " [-g group-num] [-d data-count]" << endl;
	cerr << "--group-num,-g  group number of kernel call."
	     << endl;
	cerr << "--data-count,-d count       generate random number count."
	     << endl;
	return false;
    }
    return true;
}
