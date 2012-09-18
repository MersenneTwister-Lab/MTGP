#ifndef PARSE_OPT_H
#define PARSE_OPT_H

#include <stdint.h>
#include <string>

class options {
public:
    int data_count;
    int group_num;
};

bool parse_opt(options& opt, int argc, char **argv);

#endif
