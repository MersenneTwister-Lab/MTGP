/**
 * @file calc-poly64.cpp
 *
 * @brief calculate characteristic polynomial for 32bit mtgp.
 *
 * @author Mutsuo Saito (Hiroshima University)
 * @author Makoto Matsumoto (The University of Tokyo)
 *
 * Copyright (c) 2012 Mutsuo Saito, Makoto Matsumoto, Hiroshima
 * University and University of Tokyo. All rights reserved.
 *
 * The new BSD License is applied to this software, see LICENSE.txt
 */
#include <stdint.h>
#include <inttypes.h>
#include <time.h>
#include <string.h>
#include <string>
#include <NTL/GF2XFactoring.h>
#include "linear_generator.hpp"
#include "calc_jump.hpp"
#include "mtgp64.hpp"

using namespace std;
using namespace MTToolBox;
using namespace mtgp;

int main(int argc, char** argv) {
//# sha1, mexp, type, id, pos, sh1, sh2, tbl_0, tbl_1, tbl_2, tbl_3,tmp_0, tmp_1, tmp_2, tmp_3, mask, weight, delta
    if (argc < 2) {
	cout << argv[0] << " sha1,mexp,type,id,pos,sh1,sh2,tbl_0,tbl_1,tbl_2,"
	     << "tbl_3,tmp_0,tmp_1,tmp_3,mask,..." << endl;
	return -1;
    }
    int len = strlen(argv[1]);
    char str[len + 1];
    strcpy(str, argv[1]);
    char * p;
    mtgp_param<uint64_t> param;
    p = strtok(str, ","); // sha1
    string sha1 = p;
    p = strtok(NULL, ","); // mexp
    param.mexp = strtol(p, NULL, 10);
    p = strtok(NULL, ","); // type
    string type = p;
    if (type != "uint64_t") {
	cout << "This parameter is not for mtgp64" << endl;
	cout << p << endl;
	return -1;
    }
    p = strtok(NULL, ","); // id
    param.id = strtol(p, NULL, 10);
    p = strtok(NULL, ","); // pos
    param.pos = strtol(p, NULL, 10);
    p = strtok(NULL, ","); // sh1
    param.sh1 = strtol(p, NULL, 10);
    p = strtok(NULL, ","); // sh2
    param.sh2 = strtol(p, NULL, 10);
    for (int i = 0; i < 4; i++) {
	p = strtok(NULL, ","); // tbl_0,...3
	param.tbl[i] = strtoull(p, NULL, 16);
    }
    for (int i = 0; i < 4; i++) {
	p = strtok(NULL, ","); // tbl_0,...3
	param.tmp_tbl[i] = strtoull(p, NULL, 16);
    }
    p = strtok(NULL, ","); // mask
    param.mask = strtoull(p, NULL, 16);
    mtgp64 mtgp(param.mexp, param.id);
    memset(param.p, 0, sizeof(param.p));
    mtgp.fill_table(param.p, param.tbl, 16);
    mtgp.set_param(param);
    mtgp.setup_temper();

    typedef linear_generator<uint64_t, mtgp64> lg64;
    lg64 g(mtgp);
    g.seeding(1);
    shared_ptr<GF2X> minpoly = g.get_minpoly();
    if (deg(*minpoly) != param.mexp) {
	cerr << "degree mismatch!" << endl;
	cerr << "deg minpoly:" << deg(*minpoly) << endl;
	return -1;
    }
    if (!IterIrredTest(*minpoly)) {
	cerr << "error not irreducible" << endl;
	return -1;
    }
    string str_poly;
    polytostring(str_poly, *minpoly);
//# sha1, mexp, type, id, pos, sh1, sh2, tbl_0, tbl_1, tbl_2, tbl_3,tmp_0, tmp_1, tmp_2, tmp_3, mask, weight, delta
    cout << "# \"" << sha1 << "\","
	 << dec << param.mexp << ","
	 << type << ","
	 << dec << param.id << ","
	 << dec << param.pos << ","
	 << dec << param.sh1 << ","
	 << dec << param.sh2 << ",";
    for (int i = 0; i < 4; i++) {
	cout << "0x" << hex << setw(16) << setfill('0') << param.tbl[i] << ",";    }
    cout << endl;
    cout << str_poly << endl;
    return 0;
}
