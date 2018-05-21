#include "CKern.h"
#include "GMPLMultiDimGp.h"
//#include <easy/profiler.h>
#include <chrono>
#include <thread>
int main(int argc, char *argv[]) {
//    EASY_PROFILER_ENABLE;
    const string fileName = "/home/juraj/git/master_thesis/idsc_ws/src/lbnmpcc/GPc/gpml_mat/GPmeanGP350.mat";

    GMPLMultiDimGp gp;
    gp.readMatlabFile(fileName, "gpInfo");

    CMatrix x;
    x.readMatlabFile(fileName, "x");
    const unsigned int nrows = x.getRows();
    CMatrix            mu(nrows, 3), var(nrows, 3);

    std::this_thread::sleep_for(std::chrono::milliseconds(10*1000));
    std::cout << "start" << std::endl;
    for (int i = 0; i < 100; i++) {
//        EASY_FUNCTION(profiler::colors::Magenta); // Magenta block with name "frame"
        gp.out(mu, var, x);
    }



//    profiler::dumpBlocksToFile("test_profile.prof");

}



