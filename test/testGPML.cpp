#include "CKern.h"
#include "GMPLMultiDimGp.h"

int main(int argc, char *argv[]) {
    const string fileName = "../gpml_mat/GP_result.mat";

    GMPLMultiDimGp gp;
    gp.readMatlabFile(fileName, "gpInfo");

    CMatrix x;
    x.readMatlabFile(fileName, "x");
    const unsigned int nrows = x.getRows();
    CMatrix mu(nrows, 3),var(nrows, 3);
    gp.out(mu,var,x);

    std::cout << mu << std::endl;
    std::cout << var << std::endl;
}



