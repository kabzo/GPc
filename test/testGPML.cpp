#include "CKern.h"
#include "CMatrix.h"
#include "CGp.h"
#include "CClctrl.h"
#include "CRbfKern.hpp"
#include "CRbfardKern.hpp"

class MatlabParse : public CMatInterface {
 public:
    MatlabParse() {}
    void fromMxArray(const mxArray *matlabArray) {
        xt.fromMxArray(mxArrayExtractMxArrayField(matlabArray, "xt"));
        mxArray* hypArray=mxArrayExtractMxArrayField(matlabArray, "hyp");

    }

    mxArray* toMxArray() const {
        return NULL;
    }

    CMatrix xt;
};

int main(int argc, char *argv[]) {
//    const string fileName = "../matfiles/testGpfitc.mat";
    const string fileName = "../gpml_mat/GP_result.mat";
    MatlabParse parse;
    parse.readMatlabFile(fileName, "gpInfo");
//    CMatrix X;
//    CMatrix xt;
//    CMatrix yt;
//    X.readMatlabFile(fileName, "x");
//    xt.readMatlabFile(fileName, "xt");
//    yt.readMatlabFile(fileName, "yt");
//    CMatrix scales;
//    scales.readMatlabFile(fileName, "scales");
//    CMatrix variance;
//    variance.readMatlabFile(fileName, "variance");
//
//    std::cout << X << std::endl;
//    std::cout << scales << std::endl;
//    std::cout << variance << std::endl;
//
//    CRbfardKern kern(X);
//    kern.setParamName("inverseWidth", 1);
//    kern.setVariance(variance.getVal(0,0));
//    kern.setScales(scales);
//    CMatrix K(X.getRows(), X.getRows());
//    kern.compute(K, X);
//    std::cout << K << std::endl;

}



