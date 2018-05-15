//
// Created by juraj on 13/5/18.
//

#ifndef CMATRIX_CRBFKERN_HPP
#define CMATRIX_CRBFKERN_HPP
#include "CKern.h"
// RBF Kernel, also known as the Gaussian or squared exponential kernel.
class CRbfKern : public CKern {
 public:
    CRbfKern();
    CRbfKern(unsigned int inDim);
    CRbfKern(const CMatrix &X);
    ~CRbfKern();
    CRbfKern(const CRbfKern &);
    CRbfKern *clone() const {
        return new CRbfKern(*this);
    }
    double getVariance() const;
    void setVariance(double val) {
        variance = val;
    }
    void setInverseWidth(double val) {
        inverseWidth = val;
    }
    double getInverseWidth() const {
        return inverseWidth;
    }
    void setLengthScale(double val) {
        inverseWidth = 1 / (val * val);
    }
    double getLengthScale() const {
        return 1 / sqrt(inverseWidth);
    }
    void setInitParam();
    double diagComputeElement(const CMatrix &X, unsigned int index) const;
    void diagCompute(CMatrix &d, const CMatrix &X) const;
    void setParam(double val, unsigned int paramNum);
    double getParam(unsigned int paramNum) const;
    void getGradX(CMatrix &g, const CMatrix &X, unsigned int pointNo, const CMatrix &X2, bool addG = false) const;
    void getDiagGradX(CMatrix &g, const CMatrix &X, bool addG = false) const;
    double getWhite() const;
    double computeElement(const CMatrix &X1, unsigned int index1,
                          const CMatrix &X2, unsigned int index2) const;
    void getGradParams(CMatrix &g, const CMatrix &X, const CMatrix &cvGrd, bool regularise = true) const;
    void getGradParams(CMatrix &g,
                       const CMatrix &X,
                       const CMatrix &X2,
                       const CMatrix &cvGrd,
                       bool regularise = true) const;
    double getGradParam(unsigned int index, const CMatrix &X, const CMatrix &X2, const CMatrix &cvGrd) const;
    double getGradParam(unsigned int index, const CMatrix &X, const CMatrix &cvGrd) const;
    void updateX(const CMatrix &X);

 private:
    void _init();
    double          variance;
    double          inverseWidth;
    mutable CMatrix Xdists;

};


#endif //CMATRIX_CRBFKERN_HPP
