//
// Created by juraj on 13/5/18.
//

#ifndef CMATRIX_CRBFARDKERN_HPP
#define CMATRIX_CRBFARDKERN_HPP
#include "CKern.h"

// RBF ARD Kernel --- automatic relevance determination of the RBF kernel.
class CRbfardKern : public CArdKern {
 public:
    CRbfardKern();
    CRbfardKern(unsigned int inDim);
    CRbfardKern(const CMatrix &X);
    ~CRbfardKern();
    CRbfardKern(const CRbfardKern &);
    CRbfardKern *clone() const {
        return new CRbfardKern(*this);
    }
    double getVariance() const;
    void setVariance(double val) {
        variance = val;
    }
    void setInitParam();
    double diagComputeElement(const CMatrix &X, unsigned int index) const;
    void setParam(double val, unsigned int paramNum);
    double getParam(unsigned int paramNum) const;
    void getGradX(CMatrix &g, const CMatrix &X, unsigned int pointNo, const CMatrix &X2, bool addG = false) const;
    void getDiagGradX(CMatrix &g, const CMatrix &X, bool addGrad = false) const;
    double getWhite() const;
    double computeElement(const CMatrix &X1, unsigned int index1,
                          const CMatrix &X2, unsigned int index2) const;
    void getGradParams(CMatrix &g,
                       const CMatrix &X,
                       const CMatrix &X2,
                       const CMatrix &cvGrd,
                       bool regularise = true) const;
    void getGradParams(CMatrix &g, const CMatrix &X, const CMatrix &cvGrd, bool regularise = true) const;
    double getGradParam(unsigned int index, const CMatrix &X, const CMatrix &X2, const CMatrix &cvGrd) const;
    double getGradParam(unsigned int index, const CMatrix &X, const CMatrix &cvGrd) const;
    void setScales(const CMatrix &s){
        scales = s;
    }
 private:
    void _init();
    double          variance;
    double          inverseWidth;
    mutable CMatrix gscales;
};


#endif //CMATRIX_CRBFARDKERN_HPP
