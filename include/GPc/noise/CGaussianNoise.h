//
// Created by juraj on 15/5/18.
//

#ifndef CMATRIX_CGAUSSIANNOISE_H
#define CMATRIX_CGAUSSIANNOISE_H

#include "CNoise.h"

// The Gaussian noise model as commonly used in regression.
class CGaussianNoise : public CNoise {
 public:
    // constructors
    CGaussianNoise()
    {
        _init();
    }
    CGaussianNoise(CMatrix* pyin)
    {
        _init();
        setTarget(pyin);
        initNames();
        initVals();
        initParams();
    }
    ~CGaussianNoise();

    void initStoreage();
    void initNames();
    void initVals();
    void initParams();

    ostream& display(ostream& os);
    void setParams(const CMatrix& params);
    void setParam(double val, unsigned int index);
    void getParams(CMatrix& params) const;
    double getParam(unsigned int index) const;
    void getGradParams(CMatrix& g) const;
    void getGradInputs(double& gmu, double& gvs, unsigned int i, unsigned int j) const;
    void getNuG(CMatrix& g, CMatrix& nu, unsigned int index) const;
    void updateSites(CMatrix& m, CMatrix& beta, unsigned int actIndex, const CMatrix& g, const CMatrix& nu, unsigned int index) const;
    void test(const CMatrix& muout, const CMatrix& varSigmaOut, const CMatrix& yTest) const;
    void out(CMatrix& yPred, const CMatrix& muTest, const CMatrix& varSigmaTest, const bool variance = false) const;
    void out(CMatrix& yPred, CMatrix& errorBarOut, const CMatrix& muTest, const CMatrix& varSigmaTest, const bool variance = false) const;
    void likelihoods(CMatrix& L, const CMatrix& muTest, const CMatrix& varSigmaTest, const CMatrix& yTest) const;
    double logLikelihood(const CMatrix& muTest, const CMatrix& varSigmaTest, const CMatrix& yTest) const;
    double logLikelihood() const
    {
        return logLikelihood(mu, varSigma, *py);
    }
    //void  getGradX(CMatrix& gX, const CMatrix& dmu, const CMatrix& cvs);

    inline double getMu(unsigned int i, unsigned int j) const
    {
        return mu.getVal(i, j);
    }
    inline void setMu(double val, unsigned int i, unsigned int j)
    {
        mu.setVal(val, i, j);
    }
    inline double getVarSigma(unsigned int i, unsigned int j) const
    {
        return varSigma.getVal(i, j);
    }
    inline void setVarSigma(double val, unsigned int i, unsigned int j)
    {
        SANITYCHECK(!isnan(val));
        SANITYCHECK(val>=0);
        varSigma.setVal(val, i, j);
    }
    inline double getTarget(unsigned int i, unsigned int j) const
    {
        return py->getVal(i, j);
    }
    double getBiasVal(unsigned int index) const
    {
        BOUNDCHECK(index<getOutputDim());
        BOUNDCHECK(index>=0);
        return bias.getVal(0, index);
    }
    void setBiasVal(double val, unsigned int index)
    {
        BOUNDCHECK(index<getOutputDim());
        BOUNDCHECK(index>=0);
        bias.setVal(val, 0, index);
    }
    void setBias(const CMatrix& bia)
    {
        DIMENSIONMATCH(bia.getRows()==1);
        DIMENSIONMATCH(bia.getCols()==getOutputDim());
        bias.deepCopy(bia);
    }

#ifdef _NDLMATLAB
    // Adds parameters to the mxArray.
    void addParamToMxArray(mxArray* matlabArray) const;
    // Gets the parameters from the mxArray.
    void extractParamFromMxArray(const mxArray* matlabArray);
#endif
 private:

    void _init();

    double sigma2;
    CMatrix bias;
};


#endif //CMATRIX_CGAUSSIANNOISE_H
