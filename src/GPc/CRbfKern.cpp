//
// Created by juraj on 13/5/18.
//

#include "CRbfKern.hpp"

// the RBF kernel.
CRbfKern::CRbfKern() : CKern()
{
    _init();
}
CRbfKern::CRbfKern(unsigned int inDim) : CKern(inDim)
{
    _init();
    setInputDim(inDim);
}
CRbfKern::CRbfKern(const CMatrix& X) : CKern(X)
{
    _init();
    setInputDim(X.getCols());
}
CRbfKern::CRbfKern(const CRbfKern& kern) : CKern(kern)
{
    _init();
    setInputDim(kern.getInputDim());
    variance = kern.variance;
    inverseWidth = kern.inverseWidth;
}
// Class destructor
CRbfKern::~CRbfKern()
{
}
double CRbfKern::getVariance() const
{
    return variance;
}
void CRbfKern::_init()
{
    nParams = 2;
    setType("rbf");
    setName("RBF");
    setParamName("inverseWidth", 0);
    addTransform(CTransform::defaultPositive(), 0);
    setParamName("variance", 1);
    addTransform(CTransform::defaultPositive(), 1);
    setStationary(true);
}
void CRbfKern::setInitParam()
{
    inverseWidth = 1.0;
    variance = 1.0;
}

inline double CRbfKern::diagComputeElement(const CMatrix& X, unsigned int index) const
{
    return variance;
}
void CRbfKern::diagCompute(CMatrix& d, const CMatrix& X) const
{
    DIMENSIONMATCH(d.getCols()==1);
    DIMENSIONMATCH(X.rowsMatch(d));
    d.setVals(variance);
}
// Parameters are kernel parameters
void CRbfKern::setParam(double val, unsigned int paramNo)
{
    BOUNDCHECK(paramNo < nParams);
    switch(paramNo)
    {
        case 0:
            inverseWidth = val;
            break;
        case 1:
            variance = val;
            break;
        default:
            throw ndlexceptions::Error("Requested parameter doesn't exist.");
    }
}
double CRbfKern::getParam(unsigned int paramNo) const
{
    BOUNDCHECK(paramNo < nParams);
    switch(paramNo)
    {
        case 0:
            return inverseWidth;
            break;
        case 1:
            return variance;
            break;
        default:
            throw ndlexceptions::Error("Requested parameter doesn't exist.");
    }
}
void CRbfKern::getGradX(CMatrix& gX, const CMatrix& X, unsigned int row, const CMatrix& X2, bool addG) const
{
    DIMENSIONMATCH(gX.getRows() == X2.getRows());
    BOUNDCHECK(row < X.getRows());
    DIMENSIONMATCH(X.getCols()==X2.getCols());
    double wi2 = 0.5*inverseWidth;
    double pf = variance*inverseWidth;
    DIMENSIONMATCH(gX.getCols()==X2.getCols());
    for(unsigned int k=0; k<X2.getRows(); k++)
    {
        double n2 = X.dist2Row(row, X2, k);
        for(unsigned int j=0; j<X2.getCols(); j++)
        {
            double val = pf*(X2.getVal(k, j)-X.getVal(row, j))*exp(-n2*wi2);
            if(addG)
                gX.addVal(val, k, j);
            else
                gX.setVal(val, k, j);
        }
    }
}
void CRbfKern::getDiagGradX(CMatrix& gX, const CMatrix& X, bool addG) const
{
    DIMENSIONMATCH(gX.dimensionsMatch(X));
    if(!addG)
        gX.zeros();
}
double CRbfKern::getWhite() const
{
    return 0.0;
}

double CRbfKern::computeElement(const CMatrix& X1, unsigned int index1,
                                const CMatrix& X2, unsigned int index2) const
{
    double k = X1.dist2Row(index1, X2, index2);
    k = 0.5*k*inverseWidth;
    k = variance*exp(-k);
    return k;
}

void CRbfKern::updateX(const CMatrix& X)
{
    setUpdateXused(true);
    Xdists.resize(X.getRows(),X.getRows());
    double halfInverseWidth=0.5*inverseWidth;
    unsigned int nrows = X.getRows();
    for(unsigned int j=0; j<nrows; j++)
    {
        Xdists.setVal(0,j,j);
        for(unsigned int i=0; i<j; i++)
        {
            double dist2 = X.dist2Row(i, X, j);
            Xdists.setVal(dist2,i,j);
            Xdists.setVal(exp(-dist2*halfInverseWidth),j,i);
        }
    }
}
void CRbfKern::getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& X2, const CMatrix& covGrad, bool regularise) const
{
    DIMENSIONMATCH(g.getRows()==1);
    DIMENSIONMATCH(g.getCols()==nParams);
    DIMENSIONMATCH(X.getRows()==covGrad.getRows());
    DIMENSIONMATCH(X2.getRows()==covGrad.getCols());
    double g1=0.0;
    double g2=0.0;
    double halfInverseWidth=0.5*inverseWidth;
    double halfVariance=0.5*variance;

    unsigned int nrows = X.getRows();
    for(unsigned int j=0; j<nrows; j++)
    {
        for(unsigned int i=0; i<X2.getRows(); i++)
        {
            double k = 0;
            double dist2 = 0;
            dist2 = X2.dist2Row(i, X, j);
            k = exp(-dist2*halfInverseWidth);
            double kcg_ij = k*covGrad.getVal(j,i);
            g1 -= halfVariance*dist2*kcg_ij; // dk()/dgamma in SBIK paper
            g2 += kcg_ij;                    // dk()/dalpha in SBIK paper
        }
    }
    g.setVal(g1, 0);
    g.setVal(g2, 1);
    if(regularise)
        addPriorGrad(g);
}

void CRbfKern::getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& covGrad, bool regularise) const
{
    DIMENSIONMATCH(g.getRows()==1);
    DIMENSIONMATCH(g.getCols()==nParams);
    MATRIXPROPERTIES(covGrad.isSymmetric());
    double g1=0.0;
    double g2=0.0;
    double halfInverseWidth=0.5*inverseWidth;
    double halfVariance=0.5*variance;

    unsigned int nrows = X.getRows();
    for(unsigned int j=0; j<nrows; j++)
    {
        g2 += covGrad.getVal(j,j);
        for(unsigned int i=0; i<j; i++)
        {
            double k = 0;
            double dist2 = 0;
            if(isUpdateXused()) // WVB's mod for precomputing parts of the kernel.
            {
                dist2 = Xdists.getVal(i,j);
                k = Xdists.getVal(j,i);
            }
            else
            {
                dist2 = X.dist2Row(i, X, j);
                k = exp(-dist2*halfInverseWidth);
            }
            double kcg_ij = k*covGrad.getVal(i,j);
            g1 -= 2.0*halfVariance*dist2*kcg_ij; // dk()/dgamma in SBIK paper
            g2 += 2.0*kcg_ij;                    // dk()/dalpha in SBIK paper
        }
    }
    g.setVal(g1, 0);
    g.setVal(g2, 1);
    if(regularise)
        addPriorGrad(g);
}

double CRbfKern::getGradParam(unsigned int index, const CMatrix& X, const CMatrix& X2, const CMatrix& covGrad) const
{

    BOUNDCHECK(index<nParams);
    throw ndlexceptions::NotImplementedError( "Error getGradParam is not currently implemented for CRbfKern");

}
double CRbfKern::getGradParam(unsigned int index, const CMatrix& X, const CMatrix& covGrad) const
{

    BOUNDCHECK(index<nParams);
    throw ndlexceptions::NotImplementedError( "Error getGradParam is not currently implemented for CRbfKern");

}


