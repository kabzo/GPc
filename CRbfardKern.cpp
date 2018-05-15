//
// Created by juraj on 13/5/18.
//

#include "CRbfardKern.hpp"

// the RBF ARD kernel.
CRbfardKern::CRbfardKern() : CArdKern()
{
    _init();
}
CRbfardKern::CRbfardKern(unsigned int inDim) : CArdKern(inDim)
{
    _init();
    setInputDim(inDim);
}
CRbfardKern::CRbfardKern(const CMatrix& X) : CArdKern(X)
{
    _init();
    setInputDim(X.getCols());
}
CRbfardKern::CRbfardKern(const CRbfardKern& kern) : CArdKern(kern)
{
    _init();
    setInputDim(kern.getInputDim());
    variance = kern.variance;
    inverseWidth = kern.inverseWidth;
    scales = kern.scales;
}
// Class destructor
CRbfardKern::~CRbfardKern()
{
}
double CRbfardKern::getVariance() const
{
    return variance;
}
void CRbfardKern::_init()
{
    nParams = 2;
    setType("rbfard");
    setName("RBF ARD");
    setParamName("inverseWidth", 0);
    addTransform(CTransform::defaultPositive(), 0);
    setParamName("variance", 1);
    addTransform(CTransform::defaultPositive(), 1);
    setStationary(true);
}
void CRbfardKern::setInitParam()
{
    nParams = 2+getInputDim();
    inverseWidth=1.0;
    variance = 1.0;

    // input scales.
    scales.resize(1, getInputDim());
    gscales.resize(1, getInputDim());
    scales.setVals(0.5);
    for(unsigned int i=2; i<getInputDim()+2; i++)
    {
        string name = "inputScale";
        setParamName(name, i);
    }
    for(unsigned int i=2; i<getInputDim()+2; i++)
    {
        addTransform(CTransform::defaultZeroOne(), i);
    }

}

double CRbfardKern::diagComputeElement(const CMatrix& X, unsigned int index1) const
{
    return variance;
}
// Parameters are kernel parameters
void CRbfardKern::setParam(double val, unsigned int paramNo)
{

    BOUNDCHECK(paramNo<nParams);
    switch(paramNo)
    {
        case 0:
            inverseWidth=val;
            break;
        case 1:
            variance=val;
            break;
        default:
            if(paramNo<nParams)
                scales.setVal(val, paramNo-2);
            else
            {
                throw ndlexceptions::Error("Requested parameter doesn't exist.");
            }
    }
}
double CRbfardKern::getParam(unsigned int paramNo) const
{

    BOUNDCHECK(paramNo<nParams);
    switch(paramNo)
    {
        case 0:
            return inverseWidth;
            break;
        case 1:
            return variance;
            break;
        default:
            if(paramNo<nParams)
                return scales.getVal(paramNo-2);
            else
            {
                throw ndlexceptions::Error("Requested parameter doesn't exist.");
            }
    }
}
void CRbfardKern::getGradX(CMatrix& gX, const CMatrix& X, unsigned int row, const CMatrix& X2, bool addG) const
{
    DIMENSIONMATCH(gX.getRows() == X2.getRows());
    BOUNDCHECK(row < X.getRows());
    DIMENSIONMATCH(X.getCols()==X2.getCols());
    double wi2 = 0.5*inverseWidth;
    double pf = variance*inverseWidth;
    for(unsigned int k=0; k<X2.getRows(); k++)
    {
        double n2=0.0;
        for(unsigned int j=0; j<getInputDim(); j++)
        {
            double x = X.getVal(row, j);
            x = x-X2.getVal(k, j);
            n2+=x*scales.getVal(j)*x;
        }
        for(unsigned int j=0; j<X2.getCols(); j++)
        {
            double val = 0.0;
            if(addG)
                val = gX.getVal(k, j);
            val += pf*(X2.getVal(k, j)-X.getVal(row, j))*exp(-n2*wi2)*scales.getVal(j);
            gX.setVal(val, k, j);
        }
    }
}
void CRbfardKern::getDiagGradX(CMatrix& gX, const CMatrix& X, bool addG) const
{
    DIMENSIONMATCH(gX.dimensionsMatch(X));
    if(!addG)
        gX.zeros();
}
double CRbfardKern::getWhite() const
{
    return 0.0;
}

double CRbfardKern::computeElement(const CMatrix& X1, unsigned int index1,
                                   const CMatrix& X2, unsigned int index2) const
{
    double val = 0.0;
    for(unsigned int i=0; i<getInputDim(); i++)
    {
        double x = X1.getVal(index1, i);
        x = x-X2.getVal(index2, i);
        val+=x*scales.getVal(i)*x;
    }
    return variance*exp(-val*inverseWidth*0.5);
}

void CRbfardKern::getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& X2, const CMatrix& covGrad, bool regularise) const
{
    DIMENSIONMATCH(g.getRows()==1);
    DIMENSIONMATCH(g.getCols()==nParams);
    double g1=0.0;
    double g2=0.0;
    gscales.zeros();
    double halfInverseWidth = 0.5*inverseWidth;
    for(unsigned int i=0; i<X.getRows(); i++)
    {
        for(unsigned int j=0; j<X2.getRows(); j++)
        {
            double val = 0.0;
            for(unsigned int k=0; k<getInputDim(); k++)
            {
                double x = X.getVal(i, k);
                x-=X2.getVal(j, k);
                val+=x*scales.getVal(k)*x;
            }
            double kCovGrad = exp(-halfInverseWidth*val)*covGrad.getVal(i, j);
            g1-=0.5*val*kCovGrad*variance;
            g2+=kCovGrad;
            for(unsigned int k=0; k<getInputDim(); k++)
            {
                double g3=gscales.getVal(k);
                double xi=X.getVal(i, k);
                double xj=X2.getVal(j, k);
                g3+=inverseWidth*kCovGrad*(xi*xj-.5*xi*xi-.5*xj*xj)*variance;
                gscales.setVal(g3, k);
            }
        }
    }
    g.setVal(g1, 0);
    g.setVal(g2, 1);
    for(unsigned int k=0; k<getInputDim(); k++)
        g.setVal(gscales.getVal(k), k+2);
    if(regularise)
        addPriorGrad(g);

}

void CRbfardKern::getGradParams(CMatrix& g, const CMatrix& X, const CMatrix& covGrad, bool regularise) const
{
    DIMENSIONMATCH(g.getRows()==1);
    DIMENSIONMATCH(g.getCols()==nParams);
    double g1=0.0;
    double g2=0.0;
    gscales.zeros();
    double halfInverseWidth = 0.5*inverseWidth;
    for(unsigned int i=0; i<X.getRows(); i++)
    {
        for(unsigned int j=0; j<i; j++)
        {
            double val = 0.0;
            for(unsigned int k=0; k<getInputDim(); k++)
            {
                double x = X.getVal(i, k);
                x-=X.getVal(j, k);
                val+=x*scales.getVal(k)*x;
            }
            double kCovGrad = exp(-halfInverseWidth*val)*covGrad.getVal(i, j);
            g1-=0.5*val*kCovGrad*variance;
            g2+=kCovGrad;
            for(unsigned int k=0; k<getInputDim(); k++)
            {
                double g3=gscales.getVal(k);
                double xi=X.getVal(i, k);
                double xj=X.getVal(j, k);
                g3+=inverseWidth*kCovGrad*(xi*xj-.5*xi*xi-.5*xj*xj)*variance;
                gscales.setVal(g3, k);
            }
        }
    }
    g1*=2.0;
    g2*=2.0;
    for(unsigned int i=0; i<X.getRows(); i++)
        g2+=covGrad.getVal(i, i);
    gscales.scale(2.0);
    g.setVal(g1, 0);
    g.setVal(g2, 1);
    for(unsigned int k=0; k<getInputDim(); k++)
        g.setVal(gscales.getVal(k), k+2);
    if(regularise)
        addPriorGrad(g);

}
double CRbfardKern::getGradParam(unsigned int index, const CMatrix& X, const CMatrix& X2, const CMatrix& covGrad) const
{

    BOUNDCHECK(index<nParams);
    throw ndlexceptions::NotImplementedError( "Error getGradParam is not currently implemented for CRbfardKern");

}
double CRbfardKern::getGradParam(unsigned int index, const CMatrix& X, const CMatrix& covGrad) const
{

    BOUNDCHECK(index<nParams);
    throw ndlexceptions::NotImplementedError( "Error getGradParam is not currently implemented for CRbfardKern");

}