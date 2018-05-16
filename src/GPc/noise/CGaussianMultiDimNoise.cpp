#include "noise/CGaussianMultiDimNoise.h"

CGaussianMultiDimNoise::~CGaussianMultiDimNoise() {
}

void CGaussianMultiDimNoise::initStoreage() {
    setNumParams(getOutputDim() * 2);
    mu.resize(getNumData(), getOutputDim());
    varSigma.resize(getNumData(), getOutputDim());
    bias.resize(1, getOutputDim());
    sigma2.resize(getOutputDim(), 1);
    clearTransforms();
    // transform sigma2 (the last parameter).
    addTransform(CTransform::defaultPositive(), getNumParams() - 1);
    setLogConcave(true);
    setSpherical(true);
    setMissing(false);

}

void CGaussianMultiDimNoise::_init() {
    setType("gaussian");
    setName("Gaussian");
}

void CGaussianMultiDimNoise::initNames() {
    for (unsigned int j = 0; j < getOutputDim(); j++) {
        setParamName("bias" + ndlstrutil::itoa(j), j);
        setParamName("sigma2" + ndlstrutil::itoa(j), j);
    }
}

void CGaussianMultiDimNoise::initParams() {
    bias.deepCopy(meanCol(*py));
}

void CGaussianMultiDimNoise::initVals() {
    mu.zeros();
    varSigma.zeros();
    bias.zeros();
    sigma2.zeros();
}

ostream &CGaussianMultiDimNoise::display(ostream &os) {
    double b = 0.0;
    os << "Gaussian Noise: " << endl;
    for (unsigned int j = 0; j < bias.getCols(); j++) {
        b = bias.getVal(j);
        os << "Bias on process " << j << ": " << b << endl;
    }
    os << "Variance: " << sigma2 << endl;
    return os;
}

void CGaussianMultiDimNoise::setParam(double val, unsigned int index) {
    BOUNDCHECK(index >= 0);
    BOUNDCHECK(index < getNumParams());
    if (index < getOutputDim()) {
        bias.setVal(val, index);
    } else {
        sigma2 = val;
    }
}

void CGaussianMultiDimNoise::setParams(const CMatrix &params) {
    DIMENSIONMATCH(getNumParams() == getOutputDim() + 1);
    DIMENSIONMATCH(params.getCols() == getNumParams());
    DIMENSIONMATCH(params.getRows() == 1);
    DIMENSIONMATCH(getOutputDim() == bias.getCols());
    for (unsigned int j = 0; j < bias.getCols(); j++) {
        bias.setVal(params.getVal(j), j);
    }
    sigma2 = params.getVal(getNumParams() - 1);
}

double CGaussianMultiDimNoise::getParam(unsigned int index) const {
    BOUNDCHECK(index >= 0);
    BOUNDCHECK(index < getNumParams());
    if (index < getOutputDim()) {
        return bias.getVal(index);
    } else {
        return sigma2.getVal(index - bias.getRows());
    }

}

void CGaussianMultiDimNoise::getParams(CMatrix &params) const {
    DIMENSIONMATCH(getNumParams() == getOutputDim() + 1);
    DIMENSIONMATCH(params.getCols() == getNumParams());
    DIMENSIONMATCH(params.getRows() == 1);
    DIMENSIONMATCH(getOutputDim() == bias.getCols());
    for (unsigned int j = 0; j < getOutputDim(); j++) {
        params.setVal(bias.getVal(j), j);
        params.setVal(sigma2.getVal(j), getNumParams() - bias.getRows());
    }
}

void CGaussianMultiDimNoise::getGradParams(CMatrix &g) const {
    DIMENSIONMATCH(g.getCols() == getNumParams());
    DIMENSIONMATCH(g.getRows() == 1);
    double            nu      = 0.0;
    double            u       = 0.0;
    double            gsigma2 = 0.0;
    double            b       = 0.0;
    for (unsigned int j       = 0; j < getOutputDim(); j++) {
        double gbias = 0.0;
        b = bias.getVal(j);
        for (unsigned int i = 0; i < getNumData(); i++) {
            nu = 1 / (getVarSigma(i, j) + sigma2.getVal(j));
            u  = getTarget(i, j) - getMu(i, j) - b;
            u *= nu;
            gbias += u;
            gsigma2 += nu - u * u;
        }
        g.setVal(gbias, 0, j);
    }
    g.setVal(-0.5 * gsigma2, 0, getNumParams() - 1);
}

void CGaussianMultiDimNoise::getGradInputs(double &gmu, double &gvs, unsigned int i, unsigned int j) const {
    gmu = -bias.getVal(j);
    gvs = 1 / (sigma2.getVal(j) + getVarSigma(i, j));
    gmu += getTarget(i, j) - getMu(i, j);
    gmu *= gvs;
    gvs = 0.5 * (gmu * gmu - gvs);
}

void CGaussianMultiDimNoise::getNuG(CMatrix &g, CMatrix &nu, unsigned int index) const {
    double            nuval = 0.0;
    double            gval  = 0.0;
    for (unsigned int j     = 0; j < getOutputDim(); j++) {
        nuval = 1. / (sigma2.getVal(j) + getVarSigma(index, j));
        if (isnan(nuval)) {
            cout << "Sigma2 " << sigma2 << endl;
            cout << "varSigma " << getVarSigma(index, j) << endl;
        }
        SANITYCHECK(!isnan(nuval));
        nu.setVal(nuval, index, j);
        gval = getTarget(index, j) - getMu(index, j) - bias.getVal(j);
        g.setVal(gval * nuval, index, j);
    }
}

void CGaussianMultiDimNoise::updateSites(CMatrix &m, CMatrix &beta, unsigned int actIndex,
                                         const CMatrix &g, const CMatrix &nu,
                                         unsigned int index) const {
    for (unsigned int j = 0; j < getOutputDim(); j++) {
        beta.setVal(1 / sigma2.getVal(j), actIndex, j);
        m.setVal(getTarget(index, j) - bias.getVal(j), actIndex, j);
    }
}

void CGaussianMultiDimNoise::test(const CMatrix &muout, const CMatrix &varSigmaOut, const CMatrix &yTest) const {
    DIMENSIONMATCH(yTest.dimensionsMatch(muout));
    DIMENSIONMATCH(muout.dimensionsMatch(varSigmaOut));
    DIMENSIONMATCH(yTest.getCols() == getOutputDim());
    CMatrix yPred(yTest.getRows(), yTest.getCols());
    out(yPred, muout, varSigmaOut);
    for (unsigned int i = 0; i < getOutputDim(); i++)
        cout << "Mean Squared Error on output " << i + 1 << ": "
             << yPred.dist2Col(i, yTest, i) / (double) yTest.getRows() << endl;
}

void CGaussianMultiDimNoise::out(CMatrix &yPred, const CMatrix &muTest, const CMatrix &varSigmaTest) const {
    DIMENSIONMATCH(yPred.dimensionsMatch(muTest));
    DIMENSIONMATCH(muTest.dimensionsMatch(varSigmaTest));
    yPred.deepCopy(muTest);
    for (unsigned int j = 0; j < getOutputDim(); j++)
        yPred.addCol(j, bias.getVal(j));
}

void CGaussianMultiDimNoise::out(CMatrix &yPred,
                                 CMatrix &stdDeviations,
                                 const CMatrix &muTest,
                                 const CMatrix &varSigmaTest) const {
    DIMENSIONMATCH(yPred.dimensionsMatch(stdDeviations));
    out(yPred, muTest, varSigmaTest);
    for (unsigned int i = 0; i < stdDeviations.getRows(); i++)
        for (unsigned int j = 0; j < stdDeviations.getCols(); j++)
            stdDeviations.setVal(sqrt(varSigmaTest.getVal(i, j) + sigma2.getVal(j)), i, j);
}

void CGaussianMultiDimNoise::likelihoods(CMatrix &L, const CMatrix &muTest, const CMatrix &varSigmaTest,
                                         const CMatrix &yTest) const {
    DIMENSIONMATCH(yTest.getCols() == getOutputDim());
    DIMENSIONMATCH(L.dimensionsMatch(muTest));
    DIMENSIONMATCH(yTest.dimensionsMatch(muTest));
    DIMENSIONMATCH(muTest.dimensionsMatch(varSigmaTest));
    double            arg = 0.0;
    double            var = 0.0;
    for (unsigned int i   = 0; i < muTest.getRows(); i++) {
        for (unsigned int j = 0; j < muTest.getCols(); j++) {
            arg = yTest.getVal(i, j) - muTest.getVal(i, j) - bias.getVal(j);
            arg *= arg;
            var = varSigmaTest.getVal(i, j) + sigma2.getVal(j);
            arg = 1 / sqrt(2 * M_PI * var) * exp(-.5 * arg * arg / var);
            L.setVal(arg, i, j);
        }
    }
}

double CGaussianMultiDimNoise::logLikelihood(const CMatrix &muTest, const CMatrix &varSigmaTest,
                                             const CMatrix &yTest) const {
    DIMENSIONMATCH(yTest.getCols() == getOutputDim());
    DIMENSIONMATCH(yTest.dimensionsMatch(muTest));
    DIMENSIONMATCH(yTest.dimensionsMatch(varSigmaTest));
    double            arg = 0.0;
    double            var = 0.0;
    double            L   = 0.0;
    for (unsigned int i   = 0; i < muTest.getRows(); i++) {
        for (unsigned int j = 0; j < muTest.getCols(); j++) {
            arg = yTest.getVal(i, j) - muTest.getVal(i, j) - bias.getVal(j);
            arg *= arg;
            var = varSigmaTest.getVal(i, j) + sigma2.getVal(j);
            arg = arg / var;
            L += log(var) + arg;
        }
    }
    L += muTest.getRows() * muTest.getCols() * log(2 * M_PI);
    L *= -0.5;
    return L;
}

void CGaussianMultiDimNoise::addParamToMxArray(mxArray *matlabArray) const {
    mxAddField(matlabArray, "nParams");
    mxSetField(matlabArray, 0, "nParams", convertMxArray((double) getNumParams()));
    mxAddField(matlabArray, "bias");
    mxSetField(matlabArray, 0, "bias", bias.toMxArray());
    mxAddField(matlabArray, "sigma2");
    mxSetField(matlabArray, 0, "sigma2", sigma2.toMxArray());
}

void CGaussianMultiDimNoise::extractParamFromMxArray(const mxArray *matlabArray) {
    setNumParams(mxArrayExtractIntField(matlabArray, "nParams"));
    mxArray *biasField = mxArrayExtractMxArrayField(matlabArray, "bias");
    bias.fromMxArray(biasField);
    sigma2 = mxArrayExtractDoubleField(matlabArray, "sigma2");

}