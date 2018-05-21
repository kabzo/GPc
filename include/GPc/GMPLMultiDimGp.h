//
// Created by juraj on 16/5/18.
//

#ifndef CMATRIX_CMULTIDIMGP_H
#define CMATRIX_CMULTIDIMGP_H

#include <utility>

#include "CGp.h"
#include "CKernTypes.hpp"

class GMPLMultiDimGp : public CMatInterface {
 public:

    struct GPMLKernConfig : public CMatInterface {
        GPMLKernConfig() = default;
        explicit GPMLKernConfig(const mxArray *array) { fromMxArray(array); }

        CMatrix scales;
        CMatrix xt;
        double  variance;
        void fromMxArray(const mxArray *array) override {
            xt.fromMxArray(mxArrayExtractMxArrayField(array, "xt"));
            scales.fromMxArray(mxArrayExtractMxArrayField(array, "scales"));
            variance = mxArrayExtractDoubleField(array, "variance");
        }
        mxArray *toMxArray() const override { return nullptr; }
    };

    struct GPMLNoiseConfig : public CMatInterface {
        GPMLNoiseConfig() = default;
        explicit GPMLNoiseConfig(const mxArray *array) { fromMxArray(array); }

        double  lik;
        CMatrix yt;
        void fromMxArray(const mxArray *array) override {

            yt.fromMxArray(mxArrayExtractMxArrayField(array, "yt"));
            lik = mxArrayExtractDoubleField(array, "lik");
        }
        mxArray *toMxArray() const override { return nullptr; }
    };

    struct GPMLConfig : public CMatInterface {
        GPMLConfig() = default;
        GPMLConfig(const mxArray *array) { fromMxArray(array); }

        CMatrix Alpha;
        CMatrix sW;
        CMatrix L;

        void fromMxArray(const mxArray *array) override {
            L.fromMxArray(mxArrayExtractMxArrayField(array, "L"));
            L.setTriangular(true);
            Alpha.fromMxArray(mxArrayExtractMxArrayField(array, "alpha"));
            sW.fromMxArray(mxArrayExtractMxArrayField(array, "sW"));
        }

        mxArray *toMxArray() const override { return nullptr; }
    };

    struct Result : public CMatInterface {
        Result() = default;

        CMatrix mu;
        CMatrix var;

        void fromMxArray(const mxArray *array) override {
            mu.fromMxArray(mxArrayExtractMxArrayField(array, "mu"));
            var.fromMxArray(mxArrayExtractMxArrayField(array, "var"));
        }
        mxArray *toMxArray() const override { return nullptr; }

    };

    explicit GMPLMultiDimGp() : numOut(0), set_(false) {}

    GMPLMultiDimGp(const std::vector<GPMLKernConfig> &kern,
                   const std::vector<GPMLNoiseConfig> &noise,
                   const std::vector<GPMLConfig> &config) {
        DIMENSIONMATCH(kern.size() == noise.size());
        DIMENSIONMATCH(kern.size() == config.size());
        numOut = static_cast<unsigned int>(kern.size());
        init();

        for (unsigned int idx = 0; idx < kern.size(); idx++) {
            createNoise(idx, noise[idx]);
            createKern(idx, kern[idx]);
            createConfig(idx, config[idx]);
        }
        set_ = true;
    }

    void fromMxArray(const mxArray *array) override {
        numOut = static_cast<unsigned int>(mxArrayExtractDoubleField(array, "nOut"));
        init();
        {
            mxArray           *arrayStruct = mxArrayExtractMxArrayField(array, "noise");
            for (unsigned int i            = 0; i < numOut; ++i) {
                const auto compElement = mxGetCell(arrayStruct, i);
                createNoise(i, GPMLNoiseConfig(compElement));
            }
        }
        {
            mxArray           *arrayStruct = mxArrayExtractMxArrayField(array, "hyp");
            for (unsigned int i            = 0; i < numOut; ++i) {
                const auto compElement = mxGetCell(arrayStruct, i);
                createKern(i, GPMLKernConfig(compElement));
            }
        }
        {
            mxArray           *arrayStruct = mxArrayExtractMxArrayField(array, "post");
            for (unsigned int i            = 0; i < numOut; ++i) {
                const auto compElement = mxGetCell(arrayStruct, i);
                createConfig(i, GPMLConfig(compElement));
            }
        }
        {
            mxArray           *arrayStruct = mxArrayExtractMxArrayField(array, "res");
            for (unsigned int i            = 0; i < numOut; ++i) {
                const auto compElement = mxGetCell(arrayStruct, i);
                _gp_results[i].fromMxArray(compElement);
            }
        }
    }

    void createKern(unsigned int idx, const GPMLKernConfig &conf) {
        _kerns[idx].first  = conf;
        _kerns[idx].second = CRbfardKern(_kerns[idx].first.xt);
        _kerns[idx].second.setParamName("inverseWidth", 1);
        _kerns[idx].second.setVariance(_kerns[idx].first.variance);
        _kerns[idx].second.setScales(_kerns[idx].first.scales);
    }

    void createNoise(unsigned int idx, const GPMLNoiseConfig conf) {
        _noises[idx].first  = conf;
        _noises[idx].second = CGaussianNoise(&_noises[idx].first.yt);
        _noises[idx].second.setParam(0.0, 0);
        _noises[idx].second.setParam(conf.lik, 1);
    }

    void createConfig(unsigned int idx, const GPMLConfig &conf) {
        _gp[idx].first = conf;
        _gp[idx].second =
            CGp(&_kerns[idx].second, &_noises[idx].second, &_kerns[idx].first.xt, &_gp[idx].first.Alpha,
                &_gp[idx].first.L, &_gp[idx].first.sW);
    }

    void init() {
        _kerns  = std::vector<std::pair<GPMLKernConfig, CRbfardKern>>(numOut);
        _noises = std::vector<std::pair<GPMLNoiseConfig, CGaussianNoise>>(numOut);
        _gp     = std::vector<std::pair<GPMLConfig, CGp>>(numOut);

        _gp_results = std::vector<Result>(numOut);
    }

    mxArray *toMxArray() const override { return nullptr; }

    void out(CMatrix &mu, CMatrix &var, const CMatrix &xin) {
        const unsigned int nrows = xin.getRows();
        CMatrix            mu_i(nrows, 1), var_i(nrows, 1);

        for (unsigned int  i     = 0; i < numOut; ++i) {
            clock_t begin = std::clock();
            _gp[i].second.out(mu_i, var_i, xin, true);
            clock_t end = std::clock();
            double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
            std::cout << elapsed_secs << std::endl;
            mu.setMatrix(0, i, mu_i);
            var.setMatrix(0, i, var_i);
        }
    }

    void check(const CMatrix &xin) {
        const unsigned int nrows = xin.getRows();
        CMatrix            mu_i(nrows, 1), var_i(nrows, 1);
        for (unsigned int  i     = 0; i < numOut; ++i) {
            _gp[i].second.out(mu_i, var_i, xin, true);
            std::cout << _gp_results[i].mu.equals(mu_i, 1e-3) << std::endl;
            std::cout << _gp_results[i].var.equals(var_i, 1e-3) << std::endl;
        }
    }

    bool isSet() const { return set_; }

 private:
    unsigned int                                            numOut;
    std::vector<std::pair<GPMLKernConfig, CRbfardKern>>     _kerns;
    std::vector<std::pair<GPMLNoiseConfig, CGaussianNoise>> _noises;
    std::vector<std::pair<GPMLConfig, CGp >>                _gp;

    std::vector<Result> _gp_results;

    bool set_;

};

#endif //CMATRIX_CMULTIDIMGP_H
