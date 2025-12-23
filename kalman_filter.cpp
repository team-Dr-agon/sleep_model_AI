// kalman_filter.cpp
#include <pybind11/pybind11.h>

namespace py = pybind11;

class KalmanFilter {
public:
    KalmanFilter(float X0, float P0, float Q, float R)
        : X(X0), P(P0), Q_(Q), R_(R) {}

    void predict() {
        P = P + Q_;
    }

    // 元コードの kalma(z) と同じ（updateに改名）
    float update(float z) {
        float K = P / (P + R_);
        X = X + K * (z - X);
        P = (1.0f - K) * P;
        return X;
    }

    float getState() const { return X; }
    float getCovariance() const { return P; }

    void setProcessNoise(float Q) { Q_ = Q; }
    void setSensorNoise(float R) { R_ = R; }

private:
    float X;   // 状態推定
    float P;   // 推定誤差共分散
    float Q_;  // プロセスノイズ
    float R_;  // センサノイズ
};

// 2センサ融合（元コードの x_fused）も関数として出しておく（将来使える）
float fuse_by_precision(float xA, float pA, float xB, float pB) {
    // (xA/pA + xB/pB) / (1/pA + 1/pB)
    return (xA / pA + xB / pB) / (1.0f / pA + 1.0f / pB);
}

PYBIND11_MODULE(kalman_filter, m) {
    m.doc() = "1D Kalman Filter (pybind11)";

    py::class_<KalmanFilter>(m, "KalmanFilter")
        .def(py::init<float, float, float, float>(),
             py::arg("X0"), py::arg("P0"), py::arg("Q"), py::arg("R"))
        .def("predict", &KalmanFilter::predict)
        .def("update", &KalmanFilter::update, py::arg("z"))
        .def("getState", &KalmanFilter::getState)
        .def("getCovariance", &KalmanFilter::getCovariance)
        .def("setProcessNoise", &KalmanFilter::setProcessNoise, py::arg("Q"))
        .def("setSensorNoise", &KalmanFilter::setSensorNoise, py::arg("R"));

    m.def("fuse_by_precision", &fuse_by_precision,
          py::arg("xA"), py::arg("pA"), py::arg("xB"), py::arg("pB"),
          "Fuse two estimates by inverse covariance weighting");
}
