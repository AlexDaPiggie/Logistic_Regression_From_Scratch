#include "lr_model.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(lr_model_cpp, m) {
    m.doc() = "C++ logistic regression bindings";

    py::class_<LogisticRegression>(m, "LogisticRegression")
        .def(
            py::init<int, double, double, double, double>(),
            py::arg("num_iter") = 1000,
            py::arg("alpha") = 0.01,
            py::arg("tol") = 1e-8,
            py::arg("threshold") = 0.5,
            py::arg("lambda_reg") = 0.0
        )
        .def("fit", &LogisticRegression::fit, py::return_value_policy::copy)
        .def("predict_proba", &LogisticRegression::predict_proba)
        .def("predict", &LogisticRegression::predict)
        .def("get_theta", &LogisticRegression::get_theta, py::return_value_policy::copy);
}
