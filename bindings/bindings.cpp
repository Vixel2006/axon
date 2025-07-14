#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "tensor.h"

namespace py = pybind11;

PYBIND11_MODULE(nawah, m)
{
    py::enum_<DType>(m, "DType")
        .value("float16", DType::float16)
        .value("float32", DType::float32)
        .value("int8", DType::int8)
        .value("int32", DType::int32)
        .value("uint8", DType::uint8)
        .export_values();

    py::enum_<DeviceType>(m, "DeviceType")
        .value("CPU", DeviceType::CPU)
        .value("CUDA", DeviceType::CUDA)
        .export_values();

    ;

    py::class_<Device>(m, "Device")
        .def(py::init<DeviceType, int>(), py::arg("type") = DeviceType::CPU, py::arg("index") = 0)
        .def_readwrite("type", &Device::type)
        .def_readwrite("index", &Device::index)
        .def("__eq__", &Device::operator==)
        .def("__repr__", [](const Device &d)
             { return "<Device type=" + std::string(d.type == DeviceType::CPU ? "CPU" : "CUDA") +
                      " index=" + std::to_string(d.index) + ">"; });

    py::class_<Tensor>(m, "Tensor")
        .def(py::init<const std::vector<int64_t> &, DType, Device>(),
             py::arg("shape"), py::arg("dtype"), py::arg("device"))

        .def_property_readonly("shape", &Tensor::shape, py::return_value_policy::reference)
        .def_property_readonly("strides", &Tensor::strides, py::return_value_policy::reference)
        .def_property_readonly("dtype", &Tensor::dtype, py::return_value_policy::reference)
        .def_property_readonly("device", &Tensor::device, py::return_value_policy::reference)

        .def("numel", &Tensor::numel)
        .def("is_contiguous", &Tensor::is_contiguous)
        .def("view", &Tensor::view, py::arg("shape"))
        .def("squeeze", &Tensor::squeeze, py::arg("dim") = 0)
        .def("unsqueeze", &Tensor::unsqueeze, py::arg("dim") = 0)
        .def("permute", &Tensor::permute, py::arg("order"))
        .def("transpose", &Tensor::transpose, py::arg("n"), py::arg("m"))
        .def("expand", &Tensor::expand, py::arg("shape"))
        .def("broadcast", &Tensor::broadcast, py::arg("shape"))
        .def("flatten", &Tensor::flatten, py::arg("start") = 1, py::arg("end") = -1)
        .def("__getitem__", [](const Tensor &t, py::object obj)
             {
            std::vector<std::shared_ptr<IndexStrategy>> strategies;

            if (py::isinstance<py::tuple>(obj)) {
                auto tuple = obj.cast<py::tuple>();
                for (auto item : tuple) {
                    if (py::isinstance<py::int_>(item)) {
                        strategies.push_back(std::make_shared<IntegerIndex>(item.cast<int64_t>()));
                    } else if (py::isinstance<py::slice>(item)) {
                        py::slice s = item.cast<py::slice>();
                        int64_t start, stop, step, length;
                        s.compute(t.shape().size(), &start, &stop, &step, &length);
                        strategies.push_back(std::make_shared<SliceIndex>(start, stop, step));
                    } else {
                        throw std::runtime_error("Unsupported index type");
                    }
                }
            }
            else if (py::isinstance<py::int_>(obj)) {
                strategies.push_back(std::make_shared<IntegerIndex>(obj.cast<int64_t>()));
            } else if (py::isinstance<py::slice>(obj)) {
                py::slice s = obj.cast<py::slice>();
                int64_t start, stop, step, length;
                s.compute(t.shape().size(), &start, &stop, &step, &length);
                strategies.push_back(std::make_shared<SliceIndex>(start, stop, step));
            }

            return t.get_item(strategies); });
}
