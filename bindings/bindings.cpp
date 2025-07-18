#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "tensor.h"
#include "helpers.h"

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

    py::class_<Device>(m, "Device")
        .def(py::init<DeviceType, int>(), py::arg("type") = DeviceType::CPU, py::arg("index") = 0)
        .def_readwrite("type", &Device::type)
        .def_readwrite("index", &Device::index)
        .def("__eq__", &Device::operator==)
        .def("__repr__", [](const Device &d)
             { return "<Device '" + deviceToString(d) + "'>"; });

    py::class_<Tensor>(m, "Tensor")
        .def(py::init<const std::vector<int64_t> &, DType, const std::string &, bool>(),
             py::arg("shape"),
             py::arg("dtype") = DType::float32,
             py::arg("device") = "cpu",
             py::arg("requires_grad") = true)
        .def(py::init<py::list, DType, std::string, bool>(),
             py::arg("data"),
             py::arg("dtype") = DType::float32,
             py::arg("device") = "cpu",
             py::arg("requires_grad") = false,
             "Initialize Tensor from a Python list")

        .def_property_readonly("shape", &Tensor::shape, py::return_value_policy::reference_internal)
        .def_property_readonly("strides", &Tensor::strides, py::return_value_policy::reference_internal)
        .def_property_readonly("dtype", &Tensor::dtype)
        .def_property_readonly("device", &Tensor::device, py::return_value_policy::reference_internal)
        .def_property_readonly("requires_grad", &Tensor::requires_grad)
        .def_property_readonly("data", &Tensor::data)

        .def("numel", &Tensor::numel)
        .def("is_contiguous", &Tensor::is_contiguous)
        .def("view", &Tensor::view, py::arg("shape"))
        .def("squeeze", &Tensor::squeeze, py::arg("dim") = -1)
        .def("unsqueeze", &Tensor::unsqueeze, py::arg("dim") = -1)
        .def("permute", &Tensor::permute, py::arg("order"))
        .def("transpose", &Tensor::transpose, py::arg("n"), py::arg("m"))
        .def("expand", &Tensor::expand, py::arg("shape"))
        .def("broadcast", &Tensor::broadcast, py::arg("shape"))
        .def("flatten", &Tensor::flatten, py::arg("start") = 0, py::arg("end") = -1)

        .def("__getitem__", [](const Tensor &t, py::object obj)
             {
            std::vector<std::shared_ptr<IndexStrategy>> strategies;
            const auto& shape = t.shape();

            if (py::isinstance<py::tuple>(obj)) {
                auto tuple = obj.cast<py::tuple>();
                if (tuple.size() > shape.size()) {
                    throw py::index_error("too many indices for tensor: tensor is " +
                                        std::to_string(shape.size()) + "-dimensional, but " +
                                        std::to_string(tuple.size()) + " indices were given");
                }

                for (size_t i = 0; i < tuple.size(); ++i) {
                    py::handle item = tuple[i];
                    if (py::isinstance<py::int_>(item)) {
                        strategies.push_back(std::make_shared<IntegerIndex>(item.cast<int64_t>()));
                    } else if (py::isinstance<py::slice>(item)) {
                        py::slice s = item.cast<py::slice>();
                        int64_t start, stop, step, length;
                        if (!s.compute(shape[i], &start, &stop, &step, &length)) {
                            throw py::error_already_set();
                        }
                        strategies.push_back(std::make_shared<SliceIndex>(start, step, length));
                    } else {
                        throw py::type_error("Unsupported index type in tuple");
                    }
                }
            }
            else if (py::isinstance<py::int_>(obj)) {
                if (shape.empty()) {
                    throw py::index_error("invalid index of a 0-dim tensor.");
                }
                strategies.push_back(std::make_shared<IntegerIndex>(obj.cast<int64_t>()));
            }
            else if (py::isinstance<py::slice>(obj)) {
                if (shape.empty()) {
                    throw py::index_error("Cannot slice a 0-dimensional tensor");
                }
                py::slice s = obj.cast<py::slice>();
                int64_t start, stop, step, length;
                if (!s.compute(shape[0], &start, &stop, &step, &length)) {
                    throw py::error_already_set();
                }
                strategies.push_back(std::make_shared<SliceIndex>(start, step, length));
            } else {
                throw py::type_error("Unsupported index type");
            }

            return t.get_item(strategies); })

        .def("__repr__", [](const Tensor &t)
             {
            std::stringstream ss;
            ss << "Tensor("
               << "data=" << t.data()
               << ", shape=" << shapeToString(t.shape())
               << ", dtype=" << dtypeToString(t.dtype())
               << ", device='" << deviceToString(t.device()) << "'"
               << ", requires_grad=" << (t.requires_grad() ? "True" : "False")
               << ")";
            return ss.str(); });
}
