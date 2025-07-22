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

                // Helper lambda to correctly parse a Python slice and create a SliceIndex strategy.
                // This is the core of the improved logic.
                auto process_slice = [&](py::slice s, size_t dim_index) {
                    if (dim_index >= shape.size()) {
                        throw py::index_error("too many indices for tensor");
                    }
                    const __int64_t dim_size = shape[dim_index];

                    // Manually extract start, stop, and step, handling the case where they are `None`.
                    // This passes the raw, un-normalized values to our powerful C++ `SliceIndex`.
                    __int64_t start, stop, step;

                    // Step 1: Get step, defaulting to 1 if None.
                    step = s.attr("step").is_none() ? 1 : s.attr("step").cast<__int64_t>();

                    if (step == 0) {
                        throw py::value_error("slice step cannot be zero");
                    }

                    // Step 2: Get start, providing the correct default if None (depends on step direction).
                    if (s.attr("start").is_none()) {
                        start = (step > 0) ? 0 : dim_size - 1;
                    } else {
                        start = s.attr("start").cast<__int64_t>();
                    }

                    // Step 3: Get stop, providing the correct default if None (depends on step direction).
                    if (s.attr("stop").is_none()) {
                        // Use `dim_size` for forward slicing and `-1` for backward slicing.
                        // Our C++ code is designed to understand these sentinels.
                        stop = (step > 0) ? dim_size : -1; 
                    } else {
                        stop = s.attr("stop").cast<__int64_t>();
                    }
                    
                    // Create the strategy with the raw, un-normalized values.
                    strategies.push_back(std::make_shared<SliceIndex>(start, stop, step));
                };

                // --- Main Logic ---

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
                            process_slice(item.cast<py::slice>(), i);
                        } else {
                            // Note: A production library would also handle `None` (for new axes) 
                            // and `Ellipsis` (...) here. This implementation focuses on the core types.
                            throw py::type_error("Unsupported index type in tuple");
                        }
                    }
                } else if (py::isinstance<py::int_>(obj)) {
                    if (shape.empty()) {
                        throw py::index_error("invalid index of a 0-dim tensor.");
                    }
                    strategies.push_back(std::make_shared<IntegerIndex>(obj.cast<int64_t>()));
                } else if (py::isinstance<py::slice>(obj)) {
                    if (shape.empty()) {
                        throw py::index_error("Cannot slice a 0-dimensional tensor");
                    }
                    process_slice(obj.cast<py::slice>(), 0);
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
            return ss.str(); })

        .def("__add__", &Tensor::add)
        .def("__sub__", &Tensor::sub)
        .def("__mul__", &Tensor::mul)
        .def("__matmul__", &Tensor::matmul);
    
    m.def("cuda_synchronize", &cuda_synchronize, "Synchronize CUDA device");
}
