#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "helpers.h"
#include "tensor.h"
#include "init.h"
#include "engine/ops.h"
#include <stdexcept>

static CpuOps cpu_ops;
static CudaOps cuda_ops;

Tensor relu_dispatcher(const Tensor &a) {
    if (a.device().type == DeviceType::CPU) {
        return cpu_ops.relu(a);
    } else if (a.device().type == DeviceType::CUDA) {
        return cuda_ops.relu(a);
    } else {
        // Throw an error for unsupported devices.
        throw std::runtime_error("Unsupported device for relu: " + deviceToString(a.device()));
    }
}


Tensor log_dispatcher(const Tensor &a) {
    if (a.device().type == DeviceType::CPU) {
        return cpu_ops.log(a);
    } else if (a.device().type == DeviceType::CUDA) {
        return cuda_ops.log(a);
    } else {
        // Throw an error for unsupported devices.
        throw std::runtime_error("Unsupported device for relu: " + deviceToString(a.device()));
    }
}

Tensor exp_dispatcher(const Tensor &a) {
    if (a.device().type == DeviceType::CPU) {
        return cpu_ops.exp(a);
    } else if (a.device().type == DeviceType::CUDA) {
        return cuda_ops.exp(a);
    } else {
        // Throw an error for unsupported devices.
        throw std::runtime_error("Unsupported device for relu: " + deviceToString(a.device()));
    }
}


Tensor softmax_dispatcher(const Tensor &a) {
    if (a.device().type == DeviceType::CPU) {
        return cpu_ops.softmax(a);
    } else if (a.device().type == DeviceType::CUDA) {
        return cuda_ops.softmax(a);
    } else {
        // Throw an error for unsupported devices.
        throw std::runtime_error("Unsupported device for relu: " + deviceToString(a.device()));
    }
}

Tensor conv_dispatcher(const Tensor &a, const Tensor& kernel, int stride, int padding) {
    if (a.device().type == DeviceType::CPU) {
        return cpu_ops.conv2d(a, kernel, stride, padding);
    } else if (a.device().type == DeviceType::CUDA) {
        return cuda_ops.conv2d(a, kernel, stride, padding);
    } else {
        // Throw an error for unsupported devices.
        throw std::runtime_error("Unsupported device for conv2d: " + deviceToString(a.device()));
    }
}

namespace py = pybind11;

PYBIND11_MODULE(cnawah, m) {
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
      .def(py::init<DeviceType, int>(), py::arg("type") = DeviceType::CPU,
           py::arg("index") = 0)
      .def_readwrite("type", &Device::type)
      .def_readwrite("index", &Device::index)
      .def("__eq__", &Device::operator==)
      .def("__repr__", [](const Device &d) {
        return "<Device '" + deviceToString(d) + "'>";
      });

  py::class_<Tape>(m, "Tape")
    .def(py::init<>())
    .def_readwrite("prev", &Tape::prev)
    .def_readwrite("backward_fn", &Tape::backward_fn);

  py::class_<Tensor>(m, "Tensor")
      .def(py::init<const std::vector<int64_t> &, DType, const std::string &,
                    bool>(),
           py::arg("shape"), py::arg("dtype") = DType::float32,
           py::arg("device") = "cpu", py::arg("requires_grad") = false)

      .def(py::init<py::list, DType, std::string, bool>(), py::arg("data"),
           py::arg("dtype") = DType::float32, py::arg("device") = "cpu",
           py::arg("requires_grad") = false,
           "Initialize Tensor from a Python list")


      .def_property_readonly("shape", &Tensor::shape,
                             py::return_value_policy::reference_internal)
      .def_property_readonly("strides", &Tensor::strides,
                             py::return_value_policy::reference_internal)
      .def_property_readonly("dtype", &Tensor::dtype)
      .def_property_readonly("device", &Tensor::device,
                             py::return_value_policy::reference_internal)
      .def_property_readonly("requires_grad", &Tensor::requires_grad)
      .def_property_readonly("data", &Tensor::data)
      .def_property_readonly("grad", &Tensor::grad)
      .def_property_readonly("ctx", &Tensor::ctx)

      .def("numel", &Tensor::numel)
      .def("is_contiguous", &Tensor::is_contiguous)
      .def("view", &Tensor::view, py::arg("shape"))
      .def("squeeze", &Tensor::squeeze, py::arg("dim") = -1)
      .def("unsqueeze", &Tensor::unsqueeze, py::arg("dim") = -1)
      .def("permute", &Tensor::permute, py::arg("order"))
      .def("transpose", &Tensor::transpose, py::arg("n"), py::arg("m"))
      .def("expand", &Tensor::expand, py::arg("shape"))
      .def("broadcast", &Tensor::broadcast, py::arg("shape"))
      .def("flatten", &Tensor::flatten, py::arg("start") = 0,
           py::arg("end") = -1)

      .def("__getitem__",
           [](const Tensor &t, py::object obj) {
             std::vector<std::shared_ptr<IndexStrategy>> strategies;
             const auto &shape = t.shape();

             auto process_slice = [&](py::slice s, size_t dim_index) {
               if (dim_index >= shape.size()) {
                 throw py::index_error("too many indices for tensor");
               }
               const __int64_t dim_size = shape[dim_index];
               __int64_t start, stop, step;

               step = s.attr("step").is_none()
                          ? 1
                          : s.attr("step").cast<__int64_t>();
               if (step == 0) {
                 throw py::value_error("slice step cannot be zero");
               }

               if (s.attr("start").is_none()) {
                 start = (step > 0) ? 0 : dim_size - 1;
               } else {
                 start = s.attr("start").cast<__int64_t>();
               }

               if (s.attr("stop").is_none()) {
                 stop = (step > 0) ? dim_size : -1;
               } else {
                 stop = s.attr("stop").cast<__int64_t>();
               }

               strategies.push_back(
                   std::make_shared<SliceIndex>(start, stop, step));
             };

             if (py::isinstance<py::tuple>(obj)) {
               auto tuple = obj.cast<py::tuple>();
               if (tuple.size() > shape.size()) {
                 throw py::index_error(
                     "too many indices for tensor: tensor is " +
                     std::to_string(shape.size()) + "-dimensional, but " +
                     std::to_string(tuple.size()) + " indices were given");
               }

               for (size_t i = 0; i < tuple.size(); ++i) {
                 py::handle item = tuple[i];
                 if (py::isinstance<py::int_>(item)) {
                   strategies.push_back(
                       std::make_shared<IntegerIndex>(item.cast<int64_t>()));
                 } else if (py::isinstance<py::slice>(item)) {
                   process_slice(item.cast<py::slice>(), i);
                 } else {
                   throw py::type_error("Unsupported index type in tuple");
                 }
               }
             } else if (py::isinstance<py::int_>(obj)) {
               if (shape.empty()) {
                 throw py::index_error("invalid index of a 0-dim tensor.");
               }
               strategies.push_back(
                   std::make_shared<IntegerIndex>(obj.cast<int64_t>()));
             } else if (py::isinstance<py::slice>(obj)) {
               if (shape.empty()) {
                 throw py::index_error("Cannot slice a 0-dimensional tensor");
               }
               process_slice(obj.cast<py::slice>(), 0);
             } else {
               throw py::type_error("Unsupported index type");
             }

             return t.get_item(strategies);
           })

      .def("__repr__",
           [](const Tensor &t) {
             std::stringstream ss;
             ss << "Tensor("
                << "data=" << t.data() << ", shape=" << shapeToString(t.shape())
                << ", dtype=" << dtypeToString(t.dtype()) << ", device='"
                << deviceToString(t.device()) << "'"
                << ", requires_grad=" << (t.requires_grad() ? "True" : "False")
                << ")";
             return ss.str();
           })

      .def("__add__", static_cast<Tensor (Tensor::*)(const Tensor&) const>(&Tensor::add))
      .def("__add__", static_cast<Tensor (Tensor::*)(float) const>(&Tensor::add))
      .def("__sub__", static_cast<Tensor (Tensor::*)(const Tensor&) const>(&Tensor::sub))
      .def("__sub__", static_cast<Tensor (Tensor::*)(float) const>(&Tensor::sub))
      .def("__mul__", static_cast<Tensor (Tensor::*)(const Tensor&) const>(&Tensor::mul))
      .def("__mul__", static_cast<Tensor (Tensor::*)(float) const>(&Tensor::mul))
      .def("__matmul__", &Tensor::matmul)
      .def("__truediv__", static_cast<Tensor (Tensor::*)(const Tensor&) const>(&Tensor::div),
             "Performs element-wise division with another tensor.")
      .def("__truediv__", static_cast<Tensor (Tensor::*)(float) const>(&Tensor::div),
             "Performs element-wise division with a scalar.")
      .def("__neg__", &Tensor::neg)

      .def("sum",
          [](const Tensor &self) {
              return self.sum();
          }
      )

      .def("sum",
          [](const Tensor &self, py::object dim_arg, bool keepdim) {
            if (py::isinstance<py::int_>(dim_arg)) {
              return self.sum(dim_arg.cast<int>(), keepdim);
            }
            throw py::type_error("sum(): 'dim' argument must be an integer.");
          },
          "Calculates the sum of tensor elements over a given dimension.",
          py::arg("dim") = py::none(),
          py::arg("keepdim") = false
      )


      .def("mean",
          [](const Tensor &self) {
              return self.mean();
          }
      )

      .def("mean",
          [](const Tensor &self, py::object dim_arg, bool keepdim) {
            if (py::isinstance<py::int_>(dim_arg)) {
              return self.mean(dim_arg.cast<int>(), keepdim);
            }
            throw py::type_error("mean(): 'dim' argument must be an integer.");
          },
          "Calculates the mean of tensor elements over a given dimension.",
          py::arg("dim") = py::none(),
          py::arg("keepdim") = false
      )


      .def("build_topo", &Tensor::build_topo)
      .def("backward", &Tensor::backward);

    m.def("cuda_synchronize", &cuda_synchronize, "Synchronize CUDA device");
    m.def(
        "relu",
        &relu_dispatcher,
        "Applies the Rectified Linear Unit function element-wise.",
        py::arg("a")
    );


    m.def(
        "log",
        &log_dispatcher,
        "Applies the log",
        py::arg("a")
    );


    m.def(
        "exp",
        &exp_dispatcher,
        "Applies the exp.",
        py::arg("a")
    );


    m.def(
        "softmax",
        &softmax_dispatcher,
        "Applies the softmax operation.",
        py::arg("a")
    );


    m.def(
        "conv2d",
        &conv_dispatcher,
        "Applies the conv2d operation.",
        py::arg("a"),
        py::arg("kernel"),
        py::arg("stride") = 1,
        py::arg("padding") = 1
    );

    m.def("zeros", &zeros, py::arg("shape"), py::arg("device") = "cpu", py::arg("requires_grad") = false,
              "Creates a tensor filled with zeros.");
              
    m.def("ones", &ones, py::arg("shape"), py::arg("device") = "cpu", py::arg("requires_grad") = false,
          "Creates a tensor filled with ones.");
          
    m.def("randn", &randn, py::arg("shape"), py::arg("device") = "cpu", py::arg("requires_grad") = false,
          "Creates a tensor with random numbers from a standard normal distribution.");
          
    m.def("uniform", &uniform, py::arg("shape"), py::arg("device") = "cpu", py::arg("requires_grad") = false,
          "Creates a tensor with random numbers from a uniform distribution [0, 1).");
          
    m.def("zeros_like", &zeros_like, py::arg("other"),
          "Creates a tensor of zeros with the same properties as another tensor.");
          
    m.def("ones_like", &ones_like, py::arg("other"),
          "Creates a tensor of ones with the same properties as another tensor.");

}


