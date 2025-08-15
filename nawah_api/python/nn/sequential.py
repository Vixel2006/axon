from ..core import Tensor
import numpy as np
import cnawah as nw

class Sequential:
    def __init__(self, modules={}):
        self.modules = modules
        self.params = {}
        self.buffers = {}
        self.device = "cpu"

    def _deregister_module(self, module_name):
        params_to_remove = [p for p in self.params if p.startswith(f"{module_name}.")]
        for p in params_to_remove:
            del self.params[p]

        buffers_to_remove = [b for b in self.buffers if b.startswith(f"{module_name}.")]
        for b in buffers_to_remove:
            del self.buffers[b]


    def add(self, name, module):
        if not isinstance(name, str) or not name:
            raise TypeError("Module name must be a non-empty string.")
        if name in self.modules:
            raise ValueError(f"Module name '{name}' already exists.")
        if not isinstance(module, dict) or 'fn' not in module:
            raise TypeError(f"Module for '{name}' is not a valid dictionary created by a @nw.pipe function.")

        self.modules[name] = module

        for pname, param in module.get("params", {}).items():
            full_name = f"{name}.{pname}"
            if hasattr(param, 'requires_grad') and param.requires_grad:
                self.params[full_name] = param
            else:
                self.buffers[full_name] = param

    def remove(self, module_name):
        self._deregister_module(module_name)

    def forward(self, x):
        for module in self.modules.values():
            x = module['fn'](x)
        return x

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        lines = [f"  ({name}): {module.get('name', 'Module')}" for name, module in self.modules.items()]
        return "Sequential(\n" + "\n".join(lines) + "\n)"

    def __setitem__(self, name, module):
        if not isinstance(module, dict) or 'fn' not in module:
            raise TypeError(f"Module for '{name}' is not a valid dictionary.")

        if name in self.modules:
            self._deregister_module(name)

        self.modules[name] = module
        for pname, param in module.get("params", {}).items():
            full_name = f"{name}.{pname}"
            if hasattr(param, 'requires_grad') and param.requires_grad:
                self.params[full_name] = param
            else:
                self.buffers[full_name] = param


    def __getitem__(self, name):
        return self.modules[name]

    def to(self, device: str):
        for _, param in self.params.items():
            param.to(device)
        for _, buffer in self.buffers.items():
            param.to(device)
        self.device = device

    def summary(self, input_shape):
        col_widths = {"Layer": 12, "Type": 15, "Output Shape": 22, "Trainable": 13}
        header = (f"| {'Layer':<{col_widths['Layer']}} | {'Type':<{col_widths['Type']}} | {'Output Shape':<{col_widths['Output Shape']}} | {'Trainable':<{col_widths['Trainable']}} |")
        separator = "=" * len(header)

        summary_str = f"Sequential (Input: {list(input_shape)})\n{separator}\n{header}\n"
        summary_str += f"|{'-'*(col_widths['Layer']+2)}|{'-'*(col_widths['Type']+2)}|{'-'*(col_widths['Output Shape']+2)}|{'-'*(col_widths['Trainable']+2)}|\n"

        dummy_tensor = nw.zeros(input_shape, device=self.device)
        total_params = 0
        trainable_params = 0

        for name, module in self.modules.items():
            dummy_tensor = module['fn'](dummy_tensor)
            output_shape = dummy_tensor.shape
            layer_type = module.get('name', 'Function')
            has_trainable_params = any(p.requires_grad for p in module.get("params", {}).values())
            trainable_str = "✓" if has_trainable_params else ""

            for param in module.get("params", {}).values():
                param_size = np.prod(param.shape)
                total_params += param_size
                if param.requires_grad:
                    trainable_params += param_size

            summary_str += (f"| {name:<{col_widths['Layer']}} | {layer_type:<{col_widths['Type']}} | {str(list(output_shape)):<{col_widths['Output Shape']}} | {trainable_str:<{col_widths['Trainable']}} |\n")

        summary_str += f"{separator}\n"
        summary_str += f"Total params: {total_params:,}\n"
        summary_str += f"Trainable params: {trainable_params:,}\n"
        summary_str += f"Non-trainable params: {total_params - trainable_params:,}\n"

        print(summary_str)


    def load_state_dict(self, state_dict):
        current_model_tensors = {**self.params, **self.buffers}
        for name, loaded_data in state_dict.items():
            if name not in current_model_tensors:
                print(f"Warning: Key '{name}' in state_dict is not found in the current model. Skipping.")
                continue

            model_tensor = current_model_tensors[name]
            if model_tensor.shape != loaded_data.shape:
                raise ValueError(f"Shape mismatch for '{name}': loaded tensor has shape {loaded_data.shape}, but model tensor has shape {model_tensor.shape}.")

            model_tensor.data = loaded_data

        for name in current_model_tensors:
            if name not in state_dict:
                print(f"Warning: Key '{name}' in the current model is not found in the loaded state_dict.")

