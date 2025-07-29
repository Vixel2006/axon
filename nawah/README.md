
<br>
<!--
<p align="center">
  <img src="https://i.imgur.com/uUuYfGv.png" alt="Nawah Logo" width="150"/>
</p>
-->
<h1 align="center">Nawah</h1>

<p align="center">
  A deep learning framework designed as a conversation, not a command.
</p>

<p align="center">
  <a href="#"><img alt="Build Status" src="https://img.shields.io/badge/build-passing-brightgreen?style=for-the-badge"></a>
  <a href="#"><img alt="License" src="https://img.shields.io/badge/license-MIT-blue?style=for-the-badge"></a>
  <a href="https://github.com/yushi2006/nawah/issues"><img alt="Issues" src="https://img.shields.io/github/issues/yushi2006/nawah?style=for-the-badge&color=orange"></a>
  <a href="#"><img alt="Stars" src="https://img.shields.io/github/stars/yushi2006/nawah?style=for-the-badge"></a>
</p>

---

Nawah isn't another deep learning framework. **It's an argument.** It argues that the way we build neural networks has become cluttered with boilerplate and cognitive overhead. It argues that building a model should feel like composing a fluid graph, not wrestling with a rigid class hierarchy.

Modern frameworks force your thoughts into their structure. Nawah is built to structure itself around your thoughts.

<br>

## The Philosophy: From Script to Graph

Tired of nested, unreadable forward passes?
```python
# The "Russian Doll" approach
x = self.dropout(self.act(self.bn2(self.conv2(self.act(self.bn1(self.conv1(x)))))))
```

With Nawah, you define **pipelines**, not scripts. The `>>` operator isn't just syntax sugar; it turns your model into a **first-class, queryable pipeline**.

```python
# The Nawah way: A clear, sequential graph
x_flow = x >> self.conv1 >> self.bn1 >> self.act >> self.conv2 >> self.bn2 >> self.act >> self.dropout
```

This simple shift unlocks a new level of developer experience. Let's show you how.

---

## The Nawah API: Four Ideas You'll Love

### 1. Pipelines are First-Class Citizens

Because models are defined as pipelines, you can slice and query them like a list. Need to extract features from a pretrained model? Just slice it.

```python
# Given a trained VisionTransformer model...
model = VisionTransformer()

# Get the feature extractor (all layers except the final classification head)
feature_extractor = model[:-1] 

# Get just the patch embedding and the first two transformer blocks
early_features = model[:3] 

# Apply the partial model
features = x >> feature_extractor
```
**No more complex hook registration or manual layer surgery.** You think in blocks, you slice in blocks.

### 2. Complex Patterns, Simple Decorators

Repetitive structural patterns like residual connections clutter your code. Nawah abstracts them into declarative decorators.

Want to make a block residual? Just decorate it with `@nn.Residual`.

```python
import nawah.nn as nn
import nawah.functional as F

class ResNetBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(channels)
    
    @nn.Residual # <-- That's it. Nawah handles the skip connection.
    @F.relu      # <-- Chain decorators for activation.
    def forward(self, x):
        return x >> self.conv1 >> self.bn1 >> F.relu >> self.conv2 >> self.bn2
```
Your `forward` pass defines the **core logic**. The decorators layer on the **structural pattern**.

### 3. Transparent Internals, Zero Effort

Stop guessing tensor shapes. Nawah models are self-documenting. Printing a module automatically traces a dummy input through the graph to show you the output shape at every single step.

```python
model = ResNetBlock(channels=64)
print(model)
```
```text
> ResNetBlock (Input: [1, 64, 32, 32])
===================================================================
| Layer      | Type          | Output Shape         | Trainable   |
|------------|---------------|----------------------|-------------|
| conv1      | Conv2d        | [1, 64, 32, 32]      | ✓           |
| bn1        | BatchNorm2d   | [1, 64, 32, 32]      | ✓           |
| (F.relu)   | Function      | [1, 64, 32, 32]      |             |
| conv2      | Conv2d        | [1, 64, 32, 32]      | ✓           |
| bn2        | BatchNorm2d   | [1, 64, 32, 32]      | ✓           |
| (F.relu)   | Function      | [1, 64, 32, 32]      |             |
| (Residual) | Connection    | [1, 64, 32, 32]      |             |
===================================================================
```
**No more `print(x.shape)` scattered everywhere.** Debugging is built-in.

### 4. Hackable Gradients, Made Simple

Need to implement gradient clipping, custom gradient modifications, or just inspect gradients during backward passes? Attach a hook with a simple lambda.

```python
# Get a parameter from your model
p = model.conv1.weight

# Clip gradients for this specific parameter during .backward()
p.hook_grad(lambda grad: grad.clamp(-1, 1))

# Log the norm of the gradient every time it's computed
p.hook_grad(lambda grad: print(f"Grad norm for conv1.weight: {grad.norm()}"))

# --- then later in the training loop ---
loss.backward() # Hooks are automatically triggered
```

This gives you low-level control without the boilerplate, perfectly embodying the **"blueprint and wrench"** philosophy.

---

## 💡 The Vision: The Next Step

Nawah's ultimate goal is to bridge the gap between Python's expressiveness and the bare-metal performance of compiled code.

- **NawahScript**: Imagine defining simple models with a powerful, chainable DSL.
  ```python
  # The future is concise.
  model = "Conv(3, 64, k=3, p=1) -> ReLU >> ResBlock(64) * 3 -> AvgPool -> Linear(64, 10)"
  ```
- **JIT Compiler to CUDA**: The pipeline `>>` operator isn't just an operator—it's an Abstract Syntax Tree. We're building a JIT compiler that walks this tree and **fuses operations into a single, high-performance CUDA kernel**, right from your Python code.

---

## ✅ Core Features & Status

- [x] **Expressive & Composable API** (`>>`, `@nn.Residual`)
- [ ] **Full Autograd Engine** (Forward & Backward)
- [ ] **Transparent Debugging** (Automatic shape tracing)
- [ ] **Core Layers & Optimizers** (Linear, Conv2D, BatchNorm, Adam, SGD)
- [x] **Complete CPU Training Loop**
- [ ] **CUDA Backend** (High-performance kernels under development ⚙️)
- [ ] **Model Serialization** (Saving/Loading)
- [ ] **Advanced Layers** (Transformers, Attention, LayerNorm)

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/yushi2006/nawah.git
cd nawah

# Install dependencies and build the C++/CUDA backend
pip install -e .
```

## 📦 A Taste of Nawah

```python
import nawah as nwh
import nawah.nn as nn
import nawah.functional as F
import nawah.optim as optim

class SimpleResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.entry = nn.Conv2d(3, 64, kernel_size=7, stride=2)
        self.block1 = self._make_block(64)
        self.block2 = self._make_block(64)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(64, 10)

    def _make_block(self, channels):
        @nn.Residual
        def block(x):
            return x >> nn.Conv2d(channels, channels, 3, padding=1) \
                     >> nn.BatchNorm2d(channels) \
                     >> F.relu
        return block

    def forward(self, x):
        return (x >> self.entry >> F.relu 
                  >> self.block1 >> self.block2
                  >> self.pool >> nwh.flatten 
                  >> self.head)

model = SimpleResNet()
print(model) # See the beautiful, automatic shape trace!
```

---

## 🤝 Contributing

Nawah is built for builders. If you're passionate about creating elegant developer tools and diving deep into the internals of deep learning, you belong here.

**How you can help:**
- **Expand the Layer Zoo**: Implement new and interesting layers.
- **Refine the API**: Have an idea for an even more expressive API? Let's hear it.
- **Write Docs & Examples**: Help others fall in love with the Nawah workflow.

**Fork the repo. Open an issue. Let's build the future of DL tooling together.**

---

## 👤 Author

**Yusuf Mohamed**  
ML Researcher | ML Engineer | Open-source Builder  

- Creator of **GRF** (Gated Recursive Fusion)  
- Building **Nawah** as a clean-slate deep learning framework  
- Contributor at **Hugging Face**  

📎 [GitHub – @yushi2006](https://github.com/yushi2006)  
📎 [LinkedIn – Yusuf Mohamed](https://www.linkedin.com/in/yusufmohamed2006/)

---
## 📄 License

**MIT License** — free to use, modify, and commercialize with attribution.