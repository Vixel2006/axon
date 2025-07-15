# 🧠 Nawah — A Minimal Deep Learning Framework (WIP)

**Nawah** is a lightweight, from-scratch deep learning framework built with one thing in mind:

> 🧩 **Developer Experience First.**

Not just raw performance or completeness — but a code *interface* that feels like you're building a Nawahl system, not wrestling with a jungle of wrappers.

---

## ✨ What's Special About Nawah?

> Modern DL libraries are powerful, but cluttered. Nawah flips the script.

It introduces a **clean, expressive API** for defining models — chaining layers like functions, composing blocks with decorators, and keeping your code readable at scale.

Instead of this:
```python
x = self.bn2(self.conv2(self.bn1(self.conv1(x))))
```
You write this:
```python
x = x >> self.conv1 >> self.bn1 >> self.conv2 >> self.bn2
```

Want to attach an activation? Just decorate the block:
```python
@F.relu
def conv_block(self, x):
    return x >> self.conv >> self.bn
```

✅ Composable
✅ Reusable
✅ Easy to reason about
✅ Feels like you're defining a graph, not a script

## ✅ Currently Supports

- Forward & backward tensor ops  
- Autograd engine  
- Optimizers (SGD, Adam)  
- Full training loop for MNIST (CPU only)  
- Elegant model definition syntax  
- Basic layers (Linear, Conv2D, ReLU, BatchNorm, etc.)

> ⚙️ **GPU backend is under active development (CUDA support coming in hot)**

---

## 💡 Philosophy

> Nawah is built for *builders* — not abstract users.

Most frameworks prioritize massive scale and generalization. Nawah prioritizes **clarity, hackability, and control**.

- You *see* the gradient flow  
- You *define* the ops  
- You can *compile Python code into CUDA kernels* (yes, that’s coming)  
- You design models like building Lego, not nesting Russian dolls

We care about:
- Fast iteration  
- Ergonomic design  
- Bare-metal understanding  
- Prototyping research ideas without friction

This is a **developer-first**, not “enterprise-first” framework.

---

## 📦 Example: Training a Model (Minimal Style)

```python
import nawah.nn as nn
import nawah.optim as optim
from nawah.data import Dataset, DataLoader

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(784, 128)
        self.relu = ReLU()
        self.fc2 = Linear(128, 10)

    def forward(self, x):
        return x >> self.fc1 >> self.relu >> self.fc2

model = MLP()
loss_fn = BCEWithLogitLoss()
optimizer = Adam(model.parameters(), lr=0.01)

for epoch in range(5):
    for x_batch, y_batch in train_loader:
        preds = model(x_batch)
        loss = loss_fn(preds, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## 🛣️ Roadmap

- [x] CPU training loop + layers  
- [x] Expressive model API (`>>`, decorators)  
- [x] Convolutional layers (Conv2D, BatchNorm2D)  
- [ ] GPU backend (CUDA kernels)  
- [ ] Model saving/loading  
- [ ] Transformer blocks (Multihead Attention, LayerNorm)  
- [ ] CLI launcher + config system  
- [ ] `Nawahscript`: a DSL for defining models in 3 lines or less  
- [ ] Training benchmark suite (compare against PyTorch, NumPy baselines)  

---

## 👤 Author

**Yusuf Mohamed**  
AI Researcher | ML Engineer | Open-source Builder  

- Creator of **GRF** (Gated Recursive Fusion)  
- Building **Nawah** as a clean-slate deep learning framework  
- Contributor at **Hugging Face**  

📎 [GitHub – @yushi2006](https://github.com/yushi2006)  
📎 [LinkedIn – Yusuf Mohamed](https://www.linkedin.com/in/yusufmohamed2006/)

---

## 🤝 Contributing

Nawah is currently a one-man project, but it's open for contributions.

You can help with:
- Writing CUDA kernels  
- Expanding layer coverage  
- Improving training utilities  
- Fixing bugs and improving developer experience  

If you're passionate about low-level DL tooling, compiler design, or building custom model infra — you're welcome here.

**Fork the repo. Open an issue. Build the future.**

---

## 🧠 Nawah’s Vision

Nawah isn’t trying to be the next PyTorch — it’s a rethink of how we *build* and *interface with* Nawahl networks.

### It’s built to:
- Let you define models as **semantic, readable pipelines**  
- Strip back unnecessary layers of abstraction  
- Give you control from **Python to CUDA**  
- Be a real tool for research and experimentation, not just deployment

> If PyTorch is a spaceship, Nawah is the blueprint, the engine, and the wrench.

Use it to learn. Use it to build. Use it to push the limits.

---

## 📄 License

**MIT License** — free to use, modify, and commercialize with attribution.

