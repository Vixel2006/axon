<h1 align="center">Idrak</h1>

<p align="center">
  This is not a deep learning framework. It's an argument.
</p>

<p align="center">
  <a href="#"><img alt="Build Status" src="https://img.shields.io/badge/build-passing-brightgreen?style=for-the-badge"></a>
  <a href="#"><img alt="License" src="https://img.shields.io/badge/license-MIT-blue?style=for-the-badge"></a>
  <a href="https://github.com/yushi2006/nawah/issues"><img alt="Issues" src="https://img.shields.io/github/issues/yushi2006/nawah?style=for-the-badge&color=orange"></a>
  <a href="#"><img alt="Stars" src="https://img.shields.io/github/stars/yushi2006/nawah?style=for-the-badge"></a>
</p>

---

## The Heresy We Propose

For too long, deep learning frameworks have suffered from the **Coupling Problem**. Your model's architecture (`__init__`), its core logic (`forward`), its training loop, and its configuration are all entangled within a single, monolithic class. This leads to boilerplate, hidden states, and a development process that feels more like archaeology than creation.

Idrak is built on a simple, powerful, and controversial idea: **The Great Separation.**

We argue that a sane workflow demands the complete decoupling of three distinct concepts:
1.  **The State (`Net`):** Your model's learnable parameters. It should be nothing more than a transparent, dictionary-like container. A blueprint registry.
2.  **The Configuration (`Config`):** The declarative "single source of truth" that defines the architecture, hyperparameters, and experiment settings.
3.  **The Execution (`Runner`):** The boilerplate engine that handles the `for` loops of training, validation, and inference, freeing you to focus only on the unique logic.

Stop building god objects. Start designing clean, composable systems.

<br>

## The Idrak Way in 4 Steps

See how The Great Separation leads to a cleaner, faster, and more hackable workflow.

### Step 1: Define the Truth (`Config`)

```python
import idrak as nw

class TrainingConfig:
    in_channels = 3
    base_channels = 64
    block_type = 'ResBlock'
    learning_rate = 1e-3
    batch_size = 128
    epochs = 10
    optimizer = 'Adam'
```

### Step 2: Build the State (`Net`)

```python
from idrak.factories import resnet_block

config = TrainingConfig()
model = nn.Net()

model.add("entry_conv", nn.conv2d(config.in_channels, config.base_channels, kernel_size=7))
model.add("block1", resnet_block(config.base_channels))
model.add("block2", resnet_block(config.base_channels))
model.add("head", nn.linear(config.base_channels, 10))
```

### Step 3: Define YOUR Logic (`training_step`)

```python
def my_training_step(runner_state):
    net = runner_state.net
    batch = runner_state.batch

    x, y_true = batch

    y_pred = net(x)

    loss = nw.functional.cross_entropy(y_pred, y_true)

    return {"loss": loss, "y_pred": y_pred}
```

### Step 4: Let the Runner Execute

```python
runner = nw.Runner(config=config, model=model)

history = runner.train(
    training_step_fn=my_training_step,
    train_dataset=my_train_data,
    eval_dataset=my_eval_data,
    metrics=[nw.metrics.Accuracy(), nw.metrics.F1Score()]
)
```

---

## 💣 Hackability Is Not a Feature. It's the Point.

Most libraries hide their internals like it’s some sacred artifact. Idrak doesn’t do that. Everything is an object you can poke, inspect, and mutate at runtime.

- 🔍 Access **all model parameters** with `net.params`
- 🧱 Access **buffers** (non-trainable states) with `net.buffers`
- 🔧 Swap any layer with `net['layer_name'] = new_layer`
- 🧠 Replace functions on-the-fly: `net['fc1']['fn'] = custom_linear`
- 💥 Inject any arbitrary PyFunc into a pipeline step
- 👁️‍🗨️ Pure functional `>>` pipelines — you trace, log, wrap, or fuse
- 🧬 Plug and play — it’s just Python dicts and functional calls

Frameworks treat you like a user. Idrak treats you like a hacker.

You want to freeze weights? Detach the gradient. You want to mutate activations on the fly? Go ahead. Want to track internal outputs? Inject a hook or just override the fn inline. It's your engine. Drive it like a maniac.

---

## 🔥 The Vision: A Fully Fused Stack

The Great Separation enables our ultimate goal: bridging the gap between high-level expression and bare-metal performance.

- **JIT Compiler for CUDA:** The explicit pipeline (`>>`) is a parsable AST. We will trace and fuse it into high-performance CUDA kernels.
- **Config-Driven Architecture:** Define your model in the Config, and the Net builds itself.

---

## ✅ Core Features & Status

- ✅ Core Philosophy: The Great Separation (Net, Config, Runner)
- ✅ Transparent API: Dictionary-based layers and explicit data flows.
- ✅ Core Autograd Engine: Tape-based and fully functional.
- 🔧 JIT Compilation Engine: AST parsing and fusion (In Development)
- ✅ High-Level Runner: Training, evaluation, and inference handled
- ✅ Built-in Metrics: Accuracy, Precision, F1, etc.
- ✅ Core Layers, Losses, Optimizers
- ✅ CUDA Backend: Custom kernels for performance

---

## 🚀 Getting Started

```bash
git clone https://github.com/yushi2006/idrak.git
cd idrak
pip install -e .
```

---

## 🤝 Contributing

If this argument resonates with you—if you believe clarity, hackability, and performance can co-exist—then you belong here.

- Implement Runner Features
- Expand the Layer Zoo
- Build Metrics Modules

Fork it. Break it. Build it better.

---

## 👤 Author

**Yusuf Mohamed**  
ML Researcher | ML Engineer | Open-source Builder  
Creator of GRF (Gated Recursive Fusion)  
Contributor at Hugging Face  
📎 GitHub – [@yushi2006](https://github.com/yushi2006)  
📎 LinkedIn – Yusuf Mohamed  

---

## 📄 License

MIT License — free to use, modify, and commercialize with attribution.
