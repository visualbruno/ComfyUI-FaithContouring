# 🌀 ComfyUI Wrapper for https://github.com/visualbruno/FaithC

---

<img width="804" height="570" alt="{2A2A680C-8115-4D38-9BD5-44A6B0F46535}" src="https://github.com/user-attachments/assets/1d9f1323-283a-4f27-9777-95018b0cbd09" />

---

## ⚙️ Installation Guide

> Tested on **Windows 11** with **Python 3.11** and **Torch = 2.7.0 + cu128**.

### 1. Install the requirements

#### For a standard python environment:

```bash
python -m pip install -r ComfyUI/custom_nodes/ComfyUI-FaithContouring/requirements.txt
```

---

#### For ComfyUI Portable:

```bash
python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-FaithContouring\requirements.txt
```

### 1. Install atom3d wheel

You will find the precompiled wheels in the folder **wheels**

#### For a standard python environment:

```bash
python -m pip install ComfyUI/custom_nodes/ComfyUI-FaithContouring/wheels/Windows/Torch270/atom3d-0.1.0-cp311-cp311-win_amd64.whl
```

#### For ComfyUI Portable:

```bash
python_embeded\python.exe -m pip install ComfyUI\custom_nodes\ComfyUI-FaithContouring\wheels\Windows\Torch270\atom3d-0.1.0-cp311-cp311-win_amd64.whl
```

#### Custom build:

You can compile a custom build from the repository https://github.com/visualbruno/FaithC, in the folder "Atom3d"
