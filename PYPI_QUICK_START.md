# ConMedRL PyPI 发布快速开始

## ✅ 我已为您完成的准备工作

### 1. 修复了包导入错误
- ✅ 修正了`ConMedRL/__init__.py`中的导入路径（`concarerl` → `conmedrl`）
- ✅ 移除了不存在的`done_condition_function_examples`模块导入
- ✅ 清理了`setup.py`和`pyproject.toml`中不存在的入口点

### 2. 验证了包结构
```
ConMedRL/
├── __init__.py ✅ (版本 1.0.0)
├── conmedrl.py ✅
├── conmedrl_continuous.py ✅
└── data_loader.py ✅

Data/
├── __init__.py ✅
├── mimic_iv_icu_discharge/ ✅
├── mimic_iv_icu_extubation/ ✅
├── SICdb_discharge/ ✅
└── SICdb_extubation/ ✅
```

## 🚀 您现在可以开始的步骤

### 第一步：验证当前配置

在PowerShell中运行：

```powershell
# 检查Python版本
python --version

# 验证包可以导入（在本地）
python -c "import ConMedRL; print('Version:', ConMedRL.__version__)"
```

**预期输出**: `Version: 1.0.0`

如果导入失败，设置PYTHONPATH：
```powershell
$env:PYTHONPATH = "$env:PYTHONPATH;$(Get-Location)"
```

### 第二步：安装必要工具

```powershell
# 升级pip
python -m pip install --upgrade pip

# 安装构建和上传工具
pip install --upgrade build twine wheel setuptools
```

### 第三步：执行构建前测试

```powershell
# 测试包的基本导入
python -c "from ConMedRL import FQE, FQI, TrainDataLoader, ValTestDataLoader; print('Core imports: OK')"
python -c "from Data import mimic_iv_icu_discharge; print('Data module: OK')"
```

### 第四步：清理旧构建文件

```powershell
# 如果存在旧的构建文件，清理它们
if (Test-Path build) { Remove-Item -Recurse -Force build }
if (Test-Path dist) { Remove-Item -Recurse -Force dist }
if (Test-Path ConMedRL.egg-info) { Remove-Item -Recurse -Force ConMedRL.egg-info }

# 清理__pycache__
Get-ChildItem -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force
```

### 第五步：构建包

```powershell
# 构建源码和wheel包
python -m build
```

**成功标志**: 
- 应该在`dist/`目录下生成两个文件：
  - `ConMedRL-1.0.0.tar.gz`
  - `ConMedRL-1.0.0-py3-none-any.whl`

查看生成的文件：
```powershell
Get-ChildItem dist
```

### 第六步：检查包的质量

```powershell
# 使用twine检查包
python -m twine check dist/*
```

**预期输出**: 
```
Checking dist/ConMedRL-1.0.0.tar.gz: PASSED
Checking dist/ConMedRL-1.0.0-py3-none-any.whl: PASSED
```

### 第七步：本地测试安装

```powershell
# 创建测试虚拟环境
python -m venv test_install_env

# 激活虚拟环境
.\test_install_env\Scripts\Activate.ps1

# 安装构建的包
pip install dist/ConMedRL-1.0.0-py3-none-any.whl

# 测试导入
python -c "import ConMedRL; print('ConMedRL Version:', ConMedRL.__version__)"
python -c "from ConMedRL import FQE, FQI; print('Core classes imported successfully')"
python -c "from Data import mimic_iv_icu_discharge; print('Data module imported successfully')"

# 退出虚拟环境
deactivate

# 清理测试环境
Remove-Item -Recurse -Force test_install_env
```

### 第八步：准备PyPI账户（如果还没有）

#### 8.1 注册账户

**TestPyPI（用于测试）**:
1. 访问: https://test.pypi.org/account/register/
2. 填写信息并验证邮箱

**正式PyPI**:
1. 访问: https://pypi.org/account/register/
2. 填写信息并验证邮箱

#### 8.2 启用双因素认证（必需）

两个平台都需要：
1. 登录账户
2. 进入 Account Settings → Add 2FA → Use app
3. 使用Google Authenticator等应用扫描二维码
4. **重要**: 保存恢复代码！

#### 8.3 生成API Token

**TestPyPI Token:**
1. 登录 https://test.pypi.org/
2. Account Settings → API tokens → Add API token
3. Token名称: `ConMedRL-test`
4. Scope: Entire account (首次上传)
5. **复制Token** (格式: `pypi-xxx...`)
6. 保存到安全位置

**PyPI Token:**
1. 登录 https://pypi.org/
2. 重复上述步骤
3. Token名称: `ConMedRL-production`

### 第九步：上传到TestPyPI（强烈推荐先测试）

```powershell
# 上传到TestPyPI
python -m twine upload --repository testpypi dist/*
```

**提示输入时**:
- Username: `__token__`
- Password: 粘贴您的TestPyPI token (以`pypi-`开头)

**成功后**，您会看到类似输出：
```
View at:
https://test.pypi.org/project/ConMedRL/1.0.0/
```

### 第十步：从TestPyPI测试安装

```powershell
# 创建新的测试环境
python -m venv testpypi_install

# 激活
.\testpypi_install\Scripts\Activate.ps1

# 从TestPyPI安装
# 注意：需要--extra-index-url因为依赖包在正式PyPI上
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ ConMedRL

# 测试
python -c "import ConMedRL; print('Installed version:', ConMedRL.__version__)"
python -c "from ConMedRL import FQE, FQI, TrainDataLoader; print('Success!')"

# 退出并清理
deactivate
Remove-Item -Recurse -Force testpypi_install
```

### 第十一步：发布到正式PyPI

⚠️ **最后检查**:
- [ ] 所有测试都通过了吗？
- [ ] TestPyPI安装成功了吗？
- [ ] README.md显示正确吗？
- [ ] 版本号正确吗？（发布后无法修改）
- [ ] 包名称没有被占用吗？

**确认无误后**，执行：

```powershell
# 上传到正式PyPI
python -m twine upload dist/*
```

**提示输入时**:
- Username: `__token__`
- Password: 粘贴您的正式PyPI token

**成功！** 🎉

访问您的包：https://pypi.org/project/ConMedRL/

### 第十二步：最终验证

```powershell
# 创建全新环境
python -m venv final_test

# 激活
.\final_test\Scripts\Activate.ps1

# 从PyPI安装（就像普通用户一样）
pip install ConMedRL

# 完整测试
python -c "import ConMedRL; print('✅ ConMedRL version:', ConMedRL.__version__)"
python -c "from ConMedRL import FQE, FQI, TrainDataLoader, ValTestDataLoader; print('✅ Core imports OK')"
python -c "from Data import mimic_iv_icu_discharge, mimic_iv_icu_extubation; print('✅ Data modules OK')"
python -c "from Data import SICdb_discharge, SICdb_extubation; print('✅ All modules OK')"

# 清理
deactivate
Remove-Item -Recurse -Force final_test
```

## 🎯 发布后的任务

### 1. 更新README.md

在README.md的安装部分更新为：

```markdown
### Method 1: PyPI Installation (Recommended)

```bash
# 安装ConMedRL
pip install ConMedRL

# 安装带可选依赖
pip install ConMedRL[viz]      # 可视化工具
pip install ConMedRL[models]   # 模型工具
pip install ConMedRL[dev]      # 开发工具
```

**验证安装:**
```bash
python -c "import ConMedRL; print('ConMedRL version:', ConMedRL.__version__)"
```
```

### 2. 创建GitHub Release

```powershell
# 添加并提交更改
git add .
git commit -m "Release version 1.0.0 - Published to PyPI"

# 创建标签
git tag -a v1.0.0 -m "Version 1.0.0 - Initial PyPI release"

# 推送到GitHub
git push origin main --tags
```

然后在GitHub上：
1. 访问仓库的Releases页面
2. 点击 "Create a new release"
3. 选择标签 `v1.0.0`
4. 标题: `ConMedRL v1.0.0 - Initial PyPI Release`
5. 描述发布内容
6. 发布！

### 3. 添加PyPI徽章到README.md

在README.md顶部添加：

```markdown
[![PyPI version](https://badge.fury.io/py/ConMedRL.svg)](https://badge.fury.io/py/ConMedRL)
[![Python Versions](https://img.shields.io/pypi/pyversions/ConMedRL.svg)](https://pypi.org/project/ConMedRL/)
[![License](https://img.shields.io/pypi/l/ConMedRL.svg)](https://github.com/smt970913/ConMED-RL/blob/main/LICENSE)
[![Downloads](https://pepy.tech/badge/conmedrl)](https://pepy.tech/project/conmedrl)
```

## 🔄 后续版本发布

当需要发布新版本（如修复bug或添加功能）：

### 1. 更新版本号

编辑`ConMedRL/__init__.py`:
```python
__version__ = "1.0.1"  # 或 1.1.0, 2.0.0
```

### 2. 快速发布流程

```powershell
# 清理
Remove-Item -Recurse -Force build, dist, ConMedRL.egg-info -ErrorAction SilentlyContinue

# 构建
python -m build

# 检查
python -m twine check dist/*

# 先测试（推荐）
python -m twine upload --repository testpypi dist/*

# 正式发布
python -m twine upload dist/*

# Git标签
git tag v1.0.1
git push origin main --tags
```

## 📞 遇到问题？

### 常见错误及解决方案

**错误: "The name 'ConMedRL' is already in use"**
- 检查是否之前已经发布过
- 在PyPI上搜索: https://pypi.org/search/?q=ConMedRL

**错误: "File already exists"**
- 版本号已存在，更新`__version__`到新版本

**错误: "Invalid or non-existent authentication"**
- 确认使用`__token__`作为用户名
- 检查Token是否正确复制（包含`pypi-`前缀）
- 确认使用的是对应平台的Token（TestPyPI vs PyPI）

**错误: "Package has invalid metadata"**
- 运行`python -m twine check dist/*`查看详细错误
- 检查README.md中的图片链接
- 确保所有必需字段都已填写

**导入失败**
- 检查`__init__.py`文件中的导入路径
- 确保所有导入的模块都存在
- 运行`python -m py_compile ConMedRL/*.py`检查语法错误

### 获取帮助

- 📚 详细文档: 查看`PYPI_PUBLISHING_GUIDE.md`
- 🌐 PyPI帮助: https://pypi.org/help/
- 📧 联系维护者:
  - Maotong Sun: maotong.sun@tum.de
  - Jingui Xie: jingui.xie@tum.de

---

**祝发布顺利！** 🚀

记住：
- ✅ 总是先在TestPyPI测试
- ✅ 仔细检查版本号
- ✅ PyPI发布是永久的
- ✅ 发布后创建Git标签
