# ConMedRL PyPI 发布指南

本指南将帮助您一步步将ConMedRL包发布到PyPI (Python Package Index)。

## 📋 发布前准备清单

### 1. 账户准备

#### 1.1 创建PyPI账户
- 访问 [PyPI官网](https://pypi.org/) 并注册账户
- 访问 [TestPyPI](https://test.pypi.org/) 并注册测试账户（推荐先在测试环境发布）
- 验证邮箱地址

#### 1.2 启用双因素认证（2FA）
- 登录PyPI账户
- 进入 Account Settings → Two Factor Authentication
- 选择认证方式（推荐使用认证器应用如Google Authenticator）
- 保存恢复代码到安全位置

#### 1.3 生成API Token
为了安全发布，建议使用API Token而非密码：

**对于TestPyPI：**
1. 登录 https://test.pypi.org/
2. 进入 Account Settings → API tokens
3. 点击 "Add API token"
4. Token名称: `ConMedRL-upload`
5. Scope: 选择 "Entire account" （第一次）或特定项目
6. **立即复制并保存Token**（只显示一次！）

**对于正式PyPI：**
1. 登录 https://pypi.org/
2. 重复上述步骤

### 2. 安装必要工具

```bash
# 升级pip
python -m pip install --upgrade pip

# 安装构建工具
pip install --upgrade build

# 安装上传工具
pip install --upgrade twine

# 安装测试工具（可选）
pip install --upgrade wheel setuptools
```

### 3. 验证包配置

检查以下文件是否正确配置：
- ✅ `setup.py` - 包含完整的包信息
- ✅ `pyproject.toml` - 现代Python包配置
- ✅ `README.md` - 项目说明文档
- ✅ `LICENSE` - MIT许可证
- ✅ `requirements.txt` - 依赖列表
- ✅ `MANIFEST.in` - 文件包含规则
- ✅ `ConMedRL/__init__.py` - 包含版本信息
- ✅ `Data/__init__.py` - 数据模块初始化

## 🔧 发布流程

### 步骤1: 清理旧的构建文件

```bash
# 使用build_package.py脚本
python build_package.py clean

# 或手动清理
rmdir /s /q build dist *.egg-info  # Windows
rm -rf build dist *.egg-info       # Linux/Mac
```

### 步骤2: 构建包

```bash
# 使用build_package.py（推荐）
python build_package.py build

# 或手动构建
python -m build
```

构建完成后，`dist/`目录下应该有两个文件：
- `ConMedRL-1.0.0.tar.gz` (源代码分发)
- `ConMedRL-1.0.0-py3-none-any.whl` (wheel分发)

### 步骤3: 检查包

```bash
# 使用build_package.py
python build_package.py check

# 或手动检查
python -m twine check dist/*
```

确保输出显示：
```
Checking dist/ConMedRL-1.0.0.tar.gz: PASSED
Checking dist/ConMedRL-1.0.0-py3-none-any.whl: PASSED
```

### 步骤4: 本地测试安装

```bash
# 使用build_package.py
python build_package.py test

# 或手动测试
python -m venv test_env
# Windows:
test_env\Scripts\activate
# Linux/Mac:
source test_env/bin/activate

pip install dist/ConMedRL-1.0.0-py3-none-any.whl
python -c "import ConMedRL; print(ConMedRL.__version__)"
deactivate
rmdir /s /q test_env  # Windows
rm -rf test_env       # Linux/Mac
```

### 步骤5: 上传到TestPyPI（强烈推荐）

首先在测试环境发布，确保一切正常：

```bash
# 使用build_package.py
python build_package.py testpypi

# 或手动上传
python -m twine upload --repository testpypi dist/*
```

系统会提示输入凭证：
- Username: `__token__`
- Password: 您的TestPyPI API Token（以`pypi-`开头）

上传成功后，您会看到包的URL：
```
https://test.pypi.org/project/ConMedRL/
```

### 步骤6: 从TestPyPI测试安装

```bash
# 创建新的测试环境
python -m venv testpypi_env
testpypi_env\Scripts\activate  # Windows
# source testpypi_env/bin/activate  # Linux/Mac

# 从TestPyPI安装
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ ConMedRL

# 测试导入
python -c "import ConMedRL; print('Version:', ConMedRL.__version__)"
python -c "from ConMedRL import FQE, FQI; print('Import successful!')"

# 清理
deactivate
rmdir /s /q testpypi_env  # Windows
```

⚠️ **注意**: 使用`--extra-index-url https://pypi.org/simple/`是因为ConMedRL的依赖包（如torch, pandas等）在TestPyPI上可能不存在，需要从正式PyPI安装。

### 步骤7: 发布到正式PyPI

⚠️ **重要警告**: 
- PyPI上传是**永久性**的，无法删除特定版本
- 确保已在TestPyPI充分测试
- 仔细检查版本号（发布后无法修改）

```bash
# 使用build_package.py
python build_package.py pypi

# 或手动上传
python -m twine upload dist/*
```

系统会提示输入凭证：
- Username: `__token__`
- Password: 您的PyPI API Token（以`pypi-`开头）

成功后访问：
```
https://pypi.org/project/ConMedRL/
```

### 步骤8: 从PyPI验证安装

```bash
# 创建新环境
python -m venv pypi_test_env
pypi_test_env\Scripts\activate  # Windows

# 从PyPI安装
pip install ConMedRL

# 验证
python -c "import ConMedRL; print('ConMedRL version:', ConMedRL.__version__)"
python -c "from ConMedRL import FQE, FQI, TrainDataLoader; print('Success!')"
python -c "from Data import mimic_iv_icu_discharge; print('Data module OK!')"

# 清理
deactivate
rmdir /s /q pypi_test_env
```

## 📝 版本管理

### 更新版本号

当需要发布新版本时：

1. **更新版本号**在 `ConMedRL/__init__.py`:
   ```python
   __version__ = "1.0.1"  # 或 "1.1.0", "2.0.0" 等
   ```

2. **版本命名规范**（遵循语义化版本 Semantic Versioning）:
   - `1.0.0` → `1.0.1`: 修复bug（补丁版本）
   - `1.0.0` → `1.1.0`: 添加新功能（次要版本）
   - `1.0.0` → `2.0.0`: 破坏性更改（主要版本）

3. **创建Git标签**:
   ```bash
   git add ConMedRL/__init__.py
   git commit -m "Bump version to 1.0.1"
   git tag v1.0.1
   git push origin main --tags
   ```

4. **重新构建和发布**:
   ```bash
   python build_package.py clean
   python build_package.py build
   python build_package.py check
   python build_package.py testpypi  # 先测试
   python build_package.py pypi       # 正式发布
   ```

## 🔐 安全最佳实践

### 1. 保护API Token

**Windows** - 使用环境变量：
```powershell
# 设置环境变量
setx TWINE_USERNAME "__token__"
setx TWINE_PASSWORD "pypi-your-token-here"

# 使用环境变量上传
python -m twine upload dist/*
```

**Linux/Mac** - 使用.pypirc文件：
```bash
# 创建配置文件
cat > ~/.pypirc << EOF
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-your-production-token

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-your-test-token
EOF

# 设置权限
chmod 600 ~/.pypirc
```

### 2. 使用项目范围的Token

首次上传后，为ConMedRL项目创建专用Token：

1. 访问 https://pypi.org/manage/project/ConMedRL/settings/
2. 创建新的API token
3. Scope: "Project: ConMedRL"
4. 更新.pypirc或环境变量

## 🚨 常见问题解决

### 问题1: 包名已存在
```
ERROR: The name 'ConMedRL' is already in use.
```
**解决方案**: 
- 检查是否已发布过
- 考虑使用不同的包名（如`ConMedRL-Toolkit`）
- 在setup.py和pyproject.toml中修改`name`字段

### 问题2: 版本号已存在
```
ERROR: File already exists.
```
**解决方案**:
- PyPI不允许覆盖已存在的版本
- 更新版本号到新版本
- 修改`ConMedRL/__init__.py`中的`__version__`

### 问题3: README渲染错误
```
WARNING: The description is not a valid Markdown.
```
**解决方案**:
- 使用`twine check dist/*`查看详细错误
- 检查README.md中的图片链接（使用绝对URL）
- 确保markdown语法正确

### 问题4: 依赖包版本冲突
```
ERROR: Could not find a version that satisfies the requirement...
```
**解决方案**:
- 检查`requirements.txt`和`pyproject.toml`中的依赖版本
- 使用更宽松的版本约束（如`numpy>=1.20.0`而非`numpy==1.23.5`）

### 问题5: 包导入失败
```python
ImportError: No module named 'ConMedRL'
```
**解决方案**:
- 检查`setup.py`中的`packages`参数
- 确保包含`find_packages()`或明确列出包
- 验证`__init__.py`文件存在

## 📊 发布后清单

- [ ] 访问PyPI项目页面确认信息正确
- [ ] 测试从PyPI安装包
- [ ] 更新README.md中的安装说明
- [ ] 创建GitHub Release和Tag
- [ ] 在项目主页添加PyPI徽章：
  ```markdown
  [![PyPI version](https://badge.fury.io/py/ConMedRL.svg)](https://badge.fury.io/py/ConMedRL)
  [![Downloads](https://pepy.tech/badge/conmedrl)](https://pepy.tech/project/conmedrl)
  ```
- [ ] 通知用户和社区
- [ ] 更新文档中的安装说明

## 🔄 持续维护

### 定期更新
- 响应用户问题和反馈
- 修复bug并发布补丁版本
- 添加新功能并发布次要版本
- 保持依赖包更新

### 文档维护
- 保持README.md最新
- 更新CHANGELOG.md记录版本变化
- 维护使用示例和教程

## 📚 参考资源

- [PyPI官方文档](https://packaging.python.org/)
- [Twine文档](https://twine.readthedocs.io/)
- [Semantic Versioning](https://semver.org/)
- [Python打包用户指南](https://packaging.python.org/tutorials/packaging-projects/)

---

**祝您发布顺利！** 🎉

如有问题，请联系：
- Maotong Sun: maotong.sun@tum.de
- Jingui Xie: jingui.xie@tum.de
