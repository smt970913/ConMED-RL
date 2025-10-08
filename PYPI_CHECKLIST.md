# ConMedRL PyPI 发布检查清单

## 📋 发布前自检清单

### ✅ 已完成的准备工作（由AI助手完成）

- [x] **修复包导入错误**
  - 修正`ConMedRL/__init__.py`中的导入路径
  - 移除不存在的模块导入
  
- [x] **清理入口点配置**
  - 从`setup.py`移除不存在的console_scripts
  - 从`pyproject.toml`移除不存在的project.scripts
  
- [x] **验证包结构**
  - ConMedRL模块: ✅ 完整
  - Data模块: ✅ 完整
  - 版本信息: ✅ 1.0.0

### 📝 您需要完成的步骤

#### 第一阶段：环境准备

- [ ] **安装必要工具**
  ```powershell
  pip install --upgrade pip build twine wheel setuptools
  ```

- [ ] **验证本地导入**
  ```powershell
  python -c "import ConMedRL; print(ConMedRL.__version__)"
  ```

#### 第二阶段：账户设置

- [ ] **注册TestPyPI账户** (https://test.pypi.org/account/register/)
- [ ] **注册PyPI账户** (https://pypi.org/account/register/)
- [ ] **启用2FA认证**（两个平台都需要）
- [ ] **生成TestPyPI API Token**
  - 保存格式: `pypi-AgEIcHlwaS5vcmc...`
- [ ] **生成PyPI API Token**
  - 保存格式: `pypi-AgEIcHlwaS5vcmc...`

#### 第三阶段：构建和测试

- [ ] **清理旧构建**
  ```powershell
  Remove-Item -Recurse -Force build, dist, ConMedRL.egg-info -ErrorAction SilentlyContinue
  ```

- [ ] **构建包**
  ```powershell
  python -m build
  ```

- [ ] **检查包质量**
  ```powershell
  python -m twine check dist/*
  ```
  期望: 所有检查都PASSED

- [ ] **本地测试安装**
  ```powershell
  python -m venv test_env
  .\test_env\Scripts\Activate.ps1
  pip install dist/ConMedRL-1.0.0-py3-none-any.whl
  python -c "import ConMedRL; print(ConMedRL.__version__)"
  deactivate
  Remove-Item -Recurse -Force test_env
  ```

#### 第四阶段：TestPyPI发布（强烈推荐）

- [ ] **上传到TestPyPI**
  ```powershell
  python -m twine upload --repository testpypi dist/*
  ```
  - Username: `__token__`
  - Password: [您的TestPyPI token]

- [ ] **从TestPyPI安装测试**
  ```powershell
  python -m venv testpypi_test
  .\testpypi_test\Scripts\Activate.ps1
  pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ ConMedRL
  python -c "import ConMedRL; from ConMedRL import FQE, FQI; print('Success!')"
  deactivate
  Remove-Item -Recurse -Force testpypi_test
  ```

- [ ] **检查TestPyPI项目页面**
  - 访问: https://test.pypi.org/project/ConMedRL/
  - 验证: README显示正确
  - 验证: 元数据完整
  - 验证: 依赖列表正确

#### 第五阶段：正式PyPI发布

- [ ] **最后确认**
  - 版本号正确: 1.0.0
  - 包名可用: ConMedRL
  - 所有测试通过
  - README无误
  - LICENSE正确

- [ ] **上传到PyPI**
  ```powershell
  python -m twine upload dist/*
  ```
  - Username: `__token__`
  - Password: [您的PyPI token]

- [ ] **验证PyPI页面**
  - 访问: https://pypi.org/project/ConMedRL/
  - 检查所有信息

- [ ] **最终安装测试**
  ```powershell
  python -m venv final_test
  .\final_test\Scripts\Activate.ps1
  pip install ConMedRL
  python -c "import ConMedRL; print('Version:', ConMedRL.__version__)"
  deactivate
  Remove-Item -Recurse -Force final_test
  ```

#### 第六阶段：发布后工作

- [ ] **更新README.md**
  - 添加PyPI安装说明
  - 添加PyPI徽章

- [ ] **创建Git标签和Release**
  ```powershell
  git add .
  git commit -m "Release v1.0.0 - Published to PyPI"
  git tag -a v1.0.0 -m "Version 1.0.0 - Initial PyPI release"
  git push origin main --tags
  ```

- [ ] **在GitHub创建Release**
  - 使用标签: v1.0.0
  - 附加发布说明

- [ ] **通知用户和社区**

## 🎯 配置文件摘要

### setup.py
- ✅ Package name: ConMedRL
- ✅ Version: 从`ConMedRL.__version__`读取
- ✅ Packages: `ConMedRL`, `Data`及其子包
- ✅ Requirements: 从`requirements.txt`读取
- ✅ Python requirement: >=3.8
- ✅ License: MIT

### pyproject.toml
- ✅ Build system: setuptools
- ✅ Project metadata: 完整
- ✅ Dependencies: 已定义
- ✅ Optional dependencies: dev, models, viz
- ✅ Project URLs: GitHub链接

### MANIFEST.in
- ✅ 包含: README.md, LICENSE, requirements.txt
- ✅ 包含: ConMedRL和Data的所有.py文件
- ✅ 排除: __pycache__, 开发文件, 实验文件

## ⚠️ 重要提醒

1. **PyPI发布是永久的**
   - 无法删除或修改已发布的版本
   - 只能发布新版本

2. **版本号管理**
   - 遵循语义化版本(Semantic Versioning)
   - 修改`ConMedRL/__init__.py`中的`__version__`

3. **安全性**
   - 不要在代码中硬编码API token
   - 不要提交.pypirc到Git
   - 使用Token而非密码

4. **测试TestPyPI**
   - 总是先在TestPyPI测试
   - TestPyPI可以重复上传相同版本

## 📞 获取帮助

- **详细指南**: 查看`PYPI_PUBLISHING_GUIDE.md`
- **快速开始**: 查看`PYPI_QUICK_START.md`
- **PyPI帮助**: https://pypi.org/help/
- **联系维护者**:
  - Maotong Sun: maotong.sun@tum.de
  - Jingui Xie: jingui.xie@tum.de

---

**当前状态**: 准备就绪，可以开始发布流程！ 🚀
