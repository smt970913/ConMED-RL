# ConMedRL PyPI 发布快速参考

## 🎯 一分钟快速发布

```powershell
# 1. 安装工具
pip install --upgrade build twine

# 2. 清理 + 构建
Remove-Item -Recurse -Force build, dist, *.egg-info -ErrorAction SilentlyContinue
python -m build

# 3. 检查
python -m twine check dist/*

# 4. 测试发布 (TestPyPI)
python -m twine upload --repository testpypi dist/*

# 5. 正式发布 (PyPI)
python -m twine upload dist/*
```

## 📋 必备信息

### 账户链接
- **TestPyPI注册**: https://test.pypi.org/account/register/
- **PyPI注册**: https://pypi.org/account/register/
- **TestPyPI Token**: https://test.pypi.org/manage/account/#api-tokens
- **PyPI Token**: https://pypi.org/manage/account/token/

### 上传凭证
- Username: `__token__`
- Password: 您的API Token (以`pypi-`开头)

## 🔧 常用命令

### 构建相关
```powershell
# 清理构建文件
Remove-Item -Recurse -Force build, dist, ConMedRL.egg-info -ErrorAction SilentlyContinue

# 构建包
python -m build

# 检查包
python -m twine check dist/*

# 查看生成的文件
Get-ChildItem dist
```

### 测试安装
```powershell
# 本地测试
python -m venv test_env
.\test_env\Scripts\Activate.ps1
pip install dist/ConMedRL-1.0.0-py3-none-any.whl
python -c "import ConMedRL; print(ConMedRL.__version__)"
deactivate
Remove-Item -Recurse -Force test_env

# 从TestPyPI测试
python -m venv testpypi_env
.\testpypi_env\Scripts\Activate.ps1
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ ConMedRL
python -c "import ConMedRL; print(ConMedRL.__version__)"
deactivate
Remove-Item -Recurse -Force testpypi_env

# 从PyPI测试
python -m venv pypi_env
.\pypi_env\Scripts\Activate.ps1
pip install ConMedRL
python -c "import ConMedRL; print(ConMedRL.__version__)"
deactivate
Remove-Item -Recurse -Force pypi_env
```

### 上传相关
```powershell
# 上传到TestPyPI
python -m twine upload --repository testpypi dist/*

# 上传到PyPI
python -m twine upload dist/*

# 使用环境变量（避免每次输入）
$env:TWINE_USERNAME = "__token__"
$env:TWINE_PASSWORD = "pypi-your-token-here"
python -m twine upload dist/*
```

### Git版本管理
```powershell
# 提交更改
git add .
git commit -m "Release v1.0.0"

# 创建标签
git tag -a v1.0.0 -m "Version 1.0.0 - Initial PyPI release"

# 推送到GitHub
git push origin main --tags
```

## 🔢 版本号更新

编辑 `ConMedRL/__init__.py`:
```python
__version__ = "1.0.1"  # 修改这里
```

版本号规范:
- `1.0.0` → `1.0.1`: Bug修复
- `1.0.0` → `1.1.0`: 新功能
- `1.0.0` → `2.0.0`: 破坏性更改

## 📁 项目URL

### 发布后
- **PyPI项目页**: https://pypi.org/project/ConMedRL/
- **TestPyPI项目页**: https://test.pypi.org/project/ConMedRL/

### GitHub
- **仓库**: https://github.com/smt970913/ConMED-RL
- **Issues**: https://github.com/smt970913/ConMED-RL/issues

## 📊 PyPI徽章

添加到README.md顶部：
```markdown
[![PyPI version](https://badge.fury.io/py/ConMedRL.svg)](https://badge.fury.io/py/ConMedRL)
[![Python Versions](https://img.shields.io/pypi/pyversions/ConMedRL.svg)](https://pypi.org/project/ConMedRL/)
[![License](https://img.shields.io/pypi/l/ConMedRL.svg)](https://github.com/smt970913/ConMED-RL/blob/main/LICENSE)
[![Downloads](https://pepy.tech/badge/conmedrl)](https://pepy.tech/project/conmedrl)
```

## 🐛 快速故障排除

### "The name 'ConMedRL' is already in use"
→ 包名已被占用或已发布

### "File already exists"
→ 版本号已存在，需要更新版本号

### "Invalid authentication"
→ 检查：
1. Username是`__token__`（双下划线）
2. Token完整复制（含`pypi-`前缀）
3. 使用对应平台的Token

### "Package check FAILED"
→ 运行：`python -m twine check dist/*` 查看详细错误

### ImportError after install
→ 检查：
1. `ConMedRL/__init__.py`的导入
2. 所有子模块的`__init__.py`存在
3. `setup.py`中的packages配置

## 📚 文档快速链接

- **详细指南**: `PYPI_PUBLISHING_GUIDE.md`
- **快速开始**: `PYPI_QUICK_START.md`
- **检查清单**: `PYPI_CHECKLIST.md`
- **发布总结**: `PYPI_RELEASE_SUMMARY.md`

## ✅ 发布前检查

- [ ] 安装了build和twine
- [ ] 注册了PyPI和TestPyPI账户
- [ ] 启用了2FA
- [ ] 生成了API Tokens
- [ ] 本地测试通过
- [ ] 版本号正确
- [ ] README.md无误

## 📞 获取帮助

- **PyPI帮助**: https://pypi.org/help/
- **打包指南**: https://packaging.python.org/
- **联系维护者**:
  - Maotong Sun: maotong.sun@tum.de
  - Jingui Xie: jingui.xie@tum.de

---

**提示**: 将本文档加入收藏，每次发布新版本时参考使用！ ⭐
