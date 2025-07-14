# 🚀 ConMED-RL Docker 快速入门指南

## 🎯 你现在可以做什么？

是的！现在你可以在另一台电脑的Docker中：
- ✅ **直接调用** `Data` 和 `ConMedRL` 里面的所有函数
- ✅ **使用** CDM-Software 的临床决策支持系统
- ✅ **打开和运行** Experiment Notebook 中的所有Jupyter笔记本
- ✅ **访问** 完整的开发和研究环境

## 📋 三种部署模式

### 1. 🔬 研究环境（推荐用于数据分析）

**一键启动：**
```bash
# Linux/Mac
cd Docker-Deployment
chmod +x scripts/build_research.sh
./scripts/build_research.sh

# Windows
cd Docker-Deployment
scripts\build_research.bat
```

**你可以使用：**
- 🌟 **Jupyter Lab**: http://localhost:8888 （密码：`conmed-rl-research`）
- 📊 **所有数据分析工具**: matplotlib, seaborn, plotly
- 🧠 **完整的ConMedRL框架**: 直接在notebook中import
- 📁 **所有项目文件**: 包括Experiment Notebook
- 🔧 **Flask应用**: http://localhost:5000 （可选）

### 2. 💻 开发环境（用于全栈开发）

```bash
cd Docker-Deployment
docker-compose -f docker-compose.dev.yml up --build -d
```

**你可以使用：**
- 🌟 **Jupyter Lab**: http://localhost:8888 （密码：`conmed-rl-dev`）
- 🌐 **Flask Web应用**: http://localhost:5000
- 💾 **数据库**: PostgreSQL (localhost:5432)

### 3. 🚀 生产环境（用于部署）

```bash
cd Docker-Deployment
docker-compose -f docker-compose.prod.yml up --build -d
```

**你可以使用：**
- 🌐 **Web应用**: http://localhost
- 📊 **监控**: Prometheus + Grafana

## 🎓 实际使用示例

### 在Jupyter中使用ConMedRL

```python
# 在Jupyter notebook中
import sys
sys.path.append('/app')

# 导入核心模块
from ConMedRL.conmedrl import FQI, FQE
from ConMedRL.data_loader import DataLoader

# 导入数据处理模块
from Data.mimic_iv_icu_discharge.data_preprocess import preprocess_data

# 使用示例
data_loader = DataLoader()
fqi_agent = FQI()
fqe_agent = FQE()

# 数据预处理
processed_data = preprocess_data('/app/Data/raw_data.csv')
```

### 运行Experiment Notebook

```python
# 直接在Jupyter Lab中打开
# /app/Experiment Notebook/Case_ICU_Discharge_Decision_Making.ipynb
# /app/Experiment Notebook/Case_ICU_Extubation_Decision_Making.ipynb
# /app/Experiment Notebook/Example_dataset_preprocess_MIMIC-IV.ipynb
```

### 使用CDM-Software

```python
# 启动临床决策支持系统
from CDM_Software.web_application_demo import app
app.run(host='0.0.0.0', port=5000)

# 或者直接访问 http://localhost:5000
```

## 📁 Docker中的文件结构

```
/app/
├── ConMedRL/                    # 核心OCRL框架
│   ├── conmedrl.py             # 主要算法实现
│   ├── conmedrl_continuous.py  # 连续动作空间
│   └── data_loader.py          # 数据加载器
├── Data/                        # 数据处理模块
│   ├── mimic_iv_icu_discharge/
│   ├── mimic_iv_icu_extubation/
│   └── SICdb_*/
├── CDM-Software/                # 临床决策支持软件
│   ├── web_application_demo.py
│   └── interactive_support.py
├── Experiment Notebook/         # Jupyter笔记本
│   ├── Case_ICU_Discharge_Decision_Making.ipynb
│   ├── Case_ICU_Extubation_Decision_Making.ipynb
│   └── Example_dataset_preprocess_MIMIC-IV.ipynb
└── Software_FQE_models/         # 训练好的模型
    ├── discharge_decision_making/
    └── extubation_decision_making/
```

## 🔧 常用命令

### 环境管理
```bash
# 启动研究环境
docker-compose -f docker-compose.research.yml up -d

# 查看日志
docker-compose -f docker-compose.research.yml logs -f

# 停止环境
docker-compose -f docker-compose.research.yml down

# 进入容器
docker-compose -f docker-compose.research.yml exec conmed-rl-research bash
```

### 健康检查
```bash
# 检查Jupyter Lab
curl -f http://localhost:8888/lab

# 检查Flask应用
curl -f http://localhost:5000/health

# 检查容器状态
docker-compose -f docker-compose.research.yml ps
```

## 💡 最佳实践

### 1. 数据分析工作流
```bash
# 1. 启动研究环境
./scripts/build_research.sh

# 2. 打开Jupyter Lab
# 访问 http://localhost:8888

# 3. 创建新的notebook或打开现有的
# 导入所需模块并开始分析
```

### 2. 开发工作流
```bash
# 1. 启动开发环境
docker-compose -f docker-compose.dev.yml up -d

# 2. 同时使用Jupyter和Flask
# Jupyter: http://localhost:8888
# Flask: http://localhost:5000

# 3. 实时调试和测试
```

### 3. 部署工作流
```bash
# 1. 在研究环境中完成开发
# 2. 在开发环境中测试
# 3. 部署到生产环境
docker-compose -f docker-compose.prod.yml up -d
```

## 🆘 常见问题解决

### 端口冲突
```bash
# 检查端口占用
netstat -tulpn | grep :8888

# 修改端口（在docker-compose文件中）
ports:
  - "8889:8888"  # 改为8889
```

### 导入模块失败
```python
# 在Jupyter中添加路径
import sys
sys.path.append('/app')

# 验证路径
print(sys.path)

# 检查文件是否存在
import os
os.listdir('/app/ConMedRL')
```

### Jupyter访问问题
```bash
# 检查token
docker-compose -f docker-compose.research.yml exec conmed-rl-research jupyter lab list

# 重启Jupyter
docker-compose -f docker-compose.research.yml restart
```

## 📞 获取帮助

1. 查看完整文档：`Docker-Deployment/README.md`
2. 运行验证测试：`./scripts/test_deployment.sh`
3. 查看故障排除：`Docker-Deployment/DOCKER_VALIDATION_GUIDE.md`
4. 联系维护者：maotong.sun@tum.de, jingui.xie@tum.de

## 🎉 开始使用

现在你可以：
1. 选择适合的环境模式
2. 运行相应的启动脚本
3. 在浏览器中访问Jupyter Lab
4. 开始你的ConMED-RL研究之旅！

**推荐首次使用：**
```bash
cd Docker-Deployment
chmod +x scripts/build_research.sh
./scripts/build_research.sh
```

然后访问 http://localhost:8888 开始使用！ 