# Docker部署验证指南

本指南用于在另一台计算机上验证ConMED-RL项目的Docker配置是否正确。

## 📋 前置条件检查

### 1. 环境要求
- Docker Engine 20.10+
- Docker Compose 1.29+
- Git
- curl (用于健康检查)

### 2. 环境验证
```bash
# 检查Docker版本
docker --version

# 检查Docker Compose版本
docker-compose --version

# 检查Git
git --version

# 检查curl
curl --version
```

## 🚀 快速验证流程

### 方法1: 使用自动化测试脚本（推荐）

```bash
# 克隆仓库
git clone https://github.com/your-username/ICU-Decision-Making-OCRL.git
cd ICU-Decision-Making-OCRL/Docker-Deployment

# Linux/Mac
chmod +x scripts/test_deployment.sh
./scripts/test_deployment.sh

# Windows
scripts\test_deployment.bat
```

### 方法2: 手动验证步骤

#### 步骤1: 验证配置文件
```bash
cd ICU-Decision-Making-OCRL/Docker-Deployment

# 验证docker-compose语法
docker-compose config

# 验证生产环境配置
docker-compose -f docker-compose.prod.yml config
```

#### 步骤2: 测试开发环境
```bash
# 构建并启动开发环境
docker-compose up --build -d

# 检查容器状态
docker-compose ps

# 检查健康状态
curl -f http://localhost:5000/health

# 查看日志
docker-compose logs conmed-rl-app

# 停止开发环境
docker-compose down
```

#### 步骤3: 测试生产环境
```bash
# 启动生产环境
docker-compose -f docker-compose.prod.yml up --build -d

# 检查容器状态
docker-compose -f docker-compose.prod.yml ps

# 检查健康状态
curl -f http://localhost/health

# 查看日志
docker-compose -f docker-compose.prod.yml logs -f

# 停止生产环境
docker-compose -f docker-compose.prod.yml down
```

#### 步骤4: 测试监控栈（可选）
```bash
# 启动完整监控栈
docker-compose -f docker-compose.prod.yml --profile monitoring up -d

# 检查监控服务
curl -f http://localhost:9090    # Prometheus
curl -f http://localhost:3000    # Grafana

# 清理
docker-compose -f docker-compose.prod.yml --profile monitoring down
```

## ✅ 验证检查清单

### 配置验证
- [ ] `docker-compose.yml` 语法正确
- [ ] `docker-compose.prod.yml` 语法正确
- [ ] `Dockerfile` 构建成功
- [ ] 所有服务名称一致（conmed-rl-*）

### 功能验证
- [ ] 开发环境启动成功
- [ ] 生产环境启动成功
- [ ] 健康检查端点响应正常
- [ ] 日志输出正常
- [ ] 端口映射正确

### 服务验证
- [ ] 主应用服务运行正常
- [ ] Nginx代理工作正常
- [ ] 监控服务可访问（如启用）

## 🔧 故障排除指南

### 常见问题及解决方案

#### 1. 端口冲突
```bash
# 错误: port is already allocated
# 解决: 检查端口占用
netstat -tulpn | grep :5000
netstat -tulpn | grep :80

# 或修改端口映射
# 在docker-compose.yml中修改ports配置
```

#### 2. 镜像构建失败
```bash
# 错误: Build failed
# 解决: 清理Docker缓存
docker system prune -a
docker-compose build --no-cache
```

#### 3. 容器启动失败
```bash
# 检查容器日志
docker-compose logs conmed-rl-app

# 检查容器状态
docker-compose ps

# 进入容器调试
docker-compose exec conmed-rl-app /bin/bash
```

#### 4. 健康检查失败
```bash
# 检查应用是否正常启动
docker-compose logs conmed-rl-app

# 手动测试端点
curl -v http://localhost:5000/health

# 检查应用依赖
docker-compose exec conmed-rl-app python -c "import flask; print('Flask OK')"
```

#### 5. 网络连接问题
```bash
# 检查Docker网络
docker network ls

# 检查服务间连接
docker-compose exec conmed-rl-app ping nginx
```

## 📊 性能验证

### 资源使用检查
```bash
# 监控容器资源使用
docker stats

# 检查内存使用
docker-compose exec conmed-rl-app free -h

# 检查磁盘使用
docker system df
```

### 负载测试（可选）
```bash
# 简单负载测试
for i in {1..100}; do curl -s http://localhost:5000/health; done

# 使用ab工具
ab -n 100 -c 10 http://localhost:5000/health
```

## 🔐 安全验证

### 安全配置检查
- [ ] 非root用户运行
- [ ] 最小权限原则
- [ ] 敏感信息不在镜像中
- [ ] 网络隔离正确

### 安全扫描（可选）
```bash
# 扫描镜像漏洞
docker scan conmed-rl-app

# 检查容器权限
docker inspect conmed-rl-app | grep -i user
```

## 📝 验证报告模板

### 环境信息
- OS: `uname -a`
- Docker版本: `docker --version`
- Docker Compose版本: `docker-compose --version`

### 测试结果
- [ ] 配置验证通过
- [ ] 构建成功
- [ ] 开发环境部署成功
- [ ] 生产环境部署成功
- [ ] 健康检查通过
- [ ] 性能正常
- [ ] 安全检查通过

### 问题记录
- 遇到的问题及解决方案
- 性能观察
- 改进建议

## 📞 支持信息

如果验证过程中遇到问题：
1. 查看日志: `docker-compose logs -f`
2. 检查GitHub Issues
3. 联系维护者:
   - maotong.sun@tum.de
   - jingui.xie@tum.de

## 🎯 最佳实践

1. **定期验证**: 每次更新后都进行验证
2. **环境一致性**: 确保测试环境与生产环境一致
3. **文档更新**: 及时更新配置文档
4. **备份策略**: 重要数据定期备份
5. **监控告警**: 设置适当的监控和告警 