# ClawScope 部署指南

## 快速部署（推荐）

### 方法 1: 一键部署脚本

1. **登录服务器**
```bash
ssh root@159.223.69.253
```

2. **下载并执行部署脚本**
```bash
curl -fsSL https://raw.githubusercontent.com/Jane-o-O-o-O/ClawScope/main/deploy.sh | bash
```

或者手动下载：
```bash
wget https://raw.githubusercontent.com/Jane-o-O-o-O/ClawScope/main/deploy.sh
chmod +x deploy.sh
./deploy.sh
```

3. **配置 API Key**
```bash
# 方法1: 环境变量（推荐）
export OPENAI_API_KEY='your_api_key_here'
echo 'export OPENAI_API_KEY="your_api_key_here"' >> ~/.bashrc

# 方法2: 配置文件
nano ~/.clawscope/config.yaml
# 修改 model.api_key 字段
```

4. **启动服务**
```bash
systemctl start clawscope
systemctl enable clawscope
```

5. **验证部署**
```bash
# 查看服务状态
systemctl status clawscope

# 测试 API
curl http://localhost:8080/health
curl http://localhost:8080/status
```

---

## 手动部署

### 步骤 1: 安装依赖

```bash
# Ubuntu/Debian
apt-get update
apt-get install -y python3 python3-pip python3-venv git curl

# CentOS/RHEL
yum install -y python3 python3-pip git curl
```

### 步骤 2: 克隆项目

```bash
cd /opt
git clone https://github.com/Jane-o-O-o-O/ClawScope.git
cd ClawScope
```

### 步骤 3: 创建虚拟环境

```bash
python3 -m venv venv
source venv/bin/activate
```

### 步骤 4: 安装 Python 包

```bash
# 升级 pip
pip install --upgrade pip

# 安装项目
pip install -e .

# 安装 API 服务器依赖
pip install -e .[api]

# 可选：安装所有功能
# pip install -e .[all]
```

### 步骤 5: 初始化配置

```bash
clawscope init
```

### 步骤 6: 配置 API Key

编辑配置文件：
```bash
nano ~/.clawscope/config.yaml
```

添加 API Key：
```yaml
model:
  provider: openai
  api_key: your_api_key_here
  default_model: gpt-4
```

或使用环境变量：
```bash
export OPENAI_API_KEY='your_api_key_here'
```

### 步骤 7: 启动服务

#### 方式 1: 直接运行（测试用）

```bash
clawscope serve --port 8080
```

#### 方式 2: 使用 systemd（生产环境）

创建服务文件：
```bash
cat > /etc/systemd/system/clawscope.service <<'EOF'
[Unit]
Description=ClawScope AI Agent Platform
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/ClawScope
Environment="PATH=/opt/ClawScope/venv/bin"
Environment="OPENAI_API_KEY=your_key_here"
ExecStart=/opt/ClawScope/venv/bin/clawscope serve --port 8080
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
```

启动服务：
```bash
systemctl daemon-reload
systemctl start clawscope
systemctl enable clawscope
```

#### 方式 3: 使用 Docker（推荐）

```bash
# 构建镜像
docker build -t clawscope:latest .

# 运行容器
docker run -d \
  --name clawscope \
  -p 8080:8080 \
  -e OPENAI_API_KEY=your_key_here \
  -v ~/.clawscope:/root/.clawscope \
  clawscope:latest
```

---

## 配置防火墙

### Ubuntu/Debian (ufw)
```bash
ufw allow 8080/tcp
ufw reload
```

### CentOS/RHEL (firewalld)
```bash
firewall-cmd --permanent --add-port=8080/tcp
firewall-cmd --reload
```

---

## 验证部署

### 1. 检查服务状态
```bash
systemctl status clawscope
```

### 2. 查看日志
```bash
journalctl -u clawscope -f
```

### 3. 测试 API

健康检查：
```bash
curl http://159.223.69.253:8080/health
```

查看状态：
```bash
curl http://159.223.69.253:8080/status
```

聊天测试：
```bash
curl -X POST http://159.223.69.253:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, tell me a joke!"}'
```

### 4. 访问 API 文档

浏览器访问：
- Swagger UI: http://159.223.69.253:8080/docs
- ReDoc: http://159.223.69.253:8080/redoc

---

## 常见问题

### 1. 服务无法启动

检查日志：
```bash
journalctl -u clawscope -n 50
```

常见原因：
- API Key 未配置
- 端口被占用
- Python 依赖缺失

### 2. API 返回 503

原因：ClawScope 未完成初始化

解决：检查日志，等待初始化完成

### 3. 端口无法访问

检查防火墙：
```bash
# 列出开放端口
ufw status
# 或
firewall-cmd --list-ports
```

检查服务监听：
```bash
netstat -tulpn | grep 8080
# 或
ss -tulpn | grep 8080
```

---

## 性能优化

### 1. 使用 Nginx 反向代理

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

### 2. 使用多个 Worker

修改启动命令：
```bash
ExecStart=/opt/ClawScope/venv/bin/uvicorn clawscope.server:api \
  --host 0.0.0.0 \
  --port 8080 \
  --workers 4
```

### 3. 启用缓存

在配置文件中添加：
```yaml
tools:
  cache_enabled: true
  cache_ttl: 3600
```

---

## 安全建议

1. **使用 HTTPS**
   - 配置 SSL 证书
   - 使用 Let's Encrypt

2. **API Key 安全**
   - 不要在代码中硬编码
   - 使用环境变量或密钥管理系统

3. **访问控制**
   - 配置防火墙规则
   - 使用 API 认证

4. **定期更新**
   ```bash
   cd /opt/ClawScope
   git pull
   pip install -e . --upgrade
   systemctl restart clawscope
   ```

---

## 卸载

```bash
# 停止服务
systemctl stop clawscope
systemctl disable clawscope

# 删除服务文件
rm /etc/systemd/system/clawscope.service
systemctl daemon-reload

# 删除项目文件
rm -rf /opt/ClawScope
rm -rf ~/.clawscope

# 卸载 Python 包
pip uninstall clawscope
```

---

## 技术支持

- GitHub Issues: https://github.com/Jane-o-O-o-O/ClawScope/issues
- 文档: https://github.com/Jane-o-O-o-O/ClawScope/blob/main/README.md
