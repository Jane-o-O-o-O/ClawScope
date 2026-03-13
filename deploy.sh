#!/bin/bash
# ClawScope 一键部署脚本
# 用法: bash deploy.sh

set -e

echo "======================================"
echo "ClawScope 服务器部署脚本"
echo "======================================"

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 配置
PROJECT_NAME="ClawScope"
REPO_URL="https://github.com/Jane-o-O-o-O/ClawScope.git"
INSTALL_DIR="/opt/clawscope"
SERVICE_PORT=8080
PYTHON_VERSION="3.10"

echo -e "${GREEN}[1/7] 检查系统环境...${NC}"
# 检测操作系统
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$NAME
    echo "操作系统: $OS"
else
    echo -e "${RED}无法检测操作系统${NC}"
    exit 1
fi

echo -e "${GREEN}[2/7] 安装系统依赖...${NC}"
# 安装基础依赖
if [[ "$OS" == *"Ubuntu"* ]] || [[ "$OS" == *"Debian"* ]]; then
    apt-get update
    apt-get install -y python3 python3-pip python3-venv git curl
elif [[ "$OS" == *"CentOS"* ]] || [[ "$OS" == *"Red Hat"* ]]; then
    yum install -y python3 python3-pip git curl
else
    echo -e "${YELLOW}未知系统，尝试继续...${NC}"
fi

echo -e "${GREEN}[3/7] 克隆项目代码...${NC}"
# 克隆或更新代码
if [ -d "$INSTALL_DIR" ]; then
    echo "目录已存在，更新代码..."
    cd "$INSTALL_DIR"
    git pull
else
    echo "克隆代码..."
    git clone "$REPO_URL" "$INSTALL_DIR"
    cd "$INSTALL_DIR"
fi

echo -e "${GREEN}[4/7] 创建 Python 虚拟环境...${NC}"
# 创建虚拟环境
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# 激活虚拟环境
source venv/bin/activate

echo -e "${GREEN}[5/7] 安装 Python 依赖...${NC}"
# 升级 pip
pip install --upgrade pip

# 安装项目依赖
pip install -e .
pip install -e .[api]

echo -e "${GREEN}[6/7] 创建配置文件...${NC}"
# 创建配置目录
mkdir -p ~/.clawscope/workspace/{sessions,memory,skills,knowledge,logs}

# 创建配置文件（如果不存在）
if [ ! -f ~/.clawscope/config.yaml ]; then
    cat > ~/.clawscope/config.yaml <<EOF
# ClawScope 配置文件
project: ClawScope
workspace: $(echo ~)/.clawscope/workspace

model:
  provider: openai
  default_model: gpt-4
  # API Key 将在启动时从环境变量读取
  # 请设置: export OPENAI_API_KEY=your_key_here

agent:
  type: react
  name: ClawScope
  max_iterations: 40

channels:
  telegram:
    enabled: false
  discord:
    enabled: false

services:
  cron_enabled: true
  heartbeat_enabled: true
  heartbeat_interval: 1800

tracing:
  enabled: false

tools:
  sandbox_enabled: false
  enabled:
    - read_file
    - write_file
    - execute_shell
    - web_search
    - web_fetch
EOF
    echo -e "${YELLOW}配置文件已创建: ~/.clawscope/config.yaml${NC}"
    echo -e "${YELLOW}请编辑配置文件并设置 API Key${NC}"
fi

echo -e "${GREEN}[7/7] 创建系统服务...${NC}"
# 创建 systemd 服务文件
cat > /etc/systemd/system/clawscope.service <<EOF
[Unit]
Description=ClawScope AI Agent Platform
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=$INSTALL_DIR
Environment="PATH=$INSTALL_DIR/venv/bin"
Environment="OPENAI_API_KEY="
ExecStart=$INSTALL_DIR/venv/bin/python -m uvicorn clawscope.server:api --host 0.0.0.0 --port $SERVICE_PORT
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# 重载 systemd
systemctl daemon-reload

echo ""
echo -e "${GREEN}======================================"
echo "部署完成！"
echo "======================================${NC}"
echo ""
echo "下一步操作："
echo ""
echo "1. 设置 API Key:"
echo "   export OPENAI_API_KEY='your_api_key_here'"
echo "   或编辑配置文件: ~/.clawscope/config.yaml"
echo ""
echo "2. 启动服务:"
echo "   systemctl start clawscope"
echo "   systemctl enable clawscope  # 开机自启"
echo ""
echo "3. 查看状态:"
echo "   systemctl status clawscope"
echo ""
echo "4. 查看日志:"
echo "   journalctl -u clawscope -f"
echo ""
echo "5. 测试 API:"
echo "   curl http://localhost:$SERVICE_PORT/health"
echo ""
echo "6. 手动启动 (测试用):"
echo "   cd $INSTALL_DIR"
echo "   source venv/bin/activate"
echo "   clawscope serve --port $SERVICE_PORT"
echo ""
echo -e "${YELLOW}注意: 请确保防火墙允许 $SERVICE_PORT 端口访问${NC}"
echo ""
