#!/usr/bin/env bash
# Deploy script for the monitoring system (bash/WSL compatible)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

compose_cmd() {
    if docker compose version >/dev/null 2>&1; then
        docker compose "$@"
        return
    fi
    if command -v docker-compose >/dev/null 2>&1; then
        docker-compose "$@"
        return
    fi
    echo "❌ Docker Compose 未安装"
    return 127
}

echo "=========================================="
echo "  全网信息监测系统 - 部署脚本"
echo "=========================================="
echo ""

# 1. Check Docker
echo "📦 检查 Docker..."
if ! command -v docker >/dev/null 2>&1; then
    echo "❌ Docker 未安装"
    exit 1
fi

if ! compose_cmd version >/dev/null 2>&1; then
    exit 1
fi

echo "✅ Docker 已安装"
echo ""

# 2. Build frontend
echo "🔨 构建前端..."
cd frontend
if [ ! -d "node_modules" ]; then
    npm install
fi
npm run build
echo "✅ 前端构建完成"
echo ""

# 3. Build Docker images
cd "$ROOT_DIR"
echo "🐳 构建 Docker 镜像..."
compose_cmd build

echo "✅ Docker 镜像构建完成"
echo ""

# 4. Start services
echo "🚀 启动服务..."
compose_cmd up -d

echo "✅ 服务已启动"
echo ""

# 5. Wait for services to be healthy
echo "⏳ 等待服务就绪..."
sleep 10

# 6. Check service status
echo ""
echo "📊 服务状态："
compose_cmd ps

echo ""
echo "=========================================="
echo "  部署完成！"
echo "=========================================="
echo ""
echo "访问地址："
echo "  - 前端: http://localhost"
echo "  - 后端 API: http://localhost:8000"
echo "  - API 文档: http://localhost:8000/docs"
echo "  - Grafana: http://localhost:3000 (credentials from .env: GF_SECURITY_ADMIN_PASSWORD)"
echo ""
echo "查看日志："
echo "  docker compose logs -f [service_name]"
echo ""
echo "停止服务："
echo "  docker compose down"
echo ""
