#!/bin/bash
# Test script for the monitoring system

set -e

echo "=========================================="
echo "  全网信息监测系统 - 测试脚本"
echo "=========================================="
echo ""

PROJECT_DIR="/home/admin/.openclaw/workspace/monitoring-system"
cd "$PROJECT_DIR"

# 1. Check Docker Compose configuration
echo "📋 检查 Docker Compose 配置..."
docker compose config > /dev/null
echo "✅ Docker Compose 配置有效"
echo ""

# 2. Test Python files
echo "🐍 检查 Python 文件语法..."
for file in backend/main.py backend/gpu_manager.py backend/nim_integration.py backend/redis_streams.py collector/collector.py inference/main.py training/main.py; do
    python3 -m py_compile "$file"
    echo "  ✅ $file"
done
echo ""

# 3. Check frontend build
echo "⚛️  检查前端构建..."
cd frontend
if [ ! -d "dist" ]; then
    echo "  ⚠️  前端未构建，运行 npm run build..."
    npm run build
fi
if [ -d "dist" ] && [ -f "dist/index.html" ]; then
    echo "  ✅ 前端构建完成"
else
    echo "  ❌ 前端构建失败"
    exit 1
fi
echo ""

cd "$PROJECT_DIR"

# 4. Validate all required files exist
echo "📁 检查必需文件..."
required_files=(
    "docker-compose.yml"
    "nginx/nginx.conf"
    "backend/main.py"
    "backend/Dockerfile"
    "backend/requirements.txt"
    "collector/collector.py"
    "collector/Dockerfile"
    "inference/main.py"
    "inference/Dockerfile"
    "training/main.py"
    "training/Dockerfile"
    "frontend/Dockerfile"
    "frontend/package.json"
)

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file 缺失"
        exit 1
    fi
done
echo ""

# 5. Summary
echo "=========================================="
echo "  ✓ 所有必要文件和配置检查通过"
echo "  ✓ Python 文件语法正确"
echo "  ✓ 前端构建成功"
echo "  ✓ Docker Compose 配置有效"
echo ""
echo "  可以运行 ./scripts/deploy.sh 部署系统"
echo "=========================================="
