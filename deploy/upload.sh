#!/usr/bin/env bash
# 本机（Git Bash）执行：把最小运行集传到腾讯云并可选立即部署
#   bash deploy/upload.sh root@123.456.789.0
#   bash deploy/upload.sh root@123.456.789.0 --install     # 传完直接装 nginx + systemd
# 前置：已配好 SSH 免密（ssh-copy-id root@IP）

set -euo pipefail

TARGET="${1:?用法: bash deploy/upload.sh root@服务器IP [--install]}"
DO_INSTALL=0
[ "${2:-}" = "--install" ] && DO_INSTALL=1

REMOTE_DIR=/opt/hybridscope
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> 准备远端目录"
ssh "$TARGET" "sudo mkdir -p $REMOTE_DIR && sudo chown \$(id -un):\$(id -gn) $REMOTE_DIR"

echo "==> 打包最小运行集 → $TARGET:$REMOTE_DIR"
tar -czf - \
    --exclude='__pycache__' --exclude='*.pyc' \
    --exclude='knn_hybrid/figures' --exclude='knn_hybrid/results' \
    --exclude='.git' \
    knn_hybrid hybridscope .streamlit deploy \
  | ssh "$TARGET" "tar -xzf - -C $REMOTE_DIR && ls $REMOTE_DIR"

echo "==> 上传完成"

if [ "$DO_INSTALL" = 1 ]; then
    echo "==> 远端安装（venv + systemd + nginx）"
    ssh -t "$TARGET" "cd $REMOTE_DIR && sudo bash deploy/install.sh --nginx"
fi

echo
echo "下一步：腾讯云安全组放行 80 端口，浏览器打开 http://${TARGET#*@}/"
