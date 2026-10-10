#!/usr/bin/env bash
# 裸机部署（在已同步代码的服务器上执行）
#
#   sudo bash deploy/install.sh                          # 方案 C：端口直连  http://IP:8501
#   sudo bash deploy/install.sh --auth admin:你的密码     # 方案 C+：端口直连 + 口令保护
#   sudo bash deploy/install.sh --domain hs.example.com   # 方案 A：域名 + nginx 80
#
# 方案 A 装完再跑 certbot --nginx -d 你的域名 即可上 HTTPS。

set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$APP_DIR/.venv"
MIRROR="https://mirrors.cloud.tencent.com/pypi/simple"
DOMAIN=""
AUTH=""

while [ $# -gt 0 ]; do
    case "$1" in
        --domain) DOMAIN="${2:-}"; shift 2 ;;
        --auth)   AUTH="${2:-}";   shift 2 ;;
        *)        shift ;;
    esac
done

# 端口分配：域名模式走 nginx 80 → 8501；口令模式走 nginx 8501 → 8502
if [ -n "$DOMAIN" ]; then
    LISTEN="127.0.0.1"; APP_PORT=8501
elif [ -n "$AUTH" ]; then
    LISTEN="127.0.0.1"; APP_PORT=8502
else
    LISTEN="0.0.0.0";   APP_PORT=8501
fi

echo "==> 安装目录 $APP_DIR"
echo "==> 模式：$([ -n "$DOMAIN" ] && echo "域名 ${DOMAIN} + nginx 80" || echo "端口直连 8501$([ -n "$AUTH" ] && echo " + 口令")")"

command -v python3 >/dev/null || { echo "缺少 python3"; exit 1; }
python3 -c 'import sys;sys.exit(0 if sys.version_info[:2]>=(3,10) else 1)' \
    || { echo "需要 Python >= 3.10"; exit 1; }

[ -x "$VENV/bin/python" ] || { echo "==> 创建虚拟环境"; python3 -m venv "$VENV"; }

echo "==> 安装依赖（腾讯镜像源）"
"$VENV/bin/pip" install -q -U pip -i "$MIRROR"
"$VENV/bin/pip" install -q -r "$APP_DIR/deploy/requirements-web.txt" -i "$MIRROR"

# sudo 下仍然用登录用户跑服务，避免 Streamlit 以 root 运行
RUN_USER="${SUDO_USER:-$(id -un)}"

echo "==> 写入 systemd 服务"
cat > /etc/systemd/system/hybridscope.service <<EOF
[Unit]
Description=HybridScope workbench (Streamlit)
After=network.target

[Service]
Type=simple
User=$RUN_USER
WorkingDirectory=$APP_DIR
Environment="PATH=$VENV/bin"
Environment="TZ=Asia/Shanghai"
ExecStart=$VENV/bin/streamlit run hybridscope/app.py \\
          --server.address=$LISTEN \\
          --server.port=$APP_PORT \\
          --server.headless=true \\
          --browser.gatherUsageStats=false
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now hybridscope

if [ -n "$DOMAIN" ] || [ -n "$AUTH" ]; then
    echo "==> 安装 nginx"
    if command -v apt-get >/dev/null; then
        export DEBIAN_FRONTEND=noninteractive
        apt-get update -qq
        apt-get install -y -qq nginx apache2-utils
    else
        yum install -y -q nginx httpd-tools
    fi

    if [ -n "$DOMAIN" ]; then
        sed "s/server_name _;/server_name $DOMAIN;/" \
            "$APP_DIR/deploy/nginx.conf" > /etc/nginx/conf.d/hybridscope.conf
    else
        sed -e "s|proxy_pass http://127.0.0.1:8501;|proxy_pass http://127.0.0.1:$APP_PORT;|" \
            "$APP_DIR/deploy/nginx-port.conf" > /etc/nginx/conf.d/hybridscope.conf
        user="${AUTH%%:*}"; pass="${AUTH#*:}"
        htpasswd -bc /etc/nginx/.htpasswd "$user" "$pass"
        # nginx worker 以 www-data 运行，口令文件必须可读，否则验证阶段直接 500
        chown root:www-data /etc/nginx/.htpasswd
        chmod 640 /etc/nginx/.htpasswd
        sed -i 's|^    # auth_basic|    auth_basic|; s|^    # auth_basic_user_file|    auth_basic_user_file|' \
            /etc/nginx/conf.d/hybridscope.conf
    fi

    nginx -t && systemctl enable --now nginx
fi

sleep 3
systemctl is-active --quiet hybridscope || { echo "服务启动失败"; exit 1; }
echo "==> 服务已启动"
curl -s -o /dev/null -w "健康检查 HTTP %{http_code}\n" "http://127.0.0.1:$APP_PORT/_stcore/health"

echo
if [ -n "$DOMAIN" ]; then
    echo "访问 http://$DOMAIN/   （如需 HTTPS：certbot --nginx -d $DOMAIN）"
else
    echo "访问 http://服务器公网IP:8501/   需在腾讯云安全组放行 TCP:8501"
fi
