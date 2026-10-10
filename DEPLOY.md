# HybridScope 服务器部署指南

目标：把 `hybridscope/app.py` 工作台跑到腾讯云 CVM（**<SERVER-IP>**）上，用浏览器访问。

**当前采用方案 C（端口直连）**：`http://<SERVER-IP>:8501/`，不碰 nginx，不占用 80 端口
（这台服务器上还跑着别的服务，避免抢默认站点）。拿到域名后按第 5.5 节迁移到方案 A。

镜像/目录里**只包含产品层**（`hybridscope/`、`knn_hybrid/`）。成绩数据不进服务器镜像。

---

## 0. 前置准备

| 项 | 要求 |
|---|---|
| 服务器 | 腾讯云 CVM，1 核 2 G 起步（推荐 2 核 4 G） |
| 系统 | Ubuntu 22.04 / TencentOS / CentOS 均可 |
| 安全组 | 当前放行 **8501**；以后换域名改放行 **80 / 443** |
| Python | ≥ 3.10（Docker 方案无需关心，镜像自带 3.11） |

> 腾讯云控制台 → 云服务器 → 安全组 → 入站规则，添加 `TCP:8501 0.0.0.0/0`。
> 8501 是唯一的对外入口，别再额外开别的端口。

---

## 1. 把代码送到服务器（三选一）

**A. 一键上传脚本（推荐，Git Bash 里执行）**

```bash
bash deploy/upload.sh root@你的公网IP            # 只上传
bash deploy/upload.sh root@你的公网IP --install  # 上传后直接装 systemd + nginx
```

脚本用 tar 走 SSH，只传最小运行集（`knn_hybrid`、`hybridscope`、`.streamlit`、`deploy`），
其余内容不上传。需先配免密：`ssh-copy-id root@你的公网IP`。

**B. 推到私有 GitHub 仓库，服务器 clone**

```bash
git remote add origin git@github.com:你的账号/hybridscope.git && git push -u origin main
ssh root@IP "git clone https://github.com/你的账号/hybridscope /opt/hybridscope"
```

---

## 2. 方案一：Docker（推荐）

```bash
# 服务器上执行
curl -fsSL https://get.docker.com | bash
systemctl enable --now docker

# 境内拉 Docker Hub 慢，配腾讯云镜像加速
mkdir -p /etc/docker
echo '{"registry-mirrors":["https://mirror.ccs.tencentyun.com"]}' > /etc/docker/daemon.json
systemctl restart docker

cd /opt/hybridscope
docker compose -f deploy/docker-compose.yml up -d --build
docker compose -f deploy/docker-compose.yml logs -f      # 看到 8501 就绪即可
docker ps          # STATUS 应为 Up (healthy)
```

> `docker-compose.yml` 里端口映射是 `127.0.0.1:8501:8501`，只监听本机。

若镜像拉取反复失败，改用方案二（裸机），只依赖 pip，腾讯云镜像源速度稳定。

---

## 3. 方案二：裸机 + systemd

```bash
cd /opt/hybridscope
sudo bash deploy/install.sh                        # 方案 C：端口直连 8501
sudo bash deploy/install.sh --auth admin:你的密码   # 方案 C+：加一层口令（公网推荐）
```

脚本做四件事：建 `.venv` → 用腾讯镜像源装依赖 → 写 systemd 服务并开机自启 → 按模式装 nginx。

三种模式：

| 命令 | 服务监听 | 对外入口 |
|---|---|---|
| 无参数 | `0.0.0.0:8501` | `http://IP:8501/` |
| `--auth user:pass` | `127.0.0.1:8502` + nginx 监听 8501 | `http://IP:8501/`（带口令） |
| `--domain 域名` | `127.0.0.1:8501` + nginx 监听 80 | `http://域名/` |

常用运维：

```bash
systemctl status hybridscope
systemctl restart hybridscope
journalctl -u hybridscope -f
```

### 升级到新版本

代码改动后重传即可，配置与口令不动：

```bash
bash deploy/upload.sh root@<SERVER-IP>
ssh root@<SERVER-IP> "sudo systemctl restart hybridscope"
```

前置是 SSH 免密（`ssh-copy-id root@<SERVER-IP>`）。没配免密时，可以在腾讯云控制台的
「登录 / VNC」里手动跑下面两条（代码已在 GitHub：`adhoccn/hybridscope`）：

```bash
cd /opt/hybridscope && git pull && sudo systemctl restart hybridscope
```

### HTTP 服务层（可选，2.3 起）

工作台之外还有一个 API 服务，给脚本/其他系统调用，端口 **8000**（与 8501 互不冲突）：

```bash
cd /opt/hybridscope && pip install -r deploy/requirements-web.txt
python -m hybridscope.serve --host 0.0.0.0 --port 8000
```

路由：`/health`、`/analyze`、`/analyze/text`、`/snapshot`、`/compare`、`/unify`、`/advice`。
要对外开的话记得在安全组放行 8000，并且别用 `--host 0.0.0.0` 裸奔——前面至少挡一层
nginx 或口令。

---

## 4. nginx 反向代理（仅方案 A / C+ 需要）

方案 C 端口直连不装 nginx。要用域名或加口令时：

```bash
cp /opt/hybridscope/deploy/nginx.conf /etc/nginx/conf.d/hybridscope.conf
nginx -t && systemctl reload nginx
```

配置已含 WebSocket 升级头 —— **缺了它页面能打开但按钮点了没反应**，这是最常见的坑。

---

## 5. HTTPS（可选但建议）

有域名：

```bash
apt install -y certbot python3-certbot-nginx
certbot --nginx -d 你的域名
```

只有 IP 无域名：可以跳过 HTTPS，或改用 Cloudflare/CDN 套一层。

---

## 5.5 以后拿到域名：从端口直连迁到方案 A

1. 域名解析加一条 A 记录 → `<SERVER-IP>`
2. 安全组：放行 `TCP:80`、`TCP:443`，**移除 `TCP:8501`**
3. 服务器上重跑（会重写 systemd 与 nginx 配置，代码不动）：

```bash
cd /opt/hybridscope
sudo bash deploy/install.sh --domain 你的域名
sudo certbot --nginx -d 你的域名        # 上 HTTPS
```

4. 验证：`curl -I https://你的域名/` 返回 200，浏览器打开能出权重图。

回退：把 systemd 的 `--server.address` 改回 `0.0.0.0` 并放行 8501 即可，无需重装。

---

## 6. 公网暴露的安全设置

工作台能把 CSV 传上服务器，建议至少加一层口令：

```bash
apt install -y apache2-utils
htpasswd -c /etc/nginx/.htpasswd hybridscope
# 然后打开 deploy/nginx.conf 里 auth_basic 两行的注释
nginx -t && systemctl reload nginx
```

另外两点：

- **不要上传未脱敏的真实成绩**。`cases/learning_analytics/parse_grades.py` 有脱敏流程（姓名不落盘、学号 SHA256），先用它处理再上传。
- Streamlit 是单进程，多用户同时点「运行分析」会排队。若访问量大，把端口映射改成多实例 + nginx 负载均衡。

---

## 7. 验证清单

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8501/_stcore/health             # 期望 200
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8501/                           # 期望 200
```

WebSocket（决定按钮点不点得动，期望 **101**）：

```bash
curl -s -o /dev/null -w "%{http_code}\n" -N \
  -H "Connection: Upgrade" -H "Upgrade: websocket" \
  -H "Sec-WebSocket-Version: 13" -H "Sec-WebSocket-Key: x3JJHMbDL1EzLkh9GBhXDw==" \
  -H "Sec-WebSocket-Protocol: streamlit" -H "Origin: http://<SERVER-IP>:8501" \
  http://<SERVER-IP>:8501/_stcore/stream
```

> **端点路径变了**：Streamlit 1.65 起 WebSocket 走 `/_stcore/stream`，旧的 `/_stcore/websocket`
> 已废弃并返回 403。健康检查仍是 `/_stcore/health`。

浏览器打开 `http://<SERVER-IP>:8501/`：

1. 左侧选「内置示例」→ 点「运行分析」
2. 出现红绿权重条形图 + 判定表 = 部署成功
3. 按钮点击有响应 = WebSocket 正常

---

## 8. 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| 页面能开，按钮无反应 | nginx 缺 WebSocket 升级头 | 用仓库里的 `deploy/nginx.conf` |
| `curl /_stcore/websocket` 返回 403 | 路径已废弃 | 改测 `/_stcore/stream`；浏览器不受影响 |
| 上传大 CSV 失败 | 上传上限 | `.streamlit/config.toml` 的 `maxUploadSize` 与 nginx `client_max_body_size` 要一致（默认 200 MB） |
| 分析跑到一半断 | 代理超时 | 已设 600 s；大样本调 `proxy_read_timeout` |
| 图里中文是方框 | 浏览器端字体 | 换 Chrome/Edge；Plotly 在浏览器渲染，服务器无需装字体 |
| 内存吃满被 OOM | 样本量大 + PSO | 降低 PSO 粒子数/迭代；或升配到 2 核 4 G |

---

## 9. 附：部署后的申报/汇报用法

部署链接可直接写进项目申报书与汇报材料，作为「方法可实用」的在线证据。
建议同时把 `cases/learning_analytics/REPORT.md` 与两张权重诊断图作为应用案例一并引用。
