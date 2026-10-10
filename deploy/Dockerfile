# HybridScope 工作台镜像。
# 只打包产品层真正用到的三个包，论文草稿与实验脚本不进镜像。

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=Asia/Shanghai

WORKDIR /app

COPY deploy/requirements-web.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && rm -f /tmp/requirements.txt

COPY knn_hybrid ./knn_hybrid
COPY hybridscope ./hybridscope
COPY .streamlit ./.streamlit

RUN useradd -m -u 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

# Streamlit 自带健康检查端点；镜像里没有 curl，用 urllib 代替。
HEALTHCHECK --interval=30s --timeout=8s --start-period=25s --retries=3 \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health',timeout=6).read()"

CMD ["streamlit", "run", "hybridscope/app.py"]
