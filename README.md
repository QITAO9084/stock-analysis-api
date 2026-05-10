# 股票分析 API · 部署与接入指南

## 🚀 一、本地启动（测试用）

```bash
cd stock_api
pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

启动后访问：`http://localhost:8000/docs` 可查看所有接口

---

## 🌐 二、部署到外网（Coze 需要）

Coze 插件只能调用**公网可访问**的 API，本地 localhost 不行。

### 方案 A：部署到云服务器（推荐）

```bash
# 1. 上传 stock_api/ 文件夹到云服务器
# 2. 安装依赖
pip install -r requirements.txt

# 3. 生产启动（用 gunicorn 或直接 uvicorn 暴露端口）
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

### 方案 B：用 ngrok 临时暴露本地（快速测试）

```bash
# 安装 ngrok：https://ngrok.com/download
ngrok http 8000
# 会得到一个公网地址：https://xxxx.ngrok-free.app
# 把这个地址填到 Coze 插件配置里
```

---

## 🔧 三、Coze 插件配置步骤

### Step 1：获取 OpenAPI Schema

已生成：`stock_api/openapi.json`
或者直接访问：`http://你的服务器IP:8000/openapi.json`

### Step 2：在 Coze 中创建自定义插件

1. 打开 Coze → 「插件」→「创建插件」
2. 选择「基于 OpenAPI Schema 创建」
3. 将 `openapi.json` 的内容粘贴进去
4. 设置插件名称：`stock_analysis_api`
5. 简介：`美股港股买卖点分析，提供K线、MACD、RSI、买卖信号`

### Step 3：配置插件 URL

在 Coze 插件配置页面，将 `YOUR_SERVER_IP` 替换成你的实际地址：

```
http://你的服务器IP:8000
```

例如：
```
http://47.xxx.xxx.xxx:8000
```

---

## 🤖 四、更新 Coze Agent 提示词

将原来的提示词「第一步」替换为调用插件：

```markdown
## 第一步：调用 stock_analysis_api 插件获取数据

**必须调用插件**，不要自己编造数据！

调用工具：`get_stock_analyze`
参数：
- symbol：股票代码（美股 AAPL / 港股 0700.HK）
- market：us / hk / auto

插件会返回：
- 股票基本信息（名称、现价、涨跌幅、52周高低）
- 技术指标（RSI、MACD状态、均线趋势、支撑/压力位）
- 买卖信号列表
- 最近60天K线数据（用于画图）

**拿到插件返回数据后，按以下格式输出分析。**
```

---

## 📊 五、接口说明

| 接口 | 用途 | Coze 调用 |
|------|------|-----------|
| `/stock/analyze` | 完整分析（推荐） | ✅ 主接口 |
| `/stock/info` | 仅基本信息 | 辅助 |
| `/stock/kline` | K线+指标数据 | 画图用 |
| `/stock/signal` | 仅买卖信号 | 辅助 |

---

## ⚠️ 六、yfinance 限速问题

yfinance 免费接口有频率限制，频繁调用会返回 `Rate limit`。

**解决方法：**
1. 在调用间加 `time.sleep(1)`
2. 或部署时加请求缓存（已在内置逻辑中处理）
3. 正式使用建议申请 Yahoo Finance API Key

---

## 📁 文件清单

```
stock_api/
├── main.py              ← 主程序（FastAPI）
├── requirements.txt     ← 依赖列表
├── openapi.json        ← OpenAPI Schema（给 Coze 导入）
└── coze_plugin_config.json  ← Coze 插件配置参考
```
