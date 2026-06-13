# 口罩检测系统 🛡️

基于 YOLOv5 的口罩检测系统，支持图片上传自动检测是否佩戴口罩。

## 项目架构

```
public/              ← Cloudflare Pages 部署（前端）
  index.html          前端页面
  _redirects          Cloudflare 路由配置

backend/             ← 本地/VPS 运行（后端）
  app.py              Flask API 服务
  templates/index.html  原始模板

yolov5/              ← YOLOv5 模型
  runs/train/mask_detection/weights/best.pt  训练好的权重
```

## 启动方式

### 1. 启动后端（本地或 VPS）

```bash
cd backend
python app.py
```

后端运行在 你的IP

### 2. 配置前端 API 地址

打开 `public/index.html`，修改顶部 `API_BASE` 变量：



// VPS 部署
const API_BASE = "http://你的VPS_IP:5000";
```

### 3. 前端部署（Cloudflare Pages）

连接 GitHub 仓库到 Cloudflare Pages：
- Build output directory: `public`
- No build command needed

也可本地预览：

```bash
npx wrangler pages dev public
```

## 效果

- 上传图片 → 自动检测口罩佩戴情况
- 绿色框 = 已佩戴口罩 😷
- 红色框 = 未佩戴口罩 ⚠️
- 统计：戴口罩人数 / 未戴口罩人数 / 平均置信度
