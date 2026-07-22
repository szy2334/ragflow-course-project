# 科研论文智能阅读系统

课程项目：基于多智能体协作与证据溯源的科研论文智能阅读系统。

## 前端

前端位于 [`frontend`](./frontend)，采用 Vue 3、Pinia 和 Axios，实现了论文上传、解析/索引进度、带证据的流式问答、专项分析、论文对比、阅读报告、导出和管理员页面。

```powershell
cd frontend
npm install
npm run dev
```

后端启动后，可用 `npm run api:generate` 从 `/api/v1/openapi.json` 生成最终 TypeScript API 类型。

完整业务流程请参阅 [详细设计文档](./docs/7.18/多智能体论文问答与评审系统_详细设计文档_V1.1_实施验证修订版.md)，前后端与跨模块边界请参阅 [统一接口规范](./docs/统一接口规范_V1.0.md)。

## 格式审查复现

格式审查的完整设计和当前实现说明见 [格式审查模块详细设计](./docs/格式审查模块详细设计_V1.0.md)。以下步骤可在本机复现真实格式审查链路。

```powershell
# 终端 1：后端
cd backend
Copy-Item .env.example .env
# 在 .env 中配置 DATABASE_URL、LLM_BASE_URL、LLM_API_KEY、LLM_MODEL、RAGFLOW_BASE_URL、RAGFLOW_API_KEY
python -m pip install -e ".[test]"
python -m alembic upgrade head
uvicorn app.main:app --reload --port 8000

# 终端 2：前端
cd frontend
npm install
npm run dev
```

打开 `http://127.0.0.1:5173` 后，先注册用户并由管理员创建格式档案。档案必须绑定独立的 RAGFlow 数据集、共享规则文档、各投稿模式专用规则文档，以及包含完整来源字段的版本化规则清单。上传 PDF 并等待状态变为 `ready` 后，在“格式审查”页选择论文、档案和投稿模式提交。报告中的“定位原文”会用 PDF.js 载入对应页并高亮证据坐标；缺少规则覆盖或可靠版面事实的项目会显示“无法可靠判断”。

前端开发演示账号可用于复现页面交互，但不连接 RAGFlow 或模型服务：`demo@zhiyue.local` / `Demo@2026`。

## one-xl 分支备注

fk sun cook
