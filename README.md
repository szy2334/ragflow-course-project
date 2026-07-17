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

完整接口和业务流程请参阅 [设计文档](./科研论文智能阅读系统_完整项目设计文档.md)。
