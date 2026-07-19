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

## one-xl 分支备注

fk sun cook
