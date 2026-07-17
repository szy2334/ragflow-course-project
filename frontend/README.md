# 科研论文智能阅读系统前端

Vue 3 + Pinia + Axios 的前端实现，接口严格以 `../科研论文智能阅读系统_完整项目设计文档.md` 和后端 `/api/v1/openapi.json` 为准。

```bash
npm install
npm run api:generate # 后端启动后，生成契约类型与 Axios 客户端
npm run dev
```

开发服务器默认将 `/api` 代理到 `http://localhost:8000`；可通过 `VITE_API_ORIGIN` 修改。应用不会请求文档之外的接口。

在 `npm run dev` 的开发环境中，可使用内置演示账号体验完整界面与示例流式回答：

- 邮箱：`demo@zhiyue.local`
- 密码：`Demo@2026`

演示账号仅在开发模式生效，生产构建仍只调用正式 API。
