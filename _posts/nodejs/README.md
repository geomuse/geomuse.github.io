### **Node.js 30 天学习计划**  
本计划适用于 **零基础到进阶** 的学习者，涵盖 Node.js 的 **基础语法、核心模块、异步编程、数据库、框架（Express、NestJS）、REST API、WebSocket、微服务** 等内容。  

---

## **📅 第 1 周：Node.js 基础**
### **📌 第 1 天：Node.js 入门**
- 安装 Node.js & npm（[官网下载](https://nodejs.org/)）
- 运行 `node -v` & `npm -v` 检查安装
- 运行 `node` 进入 REPL（交互式环境）并执行 `console.log("Hello Node.js")`
- 了解 `package.json` & `npm init -y`

### **📌 第 2 天：JavaScript 复习**
- ES6+ 语法（`let/const`、解构、箭头函数）
- 模块化 `import/export` & `require`
- 异步编程（`Promise`、`async/await`）

### **📌 第 3 天：Node.js 模块**
- 内置模块：`fs`（文件系统）、`path`（路径）、`os`（系统信息）
- 创建并导入自定义模块
- 使用 `require()` 和 `import` 加载模块

### **📌 第 4 天：事件 & Stream**
- `EventEmitter` 事件监听机制
- 读写文件流 `fs.createReadStream()`
- 管道流 `pipe()`

### **📌 第 5 天：Node.js 异步编程**
- `setTimeout`、`setInterval`
- `fs.promises` 处理异步文件操作
- `Promise.all()` & `async/await` 用法

### **📌 第 6 天：HTTP 服务器**
- 使用 `http` 模块创建服务器
- 处理 GET/POST 请求
- 解析 `querystring`

### **📌 第 7 天：项目 1 - 构建简单 API**
- 创建简单 JSON API（返回用户数据）
- 使用 `nodemon` 监听代码变更
- `dotenv` 读取环境变量

---

## **📅 第 2 周：进阶 Node.js**
### **📌 第 8 天：Express 框架**
- 安装 Express (`npm install express`)
- 创建 Express 服务器 (`app.listen(port)`)
- 处理 GET、POST 请求

### **📌 第 9 天：中间件**
- Express 内置中间件（`express.json()`、`express.static()`）
- 自定义中间件（`req, res, next`）

### **📌 第 10 天：RESTful API**
- 路由管理 (`app.use('/api', router)`)
- RESTful API 规范（GET、POST、PUT、DELETE）

### **📌 第 11 天：MongoDB & Mongoose**
- 安装 MongoDB (`npm install mongoose`)
- 连接数据库 `mongoose.connect()`
- 定义 Schema & Model (`new mongoose.Schema()`)

### **📌 第 12 天：CRUD 操作**
- `create()` 添加数据
- `find()` 查询数据
- `updateOne()` 更新数据
- `deleteOne()` 删除数据

### **📌 第 13 天：用户认证（JWT）**
- `jsonwebtoken` (`npm install jsonwebtoken`)
- 生成 JWT (`jwt.sign()`)
- 解析 JWT (`jwt.verify()`)

### **📌 第 14 天：项目 2 - 用户管理 API**
- 结合 MongoDB + Express 实现用户增删改查
- 使用 `bcrypt` 进行密码加密
- 使用 JWT 进行身份验证

---

## **📅 第 3 周：高级 Node.js**
### **📌 第 15 天：文件上传**
- 使用 `multer` 处理文件上传
- 存储图片到本地/云存储

### **📌 第 16 天：WebSocket 即时通信**
- 使用 `ws` (`npm install ws`)
- 创建 WebSocket 服务器
- 客户端连接 & 消息传递

### **📌 第 17 天：Redis**
- 安装 Redis (`npm install redis`)
- 连接 Redis (`redis.createClient()`)
- 使用 Redis 进行缓存

### **📌 第 18 天：GraphQL**
- Express + Apollo Server (`npm install apollo-server-express`)
- 定义 GraphQL Schema
- 处理 GraphQL Query & Mutation

### **📌 第 19 天：定时任务**
- `node-cron` 实现定时任务 (`npm install node-cron`)
- 每天定时执行任务

### **📌 第 20 天：单元测试**
- `jest` (`npm install jest supertest`)
- 编写 API 测试用例
- 运行 `npm test`

### **📌 第 21 天：项目 3 - 聊天系统**
- WebSocket 实现多人聊天
- Express + MongoDB 存储消息
- Redis 进行消息队列优化

---

## **📅 第 4 周：深入 Node.js**
### **📌 第 22 天：NestJS 框架**
- 安装 NestJS (`npm install -g @nestjs/cli`)
- 创建项目 (`nest new project`)
- 控制器 & 服务 (`@Controller`, `@Get`)

### **📌 第 23 天：微服务架构**
- `@nestjs/microservices`
- 使用 Redis、Kafka 进行微服务通信

### **📌 第 24 天：Docker 容器化**
- `Dockerfile` 编写
- `docker-compose.yml` 配置 MongoDB

### **📌 第 25 天：CI/CD**
- 使用 GitHub Actions 自动部署
- Docker + Kubernetes 部署

### **📌 第 26 天：性能优化**
- 使用 `cluster` 进行多进程优化
- `PM2` 进程管理 (`npm install pm2 -g`)

### **📌 第 27 天：安全防护**
- `helmet` (`npm install helmet`)
- `rate-limit` 进行限流

### **📌 第 28 天：日志管理**
- `winston` (`npm install winston`)
- 配置日志文件存储

### **📌 第 29 天：Serverless**
- AWS Lambda + Node.js 部署 API
- Vercel 部署 Serverless 服务

### **📌 第 30 天：最终项目 - 企业级 API**
- 结合 Express、NestJS、WebSocket、Docker、Redis
- 实现一个完整的企业级 API 系统

---

## **💡 进阶学习方向**
- TypeScript + Node.js
- NestJS 深入（GraphQL、gRPC）
- AWS Lambda + Serverless
- Deno（Node.js 的替代方案）

这个 30 天学习计划涵盖 **从零基础到进阶** 的知识点，让你能够 **独立开发完整的 Node.js 项目**！💪🚀