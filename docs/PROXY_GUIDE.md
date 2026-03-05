# Proxy Guide (Server / Headless)

## 目标
为交易 API 连通性与 live 进程提供应用层代理支持（HTTP/SOCKS），并通过连通性探测验证可达性。

## 推荐部署方式
推荐在服务器部署以下任一 headless 代理内核：

- `mihomo` (Clash.Meta)
- `sing-box`

两者都可提供本地监听端口，例如：

- HTTP proxy: `127.0.0.1:7890`
- SOCKS5 proxy: `127.0.0.1:7891`

系统只依赖“本地代理端口”，不依赖 GUI。

## 在控制台中配置
1. 打开前端 `配置中心`，创建 `Proxy Profile`。
2. `proxy_type` 选 `http` 或 `socks5`，填写 host/port（可选用户名密码）。
3. 绑定到：
   - `global`：全局默认
   - `account`：按交易账户
   - `process`：按进程实例（通常需要重启该进程）
4. 执行 `/api/v2/ops/connectivity/probe` 验证 `rest_ok/ws_ok`。

## 热更新与重启
- `account/global` 绑定：对新连接可热生效。
- `process` 绑定：通常 `requires_restart=true`，需重启对应进程后生效。

## WS 与 SOCKS 说明
- 默认 WS 客户端对 SOCKS5 可能不直接可用。
- 若选择 SOCKS5 且 WS 探测失败，系统会返回明确错误：
  `ws_socks5_not_supported_by_default_client_use_http_connect_proxy`
- 建议优先使用支持 HTTP CONNECT 的 HTTP 代理配置来保障 WS 可达。

## AutoDL network_turbo 说明
`network_turbo` 不保证交易所 API 一定可达。是否可用以系统 `connectivity probe` 结果为准。
