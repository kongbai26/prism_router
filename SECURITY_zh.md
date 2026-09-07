# Security Policy

[English](SECURITY.md) | [简体中文](SECURITY_zh.md)

## 🛡️ 支持的版本

我们为以下版本的 Prism Router 提供安全漏洞修复和维护：

| 版本   | 支持状态           |
| ------ | ------------------ |
| 0.1.x  | :white_check_mark: |

---

## 🚨 报告安全漏洞

安全对 LLM 代理与网关服务至关重要。如果你在 Prism Router 中发现了安全漏洞（例如 API Key 泄露隐患、鉴权绕过、SSRF、未授权访问等）：

1. **请勿直接公开创建公共 Issue。**
2. 请通过本仓库的 [GitHub Private Vulnerability Reporting](https://github.com/kongbai26/prism_router/security/advisories/new) 私密提交。
3. 在报告中请尽量提供以下信息：
   - 漏洞类型及受影响的功能模块（如 `/v1/responses`、鉴权中间件、Fallback 链等）
   - 复现步骤、最小验证示例（PoC）或攻击场景说明
   - 漏洞可能带来的影响评估
   - 如有修复方案建议，欢迎一并附上

### 响应周期
- 维护团队将在 48 小时内确认并评估漏洞报告。
- 在修复补丁发布前，请与我们协同遵循负责任的漏洞披露机制（Responsible Disclosure）。
