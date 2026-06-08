<p align="center">
  <a href="https://asqav.com">
    <img src="https://asqav.com/logo-text-white.png" alt="Asqav" width="200">
  </a>
</p>
<p align="center">
  AI Agent 合规凭证 - 审计追踪、审批、策略门控、合规报告
</p>
<p align="center">
  <a href="https://github.com/jagmarques/asqav-sdk">English</a> | 中文文档
</p>

# Asqav

[asqav.com](https://asqav.com) 的 Python SDK，为 AI Agent 提供证据层。所有 ML-DSA 密码学运算在服务端执行。开箱即用的审计追踪、审批流、策略门控和合规报告。

## 安装

```bash
pip install asqav
```

## 快速开始

```python
import asqav

asqav.init(api_key="sk_...")
agent = asqav.Agent.create("my-agent")

sig = agent.sign(
    "payment.wire_transfer",
    {"amount_eur": 850000, "beneficiary_iban": "DE89370400440532013000"},
    receipt_type="protectmcp:decision",
    risk_class="high",
    issuer_id="legal:Acme GmbH",
    iteration_id="task-2026-Q2-4821",
)

print(sig.compliance_mode)        # True（默认；传入 compliance_mode=False 可关闭）
print(sig.action_ref)             # "sha256:..."，对 JCS 规范化 action 的哈希
print(sig.previous_receipt_hash)  # 64 位 hex；每个 Agent 的首条记录为 "0"*64
print(sig.verification_url)
```

每次签名默认会生成一份符合 IETF Internet-Draft [`draft-marques-asqav-compliance-receipts`](https://datatracker.ietf.org/doc/draft-marques-asqav-compliance-receipts/) 的 Compliance Receipt：ML-DSA-65（FIPS 204）签名、链式哈希、保留的 `policy_digest`、fail-closed 时间戳锚定，以及一个公开的验证 URL。如需关闭，请传入 `compliance_mode=False`。

## CLI

包内附带 `asqav` CLI，与 Python API 对应。设置 `ASQAV_API_KEY` 后运行：

```bash
asqav verify <signature_id> [--output json]   # 含 IETF 字段（如有）
asqav sign --agent-id ID --action-type T --action-json action.json \
           --compliance-mode --receipt-type protectmcp:decision \
           --risk-class high --issuer-id legal:Acme
asqav agents list / create / revoke
asqav sessions list / end
asqav replay <agent_id> <session_id>
asqav replay-verify <agent_id> <session_id> [--strict]   # IETF 链
asqav preflight <agent_id> <action_type>
asqav budget check / record
asqav approve <session_id> <entity_id>
asqav compliance frameworks / export
asqav audit-pack export --start ISO --end ISO --output-file bundle.json
asqav audit-pack policy <sha256:hex>
asqav payloads erase <signature_id>           # P4：GDPR 删除权
asqav org set-compliance-strict <org_id> --enable|--disable
asqav keys generate --algorithm ed25519|es256 [--out priv.pem]
asqav policies / webhooks list / create / delete
```

以上所有 CLI 命令在免费套餐下均可使用。Asqav 云端是判定 Key 权限的最终依据。

IETF Compliance Receipts profile 相关命令（`sign --compliance-mode`、`audit-pack export`、`audit-pack policy`、`payloads erase`、`replay-verify --strict`、`org set-compliance-strict`）与 SDK 中 `Agent.sign(...)` 和 `verify_compliance_receipt(...)` 的参数一一对应。完整参数说明见 `docs/CLI.md`。

## 路线图

Asqav 当前已上线的六项能力：

- 云端 hash-only 模式 - 已上线（`*.asqav.com` 默认开启）。
- 自托管签名器（split-trust）- 已上线。
- 自带 KMS（AWS KMS / GCP KMS）- 已上线，Enterprise 层级。
- 客户自有存储 - 已上线（自托管；relay payload 白名单在代码中强制）。
- SCITT / COSE_Sign1 收据导出 - 已上线（公开端点 `GET /api/v1/signatures/{id}/cose` 返回 `application/cose`）。
- 气隙 / 本地部署模式 - 已上线（离线 license + 零出网，详见后端仓库 `docs/airgapped-mode.md`）。

最新特性见文档站 <https://asqav.com/docs>。

## 标准

Asqav 的合规凭证基于 IETF Internet-Draft [`draft-marques-asqav-compliance-receipts`](https://datatracker.ietf.org/doc/draft-marques-asqav-compliance-receipts/)，profile 了上游 [`draft-farley-acta-signed-receipts`](https://datatracker.ietf.org/doc/draft-farley-acta-signed-receipts/)，并绑定了 EU AI Act 第 12、26 条以及 DORA 第 17 条。

## Compliance Receipts（IETF profile）

Compliance Receipts 是 SDK 的默认模式。每次 `agent.sign(...)` 都会生成一份符合 [`draft-marques-asqav-compliance-receipts`](https://datatracker.ietf.org/doc/draft-marques-asqav-compliance-receipts/) 的凭证：ML-DSA-65 签名、RFC 3161 + OpenTimestamps 锚点、保留的 `policy_digest`、哈希链 `previousReceiptHash`。如需旧版结构，传入 `compliance_mode=False`。

调用方最常用的四个信封扩展字段：

- `receipt_type` - `protectmcp:decision`、`protectmcp:restraint` 或 `protectmcp:lifecycle`。
- `risk_class` - 受控词汇：`unacceptable | high | limited | minimal | gpai | low | medium | unknown`。
- `iteration_id` - 逻辑任务 id，与 session 区分。
- `sandbox_state` - 高风险门控用的 `enabled | disabled | unavailable`。
- `incident_class` - DORA / NYDFS / CIRCIA token（可为数组）。
- `issuer_id` - LEI（ISO 17442）、EIN、CIK，或非 LEI 部署方的 W3C DID。

### Audit Pack 导出

云端会对一段时间窗内的收据签发一份 Compliance Audit Pack（对应 IETF -03 第 7 节）。SDK 提供封装：

```python
pack = asqav.fetch_audit_pack(start="2026-05-01T00:00Z", end="2026-06-01T00:00Z")
print(pack["bundle_digest"])              # sha256:<hex>
print(pack["bundle_signature"])           # 对 bundle 的 ML-DSA-65 base64 签名
print(pack["regime_mapping"])             # {regime_token: [record_id, ...]}
print(pack["algorithm_registry_version"]) # 签发时锚定的算法注册表版本
```

气隙场景请改用 `asqav.export_bundle(signatures, framework="dora")`：它在内存中对收据列表计算 Merkle 根，不调用云端。只要云端可达，请优先用 `fetch_audit_pack`，因为只有云端签名才能给审计方一份防篡改的清单。

本地侧的基础校验（必填字段是否存在、命名空间、300 秒时钟偏差边界、前驱重派生）通过 `asqav.verify_compliance_receipt(envelope, predecessor_envelope=...)` 提供。权威验证方仍是云端，该 helper 仅作便利。

按 profile 第 10.8 节的算法敏捷性，可用算法通过 `asqav.SUPPORTED_ALGORITHMS` 暴露。`Agent.create(...)` 可传入 `algorithm="ed25519"` 或 `"es256"` 用于经典（非 ML-DSA）身份，或用 `asqav.generate_local_keypair("ed25519")` 进行离线场景。

## 文档

- 仓库：<https://github.com/jagmarques/asqav-sdk>
- 完整文档：<https://asqav.com/docs>
- 路线图：<https://asqav.com/roadmap>

## License

MIT。在 [asqav.com](https://asqav.com) 申请 API Key。
