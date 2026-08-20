# Implementation Plan

- [x] 1. 明确页面、API、派生产物和只读边界
  - 保留实时事件与 Coach 原件
  - 使用本地同源 HTTP 服务完成写入与重算
  - _Requirement: R5.1, R5.6, R5.9, R5.11_

- [x] 2. 实现人工标注校验与评估 workflow
  - 校验 schema、会话、事件、边界和 pending 状态
  - 原子写入人工标注、评估和状态文件
  - _Requirement: R5.6, R5.7, R5.9, R5.10, R5.11_

- [x] 3. 实现人工边界证据与 Local Coach 重算
  - 读取完整 FrameRecord JSONL
  - 重算质量、生物力学、校准和建议
  - 写入来源与变更审计字段
  - _Requirement: R5.1, R5.8, R15, R16_

- [x] 4. 实现 final_report 人工校准闭环页面
  - 文件导入、校验状态、评估指标和错误显示
  - 实时 Coach / 人工校准 Coach 并列对比
  - _Requirement: R5.2, R5.3, R5.4, R5.10, R5.11_

- [x] 5. 完成自动化与浏览器验证
  - 单元、HTTP、现有回归测试
  - 当前 4 事件会话 pending 与 finalized 两条流程
  - _Requirement: R13, R14, R15, R16_
