# cli.py 中文注释设计

**日期**: 2026-05-09

## 目标

将 `sentrysearch/cli.py` 中的英文注释和用户可见消息翻译为中文，并为缺少文档的关键函数补充中文 docstring。

## 范围

仅修改 `sentrysearch/cli.py`，不改变任何功能逻辑。

## 变更内容

### 翻译现有注释
- 模块头部 docstring
- 所有函数 docstring（公开和内部函数）
- 所有 `click.echo` / `click.secho` 用户可见消息
- `#` 行内注释（分隔线注释、技术性注释）

### 补充缺失 docstring 的函数
- `_resolve_remote_api_key` — 解析远程 API 密钥
- `_cache_last_clip` — 记录最近保存的 clip 路径
- `_open_file` — 用系统默认应用打开文件
- `_overlay_output_path` — 生成叠加层输出路径
- `_is_permanent_failure` — 判断错误是否为不可恢复的永久性错误
- `_present_results` — 格式化并展示搜索结果
- `_print_shell_results` — shell 模式下打印结果

### 不翻译的内容
- 代码中的变量名、函数名（保持英文）

### 测试文件更新
- 更新 `tests/test_cli.py` 中断言以匹配新的中文消息
