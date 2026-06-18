# SpecWaveCal — 项目 CLAUDE.md

## 网络与 Git 推送

### 问题

本机网络环境存在 TCP 端口阻断：

| 协议 | 端口 | 状态 |
|------|------|------|
| Git HTTPS (`git push/pull`) | 443 | ❌ TCP 层阻断，无法直连 |
| Git SSH (`git@github.com`) | 22 | ❌ Permission denied（`~/.ssh/id_rsa` 未关联 GitHub） |
| `gh` CLI REST API (`api.github.com`) | 443 | ✅ 可用，已认证 |

### 正确推送方式

**永远不要直接执行 `git push`**——它会反复重试 21 秒后超时。

使用项目自带的 `gh-api-push.sh` 脚本，走 GitHub REST API 的 Git Data API 推送：

```bash
# 基本用法（推送当前分支到 origin）
./gh-api-push.sh

# 指定分支和提交信息
./gh-api-push.sh -b main -m "feat: add peak detection"

# 预演（不实际推送）
./gh-api-push.sh --dry-run
```

**原理**：为每个文件创建 Git Blob（base64 编码）→ 组装 Git Tree → 创建 Git Commit → 更新 Branch Ref。

**前提**：
- `gh` CLI 已安装且已认证（`gh auth status` 确认）
- 远端 GitHub 仓库已存在
- 仓库 URL: `https://github.com/Guaaguaaguaa/SpecWaveCal.git`

### 初始化新仓库到 GitHub 的标准流程

```bash
# 1. 本地初始化
git init && git add -A && git commit -m "Initial commit"

# 2. 创建 GitHub 仓库（如尚未创建）
gh repo create SpecWaveCal --public --description "..."

# 3. 添加 remote
git remote add origin https://github.com/Guaaguaaguaa/SpecWaveCal.git

# 4. 推送（用 API 脚本，不是 git push！）
./gh-api-push.sh -b main
```

### Git 用户信息

- GitHub 用户: `Guaaguaaguaa`
- Email: `xinxin_ok_good@126.com`
- GitHub Token: 已存在 Windows 凭据管理器，`gh` 自动使用

## 项目结构

```
SpecWaveCal/
├── run_calibration.py   # 主校准入口
├── run_explore.py       # 探索模式入口
├── debug_peak.py        # 峰值调试工具
├── lamp_registry.py     # 光源注册表
├── gh-api-push.sh       # ★ API 推送脚本
└── wavecal/
    ├── __init__.py
    ├── pipeline.py      # 校准流水线
    ├── calibration.py   # 波⻓校准核心
    ├── peak_finder.py   # 峰值查找
    ├── baseline.py      # 基线校正
    ├── explorer.py      # 参数探索
    ├── config.py        # 配置管理
    └── logger.py        # 日志
```

## 常见操作

### 推送代码

```bash
# 1. 正常 git add / commit
git add -A && git commit -m "描述改动"

# 2. 用 API 推送
./gh-api-push.sh
```

### 从 GitHub 拉取（如果远端有新提交）

```bash
# 方式 1: gh API 获取远端文件覆盖本地（简单但粗暴）
# 方式 2: 直接通过浏览器下载 ZIP，解压覆盖后重新 commit
# 方式 3: 换用其它网络环境执行 git pull
```
