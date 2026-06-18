#!/bin/bash
# =============================================================================
# gh-api-push.sh — Push git commits via GitHub REST API
# =============================================================================
# 用途：当 git:// 端口（443 HTTPS、22 SSH）被网络阻断时，通过 GitHub REST API
#       走 Git Data API 推送代码。本机 gh CLI 可正常访问 api.github.com。
#
# 用法：
#   ./gh-api-push.sh                        # 推送当前分支到 origin
#   ./gh-api-push.sh -b main                # 指定分支
#   ./gh-api-push.sh -m "commit message"    # 自定义提交信息
#   ./gh-api-push.sh -r origin -b main -m "fix: ..."
#
# 前置条件：
#   - gh CLI 已安装且已认证（gh auth status）
#   - 当前目录是一个 git 仓库
#   - 远端 GitHub 仓库已存在
# =============================================================================

set -euo pipefail

# ---- 参数解析 ----
REMOTE="origin"
BRANCH=""
MESSAGE=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        -r|--remote) REMOTE="$2"; shift 2 ;;
        -b|--branch) BRANCH="$2"; shift 2 ;;
        -m|--message) MESSAGE="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help)
            sed -n '2,20p' "$0" | grep -v '^#!/'
            exit 0
            ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

# ---- 推断分支 ----
if [[ -z "$BRANCH" ]]; then
    BRANCH=$(git rev-parse --abbrev-ref HEAD)
fi
echo "==> Remote: $REMOTE  |  Branch: $BRANCH"

# ---- 从 git remote URL 解析 owner/repo ----
REMOTE_URL=$(git remote get-url "$REMOTE")
# 先去掉末尾 .git，再取 owner/repo
OWNER_REPO=$(echo "${REMOTE_URL%.git}" | sed -E 's|.*github\.com[:/]([^/]+/[^/]+)$|\1|')
OWNER="${OWNER_REPO%/*}"
REPO="${OWNER_REPO#*/}"

if [[ -z "$OWNER" || -z "$REPO" ]]; then
    echo "ERROR: Cannot parse owner/repo from remote URL: $REMOTE_URL"
    exit 1
fi
echo "==> Repo: $OWNER/$REPO"

# ---- 获取所有待推送的文件 ----
FILES=$(git ls-files)
if [[ -z "$FILES" ]]; then
    echo "ERROR: No tracked files found."
    exit 1
fi
FILE_COUNT=$(echo "$FILES" | wc -l)
echo "==> Files to push: $FILE_COUNT"

# ---- 提交信息 ----
if [[ -z "$MESSAGE" ]]; then
    MESSAGE=$(git log -1 --format='%s' 2>/dev/null || echo "Commit via gh-api-push")
fi
echo "==> Commit message: $MESSAGE"

# ---- 获取当前 HEAD 提交的 author/committer 信息 ----
AUTHOR_NAME=$(git log -1 --format='%an' 2>/dev/null || git config user.name)
AUTHOR_EMAIL=$(git log -1 --format='%ae' 2>/dev/null || git config user.email)
echo "==> Author: $AUTHOR_NAME <$AUTHOR_EMAIL>"

# ---- 检查远端分支是否存在 ----
REMOTE_REF_EXISTS=false
HEAD_SHA=""
BASE_TREE=""
if gh api "repos/$OWNER/$REPO/git/refs/heads/$BRANCH" --silent 2>/dev/null; then
    REMOTE_REF_EXISTS=true
    HEAD_SHA=$(gh api "repos/$OWNER/$REPO/git/refs/heads/$BRANCH" -q '.object.sha')
    BASE_TREE=$(gh api "repos/$OWNER/$REPO/git/commits/$HEAD_SHA" -q '.tree.sha')
    echo "==> Remote HEAD: $HEAD_SHA"
    echo "==> Remote tree: $BASE_TREE"
else
    echo "==> Remote branch '$BRANCH' does not exist (initial push)"
fi

if $DRY_RUN; then
    echo "==> DRY RUN — stopping here."
    exit 0
fi

# ===================================================================
# 核心逻辑：为每个文件创建 blob，组装 tree，创建 commit，更新 ref
# ===================================================================

# ---- Step 1: 创建 blobs ----
echo ""
echo "--- Creating blobs ---"
declare -A BLOB_MAP

while IFS= read -r file; do
    [[ -z "$file" ]] && continue
    printf "  %-50s " "$file"
    CONTENT=$(base64 -w0 "$file")
    BLOB_SHA=$(gh api "repos/$OWNER/$REPO/git/blobs" \
        -f content="$CONTENT" -f encoding="base64" -q '.sha')
    BLOB_MAP["$file"]="$BLOB_SHA"
    echo "SHA: $BLOB_SHA"
done <<< "$FILES"

# ---- Step 2: 构建 tree entries JSON 并创建 tree ----
echo ""
echo "--- Creating tree ---"

TREE_ENTRIES="["
FIRST=true
while IFS= read -r file; do
    [[ -z "$file" ]] && continue
    MODE="100644"
    [[ -x "$file" ]] && MODE="100755"
    BLOB_SHA="${BLOB_MAP[$file]}"

    if $FIRST; then FIRST=false; else TREE_ENTRIES+=","; fi
    TREE_ENTRIES+="{\"path\":\"$file\",\"mode\":\"$MODE\",\"type\":\"blob\",\"sha\":\"$BLOB_SHA\"}"
done <<< "$FILES"
TREE_ENTRIES+="]"

# 构建 API 请求体
TREE_BODY="{\"tree\":$TREE_ENTRIES"
if [[ -n "$BASE_TREE" ]]; then
    TREE_BODY+=",\"base_tree\":\"$BASE_TREE\""
fi
TREE_BODY+="}"

TREE_SHA=$(gh api "repos/$OWNER/$REPO/git/trees" --input - <<< "$TREE_BODY" -q '.sha')
echo "  New tree: $TREE_SHA"

# ---- Step 3: 创建 commit ----
echo ""
echo "--- Creating commit ---"

COMMIT_BODY=$(cat <<EOF
{
  "message": "$MESSAGE",
  "tree": "$TREE_SHA",
  "parents": [$([ -n "$HEAD_SHA" ] && echo "\"$HEAD_SHA\"" || echo "")],
  "author": {"name": "$AUTHOR_NAME", "email": "$AUTHOR_EMAIL"},
  "committer": {"name": "$AUTHOR_NAME", "email": "$AUTHOR_EMAIL"}
}
EOF
)

COMMIT_SHA=$(gh api "repos/$OWNER/$REPO/git/commits" --input - <<< "$COMMIT_BODY" -q '.sha')
echo "  New commit: $COMMIT_SHA"

# ---- Step 4: 更新 ref ----
echo ""
echo "--- Updating ref ---"

if $REMOTE_REF_EXISTS; then
    gh api "repos/$OWNER/$REPO/git/refs/heads/$BRANCH" \
        -X PATCH -f sha="$COMMIT_SHA" --silent
else
    gh api "repos/$OWNER/$REPO/git/refs" \
        -f ref="refs/heads/$BRANCH" -f sha="$COMMIT_SHA" --silent
fi

echo "==> DONE: $BRANCH → $COMMIT_SHA"
echo "    https://github.com/$OWNER/$REPO/commit/$COMMIT_SHA"
