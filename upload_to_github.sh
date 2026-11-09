#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/Users/sathyasaravaiya/Documents/Codex/LLMDemo"
BUNDLED_GIT="/Users/sathyasaravaiya/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/fallback/git"

if [[ ! -d "$PROJECT_DIR" ]]; then
  echo "Error: project directory does not exist: $PROJECT_DIR" >&2
  exit 1
fi

if git --version >/dev/null 2>&1; then
  GIT_BIN="$(command -v git)"
elif [[ -x "$BUNDLED_GIT" ]]; then
  GIT_BIN="$BUNDLED_GIT"
else
  echo "Error: Git is unavailable. Install Apple Command Line Tools with: xcode-select --install" >&2
  exit 1
fi

cd "$PROJECT_DIR"

echo "Project: $PROJECT_DIR"
echo "Git: $($GIT_BIN --version)"
echo

if [[ ! -f .gitignore ]]; then
  echo "Error: .gitignore is missing. Create it before uploading." >&2
  exit 1
fi

required_ignores=(".venv/" "artifacts/")
for pattern in "${required_ignores[@]}"; do
  if ! grep -Fxq "$pattern" .gitignore; then
    echo "Error: .gitignore must contain: $pattern" >&2
    exit 1
  fi
done

large_unignored="$(find . -type f -size +90M -print0 | while IFS= read -r -d '' file; do
  relative="${file#./}"
  if ! "$GIT_BIN" check-ignore -q -- "$relative" 2>/dev/null; then
    printf '%s\n' "$relative"
  fi
done)"

if [[ -n "$large_unignored" ]]; then
  echo "Error: files larger than 90 MB are not ignored:" >&2
  printf '%s\n' "$large_unignored" >&2
  exit 1
fi

read -r -p "GitHub repository HTTPS URL: " REPOSITORY_URL
if [[ ! "$REPOSITORY_URL" =~ ^https://github\.com/[^/]+/[^/]+(\.git)?$ ]]; then
  echo "Error: expected a URL such as https://github.com/USERNAME/LLMDemo.git" >&2
  exit 1
fi

read -r -p "Branch [main]: " BRANCH
BRANCH="${BRANCH:-main}"
read -r -p "Commit message [Initial LLMDemo import]: " COMMIT_MESSAGE
COMMIT_MESSAGE="${COMMIT_MESSAGE:-Initial LLMDemo import}"

"$GIT_BIN" init

if ! "$GIT_BIN" config user.name >/dev/null 2>&1; then
  read -r -p "Git author name: " AUTHOR_NAME
  "$GIT_BIN" config user.name "$AUTHOR_NAME"
fi
if ! "$GIT_BIN" config user.email >/dev/null 2>&1; then
  read -r -p "Git author email: " AUTHOR_EMAIL
  "$GIT_BIN" config user.email "$AUTHOR_EMAIL"
fi

"$GIT_BIN" add .

echo
echo "Files staged for upload:"
"$GIT_BIN" status --short
echo

if "$GIT_BIN" diff --cached --quiet; then
  echo "No new changes need to be committed."
else
  "$GIT_BIN" commit -m "$COMMIT_MESSAGE"
fi

"$GIT_BIN" branch -M "$BRANCH"
if "$GIT_BIN" remote get-url origin >/dev/null 2>&1; then
  "$GIT_BIN" remote set-url origin "$REPOSITORY_URL"
else
  "$GIT_BIN" remote add origin "$REPOSITORY_URL"
fi

read -r -s -p "GitHub token (input hidden): " GITHUB_TOKEN
echo
if [[ -z "$GITHUB_TOKEN" ]]; then
  echo "Error: token cannot be empty." >&2
  exit 1
fi

export GITHUB_TOKEN
trap 'unset GITHUB_TOKEN' EXIT

CREDENTIAL_HELPER='!f() { printf "%s\n" "username=x-access-token" "password=$GITHUB_TOKEN"; }; f'
"$GIT_BIN" -c credential.helper= -c credential.helper="$CREDENTIAL_HELPER" push -u origin "$BRANCH"

unset GITHUB_TOKEN
trap - EXIT
echo
echo "Upload completed: $REPOSITORY_URL"
