#!/usr/bin/env bash
set -euo pipefail

DEV_URL="https://github.com/4spanch1k/soup-kernel-challenge.git"
SRC_REF="${SRC_REF:-origin/main}"
PLATFORM_URL="${PLATFORM_URL:-https://github.com/BAITC-Hacks/hack-b6e5a4cf-mononyx}"
KEEP=()
DRY_RUN=0

for argument in "$@"; do
  case "$argument" in
    --dry-run)
      DRY_RUN=1
      ;;
    *)
      printf 'Ошибка: неизвестный параметр: %s\n' "$argument" >&2
      printf 'Использование: SRC_REF=origin/main PLATFORM_URL=<URL> bash scripts/push_to_platform.sh [--dry-run]\n' >&2
      exit 2
      ;;
  esac
done

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/platform-push.XXXXXX")"
trap 'rm -rf "$WORK_DIR"' EXIT
REPO_DIR="$WORK_DIR/platform"

printf 'Клонирую платформенный репозиторий: %s\n' "$PLATFORM_URL"
git clone --quiet "$PLATFORM_URL" "$REPO_DIR"
cd "$REPO_DIR"

git fetch --quiet origin main
git switch --quiet --create sync-final --track origin/main
PLATFORM_BASE="$(git rev-parse origin/main)"

printf 'Добавляю репозиторий разработки и получаю ветки и теги.\n'
git remote add dev "$DEV_URL"
git fetch --quiet --prune --tags dev

case "$SRC_REF" in
  origin/*)
    DEV_REF="refs/remotes/dev/${SRC_REF#origin/}"
    ;;
  refs/remotes/origin/*)
    DEV_REF="refs/remotes/dev/${SRC_REF#refs/remotes/origin/}"
    ;;
  dev/*)
    DEV_REF="refs/remotes/$SRC_REF"
    ;;
  refs/remotes/dev/*|refs/tags/*)
    DEV_REF="$SRC_REF"
    ;;
  *)
    if git show-ref --verify --quiet "refs/tags/$SRC_REF"; then
      DEV_REF="refs/tags/$SRC_REF"
    else
      DEV_REF="refs/remotes/dev/$SRC_REF"
    fi
    ;;
esac

if ! DEV_COMMIT="$(git rev-parse --verify --quiet "${DEV_REF}^{commit}")"; then
  printf 'Ошибка: не найдена ветка или тег источника "%s" в репозитории разработки.\n' "$SRC_REF" >&2
  exit 1
fi
SOURCE_TREE="$(git rev-parse "${DEV_COMMIT}^{tree}")"

printf 'Источник: %s (%s)\n' "$SRC_REF" "$DEV_COMMIT"
printf 'Платформенная main: %s\n' "$PLATFORM_BASE"
printf 'Заменяю дерево платформенной ветки деревом источника.\n'
git read-tree --reset -u "$DEV_COMMIT"

KEEP_COUNT=0
if [[ -n "${KEEP[*]-}" ]]; then
  KEEP_COUNT=${#KEEP[@]}
  for path in "${KEEP[@]}"; do
    if git cat-file -e "$PLATFORM_BASE:$path" 2>/dev/null; then
      git checkout "$PLATFORM_BASE" -- "$path"
      printf 'Оставляю платформенный путь: %s\n' "$path"
    else
      printf 'Путь KEEP отсутствует в платформенной main, пропускаю: %s\n' "$path"
    fi
  done
fi

CODE_DIFF="$(git diff --stat "$DEV_REF" -- solution harness tests task)"
if [[ -n "$CODE_DIFF" ]]; then
  printf 'Ошибка: solution/, harness/, tests/ или task/ отличаются от выбранного источника:\n%s\n' "$CODE_DIFF" >&2
  exit 1
fi
printf 'Проверка solution/, harness/, tests/, task/: различий с источником нет.\n'

if git diff --cached --quiet origin/main; then
  printf 'Ошибка: дерево платформы уже совпадает с подготовленным деревом; создавать пустой коммит не буду.\n' >&2
  exit 1
fi

INDEX_TREE="$(git write-tree)"
git commit --quiet -m "sync final submission from development repository"

NEW_MESSAGES="$(git log --format=%B origin/main..HEAD)"
CLAUDE_COUNT="$(printf '%s\n' "$NEW_MESSAGES" | grep -ci 'claude' || true)"
printf 'Проверка новых сообщений коммитов: claude=%s (ожидается 0).\n' "$CLAUDE_COUNT"
if [[ "$CLAUDE_COUNT" != "0" ]]; then
  printf 'Ошибка: в сообщении нового коммита найдено слово Claude.\n' >&2
  exit 1
fi

ACTUAL_TREE="$(git rev-parse 'HEAD^{tree}')"
if [[ "$KEEP_COUNT" -eq 0 ]]; then
  EXPECTED_TREE="$SOURCE_TREE"
else
  EXPECTED_TREE="$INDEX_TREE"
fi
if [[ "$ACTUAL_TREE" != "$EXPECTED_TREE" ]]; then
  printf 'Ошибка: дерево коммита (%s) не совпало с ожидаемым (%s).\n' "$ACTUAL_TREE" "$EXPECTED_TREE" >&2
  exit 1
fi
printf 'Проверка дерева коммита пройдена: %s\n' "$ACTUAL_TREE"

if [[ -n "$(git status --porcelain)" ]]; then
  printf 'Ошибка: после коммита рабочее дерево неожиданно изменено; push отменён.\n' >&2
  git status --short >&2
  exit 1
fi

NEW_COMMIT="$(git rev-parse HEAD)"
printf '\nПеренос подготовлен.\n'
printf '  Ветка: sync-final\n'
printf '  Платформенная база: %s\n' "$PLATFORM_BASE"
printf '  Источник: %s\n' "$DEV_COMMIT"
printf '  Новый коммит: %s\n' "$NEW_COMMIT"
printf '  Изменения: %s\n' "$(git diff --shortstat origin/main..HEAD)"
printf 'Команда отправки: git push origin sync-final:main\n'

if [[ "$DRY_RUN" -eq 1 ]]; then
  printf 'Режим --dry-run: push не выполнялся.\n'
  exit 0
fi

answer=''
if ! IFS= read -r -p 'Выполнить обычный push в main? [y/N] ' answer; then
  printf '\nОтвет не получен; push пропущен.\n'
  exit 0
fi

case "$answer" in
  y|Y|yes|YES|Yes)
    git push origin sync-final:main
    printf 'Push завершён. Проверьте main в GitHub.\n'
    ;;
  *)
    printf 'Push пропущен. Подготовленная ветка удалится вместе с временной папкой.\n'
    ;;
esac
