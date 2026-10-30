# Как прогнать решение на Kaggle T4 (единственное место, где цифры считаются)

Ничего из `solution/` пока не запускалось на GPU. Первый прогон показывает, что работает.

## Один раз
1. Kaggle: **Create → New Notebook → File → Import Notebook**, файл `notebook/soup_kernel_challenge.ipynb`
   (в нём уже стоит `REPO_URL` нашего репозитория).
2. Справа: **Accelerator = GPU T4 x2**, **Internet = on** (нужен аккаунт с подтверждённым телефоном).
3. Репозиторий приватный → **Add-ons → Secrets → `GITHUB_TOKEN`** (fine-grained токен GitHub,
   только Contents: Read-only на этот репозиторий). Токен создаёте вы, в чат его не присылайте.
   Необязательно: секрет `HF_TOKEN` (меньше лимитов Hugging Face).

## Что запускать (от быстрого к полному)
Чтобы проверить конкретную ветку, поставьте в первой ячейке `COMMIT = "<sha или имя ветки>"`.

| Шаг | Команда (ячейка с `!cd /kaggle/working/repo && ...`) | Что покажет |
|---|---|---|
| 1 | `python task/prepare_data.py && python -m pytest solution -q` | компилируются ли Triton-ядра, совпадают ли они с PyTorch |
| 2 | `SOUP_APPLY_SOLUTION=1 python -m pytest tests -q` | 79 тестов Soup с нашими патчами (проверка 10) |
| 3 | `python harness/run_task.py --only solution --out runs_try --profile` | быстрый прогон только решения + таблица ядер |
| 4 | `python harness/run_task.py && python harness/check_rules.py && python harness/score.py runs/result.json` | полный замер и 10 проверок |

Присылайте мне текст вывода шагов 1–2 целиком (если упало) и `runs/checks.json` + `runs/result.json`
после шага 4. Для сравнения recompute: тот же шаг 4 с `SOUP_RECOMPUTE_MLP=1` в окружении.

## Порядок веток
PR #1 (CE) → PR #2 (RMSNorm+RoPE) → PR #3 (recompute, выключен). Сначала проверяйте PR #1
отдельно: в нём самый большой выигрыш по памяти и больше всего неопределённости.
