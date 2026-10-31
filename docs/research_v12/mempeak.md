# Диагностика пика памяти на T4

Статус на 31.10.2026: **реализовано локально, на Kaggle T4 не запускалось**. На этой машине PyTorch/CUDA нет. Проверки ограничены `bash -n`, `python3 -m py_compile` и разбором JSON notebook; они не подтверждают работу private API PyTorch 2.11 на GPU.

## Что собирается

`scripts/t4_mempeak.py` запускает ровно один `harness/run_task.py --only solution`. Harness по-прежнему формирует конфиг и вызывает `harness/_train.py`. Переменные `PYTHONPATH` и `SOUP_T4_MEMPEAK` подключают `scripts/sitecustomize.py` только к дочернему `_train.py` с флагом `--solution`. Harness, `solution/`, данные и параметры обучения не патчатся.

Recorder добавляет callback на границы шага и временные обёртки Trainer/Accelerate для фаз `input`, `forward`, `loss`, `backward`, `optimizer` и `other`. Глобальные module forward hooks не используются. Перед каждым шагом и на границах фаз сбрасывается `torch.cuda.reset_peak_memory_stats()`. На закрытии каждой фазы снимаются `max_memory_allocated()` и `max_memory_reserved()`; строка шага содержит максимум среди фаз. Токены считаются по CPU `attention_mask` до переноса batch на GPU; отдельно сохраняется число токенных слотов.

После `on_train_end` чтение `torch.cuda.max_memory_allocated/reserved()` для поля `metrics.json` возвращает накопленный максимум всего прогона. Поэтому фазовые сбросы не оставляют в метриках harness только значение последней фазы.

В нашей реализации chunked CE вызывается внутри пропатченного causal-LM `forward`, поэтому её GPU-аллокации могут иметь фазу `forward`. `loss` показывает вычисления Trainer после выхода модели.

Максимумы allocated и reserved отслеживаются независимо; новый глобальный максимум любого из них вызывает `torch.cuda.memory._snapshot()` и сохраняет raw snapshot в pickle. Для него включена история аллокатора с Python stack traces и лимитом 100 000 записей через `_record_memory_history(max_entries=..., stacks="python")`. Логи начинаются с `[mem]`. Вывод содержит оба шага максимума, фазу, размер и состояние живого блока, агрегаты по полному стеку `файл:строка`, а также топ-30 блоков.

Snapshot снимается на ближайшей Python-границе фазы, где уже виден новый allocator high-water mark. Он показывает живые блоки именно на границе снимка; короткоживущий временный буфер, который освободился до этой границы, может быть виден в истории событий, но отсутствовать в списке текущих блоков. Для разбора таких событий откройте `mempeak.pickle` в [PyTorch Memory Visualizer](https://pytorch.org/memory_viz) и изучите Active Memory Timeline / Allocator State History.

Для сопоставления адресов блока со свойствами Tensor recorder сканирует Python GC и ищет CUDA tensors с тем же storage address. Для найденного объекта печатаются shape, dtype и `requires_grad`. Это частичная привязка: Tensor, сохранённый только внутри C++ autograd state, может не находиться через `gc`; в этом случае stack trace аллокации остаётся доступен, а форма и dtype будут помечены как недоступные. Snapshot PyTorch видит память своего CUDA allocator. Память, выделенная напрямую CUDA или сторонней библиотекой вне этого allocator, в нём отсутствует; это ограничение описано в [документации PyTorch о CUDA memory snapshots](https://docs.pytorch.org/docs/stable/torch_cuda_memory).

## Как запустить на Kaggle

В ноутбуке `notebook/t4_experiments9.ipynb` выберите Accelerator Tesla T4, Internet on и запустите все ячейки. Он клонирует ветку `c17/mempeak`, ставит зафиксированные версии Soup/PyTorch-стека, затем запускает `bash scripts/t4_mem2.sh dense`.

Режимы shell-обёртки:

- `bash scripts/t4_mem2.sh public` — публичный `data/train.jsonl` без изменения.
- `bash scripts/t4_mem2.sh dense` — диагностический плотный файл и тот же один solution run. `T4_MEM_OUT` позволяет задать другой каталог результатов.

Если публичные `data/train.jsonl` и `data/eval_public.jsonl` отсутствуют, wrapper вызывает `task/prepare_data.py`. Он загружает 256 публичных строк обучения и 256 строк оценки. `harness/run_task.py` передаёт путь train-файла далее как `--train-file`, поэтому режим dense не требует менять harness.

Dense-вариант воспроизводит исторический `scripts/make_edge512.py` из `origin/c5/zpack`: для каждой публичной строки повторяется её `output` до длины не менее 6000 символов; затем Soup ограничивает строку 512 токенами. Получается 256 строк, а каждый batch 4×512 заполнен без padding. Это искусственный диагностический вход, не challenge dataset для измерения результата.

Результаты по умолчанию лежат в `/kaggle/working/t4_mempeak_<mode>/runs/solution_1/`:

- `train.log` — вывод обучения и строки `[mem]`;
- `mempeak/mempeak_steps.jsonl` — allocated/reserved, tokens/slots и фазовые пики каждого шага;
- `mempeak/mempeak.pickle` — raw PyTorch snapshot, совместимый с Memory Visualizer;
- `mempeak/mempeak_meta.json` — шаг и фаза выбранного снимка.

Диагностический запуск включает прогрев harness и оценку adapter после обучения. Он запускает один solution run; повторный или полный набор checks не запускается.

## Как читать результат и выбирать следующий файл

Сначала найдите `[mem] overall_peak_allocated_step=...` и `[mem] overall_peak_reserved_step=...`; их шаги могут различаться. Сверьте соответствующие строки `mempeak_steps.jsonl` и откройте `mempeak_meta.json`, чтобы увидеть, на каком шаге и фазе был сохранён `mempeak.pickle`.

| Наблюдение в фазе/стеке | Что проверить дальше |
|---|---|
| `loss`, stack ведёт к `lm_head`, `head_patch.py`, `ce_loss.py`; крупный блок похож на `(token_chunk, vocab)` | Проверить временные буферы и градиентные чанки в `solution/ce_loss.py`, затем путь вызова в `solution/head_patch.py`. Изменение допустимо только при сохранении loss, entropy, accuracy и rewind. |
| `forward` или `backward`, стек ведёт к `layer_tail.py`, MLP, SDPA; много блоков формы `[..., intermediate_size]` | Разделить MLP/attention/residual по frame stack и dtype. MLP boundaries проходят через `solution/layer_tail.py`; fused residual norm — через `solution/add_rmsnorm.py`. Пересчёт активаций относится к `solution/recompute.py`, но его время уже требует отдельного T4 A/B. |
| Высокий объём fp32 у residual/embedding | Проверить, относятся ли блоки к residual stream или embedding/head. Граница layer forward — `solution/layer_tail.py`; компактное tied embedding/head — `solution/embed16.py`. Сохранять требования к значениям и обучению. |
| Пик именно в `optimizer`, и stack связан с gradient/state/optimizer step | Идентифицировать блок и его dtype/размер. Настройки optimizer и расписания менять нельзя; сначала установить, является ли это состоянием параметров LoRA, градиентом или временной копией. |
| Snapshot живых блоков существенно ниже `nvidia-smi`, либо прежняя разница dense/public не проявляется в `max_memory_allocated` | PyTorch snapshot не атрибутирует память вне CUDA allocator. Сопоставить `torch.cuda.memory_reserved()` с `nvidia-smi`/`torch.cuda.device_memory_used()` и проверить внешние CUDA allocations. |

Стек показывает место создания allocation, фазовая колонка — границу, в которой наблюдался соответствующий высокий watermark. Если данные не указывают на одну причину, не менять несколько файлов сразу: сначала сопоставить allocator history с формами/dtype и повторить диагностический запуск на том же файле.

## Ограничения и проверка

- **T4 запуск не выполнялся.** Проверить, что Kaggle загрузил именно `c17/mempeak` и что `train.log` содержит строки `[mem]`.
- Snapshot tracks CUDA allocations, управляемые PyTorch; он не объясняет CUDA-память вне allocator.
- Shape/dtype доступны только если живой block удалось связать с Python Tensor через storage address. Для прочих блоков остаются размер, состояние и stack.
- Инструментирование добавляет Python-работу на границах фаз и сериализацию выбранного snapshot. Числа этого запуска служат диагностике пика, а не сравнению скорости.
- Локально прошли только синтаксические проверки. Совместимость callback/allocator API с Kaggle Torch 2.11 и реальное время 5–8 минут **не проверялись**.
