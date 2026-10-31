# Почему в Kaggle `train.log` не было `[varlen]`

Проверка исходников с ветки `c18/why-no-varlen`, без GPU. Внешние API сверены с тегами Soup
`v0.77.0`, TRL `v0.29.1`, Transformers `v5.16.1`, PEFT `v0.20.0`.

## Вывод

В прежнем `solution/layer_tail.py` ранний guard содержал `or kwargs`. Для зафиксированного пути
SFT + LoRA эти kwargs непустые: PEFT `PeftModelForCausalLM.forward` вызывает базовую модель,
передавая `output_attentions=None`, `output_hidden_states=None` и `return_dict=None`; наш
`head_patch.py` передаёт их дальше в `self.model(...)`. Поэтому прежний patched
`Qwen2Model.forward` возвращал `fallback()` на первом guard до вычисления `packed_idx` и до всех
вызовов `_varlen_why` / `_varlen_note`. Это объясняет отсутствие `[varlen]` и совпавшие с обычным
путём замеры. Это вывод по потоку аргументов исходников, а не runtime-трасса старого процесса.

Добавлено безопасное удаление только этих трёх kwargs, когда их значения равны значениям по
умолчанию для этого Qwen2 config. Все остальные kwargs по-прежнему ведут к оригинальному
`forward`. Каждая причина раннего `fallback()` теперь считается отдельно; под
`SOUP_X_WARM_LOG=1` печатается первое появление каждой причины.

Отсутствие `[varlen]` в прежнем логе само по себе не доказывает, что `packed_idx` был `None` или что
`print` потерялся. Старая функция вообще не логировала причины раннего выхода на первом guard.

## Путь вызова и аргументы

Soup v0.77.0 строит `SFTTrainer` без явного `data_collator`; при включённом rewind он оборачивает
этот класс в подкласс. Rewind `compute_loss` делегирует через `super().compute_loss`, а `apply()`
успевает заменить `SFTTrainer.compute_loss` до запуска Soup. В обычной текстовой ветке Soup задаёт
PEFT `TaskType.CAUSAL_LM`, так что используется `PeftModelForCausalLM`.

Путь обучения:

```text
TRL DataLoader / collator
  -> Rewind trainer.compute_loss -> patched SFTTrainer.compute_loss
  -> transformers.Trainer.compute_loss -> model(**inputs)
  -> PEFT PeftModelForCausalLM.forward
  -> PEFT LoraModel / BaseTuner.forward -> patched Qwen2ForCausalLM.forward
  -> head_patch: self.model(...) -> patched Qwen2Model.forward
```

TRL 0.29.1 стандартный collator при `padding_free=False` возвращает `input_ids`, `labels` и
`attention_mask`; последний строится padding-ом списка масок, то есть это 2D tensor `[B,T]` с нулями
на padding. В этой задаче Soup не передаёт свой collator, `packing` не включён, а `padding_free`
имеет default `False`. Этот обычный collator не выдаёт `position_ids`; они присутствуют в его
padding-free ветке.

| На входе patched `Qwen2ForCausalLM.forward` | Значение для этого запуска |
|---|---|
| `input_ids` | Tensor `[B,T]` от collator; `inputs_embeds=None` |
| `attention_mask` | 2D Tensor `[B,T]`, не dict; `Qwen2Model` получает исходный mask до построения causal-mask mapping |
| `position_ids` | `None`: обычный collator не передаёт их |
| `past_key_values` | `None`: это train forward, generation cache не передаётся |
| `use_cache` | `False`: `head_patch.compute_loss` добавляет его в inputs перед вызовом base `Trainer.compute_loss` |
| `labels` | Tensor `[B,T]`; это именованный аргумент LM forward и дальше в `Qwen2Model` не передаётся |
| `logits_to_keep` | Значение по умолчанию `0`; это именованный аргумент LM forward и дальше не передаётся |
| `num_items_in_batch` | Transformers добавляет его в `inputs`, так как PEFT forward принимает `**kwargs`; это скалярное число не-`-100` causal labels после shift. `head_patch` извлекает его до вызова `self.model(...)` |
| другие kwargs из PEFT | `output_attentions=None`, `output_hidden_states=None`, `return_dict=None`; они проходят через head patch в `Qwen2Model.forward` |
| `cache_position` | В этом train batch не создаётся и не передаётся. Он мог бы быть произвольным kwarg в другом caller, но здесь источника для него нет |

Именно в `solution/head_patch.py:90-92` вызывается `self.model(...)` с явными
`input_ids`, `attention_mask`, `position_ids`, `past_key_values`, `inputs_embeds`, `use_cache` и
оставшимися `**kwargs`. У `Qwen2ForCausalLM` `self.model` — вложенный объект `Qwen2Model`, созданный
Transformers; вызов модуля непосредственно dispatch-ится в его patched `forward`. `labels`,
`logits_to_keep` и уже извлечённый `num_items_in_batch` в этот вызов не входят.

`PeftModelForCausalLM.forward` передаёт три выходных флага явно, даже когда значения `None`. В
`head_patch` они становятся элементами `kwargs`, потому что его сигнатура их отдельно не объявляет;
далее они попадают в `Qwen2Model.forward`. Следовательно, до внесённого изменения `or kwargs`
гарантированно срабатывал на этом источниками заданном пути, если активны указанные патчи.

`num_items_in_batch` — отдельный случай: `Trainer.compute_loss` добавляет его в `inputs` для модели
с `**kwargs`, но patched head сразу делает `kwargs.pop("num_items_in_batch", None)`. Он не является
причиной `or kwargs` в `Qwen2Model`.

## Ранние `fallback()` в `layer_tail.py`

В таблице «сработает?» относится к условию возврата, как оно было до этой правки. «Нет» значит
условие возврата ложно для зафиксированного train пути; «неизвестно» означает, что исходники
задают ожидаемое значение, но конкретная Kaggle runtime-тензорная проверка в логе не сохранена.

| Условие возврата | Сработает? | Основание |
|---|---|---|
| Не ровно один из `input_ids` / `inputs_embeds` | Нет | Collator подаёт `input_ids`; Qwen2 принимает оба входа в сигнатуре ([TRL collator](https://github.com/huggingface/trl/blob/v0.29.1/trl/trainer/sft_trainer.py#L184-L212), [Qwen2Model](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/qwen2/modeling_qwen2.py#L311-L320)). |
| `type(self) is not Qwen2Model` | Нет | Qwen2ForCausalLM создаёт `self.model = Qwen2Model(config)` ([источник](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/qwen2/modeling_qwen2.py#L374-L378)). |
| `not self.training` | Нет | Trainer вызывает `model.train()` перед loss ([источник](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/trainer.py#L1747-L1760)). |
| `not torch.is_grad_enabled()` | Нет | В train step после loss вызывается `accelerator.backward(loss)` ([источник](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/trainer.py#L1759-L1786)). |
| `self.gradient_checkpointing` | Нет | Soup schema: `default=False`; Qwen2Model тоже инициализирует поле выключенным ([Soup](https://github.com/MakazhanAlpamys/Soup/blob/v0.77.0/src/soup_cli/config/schema.py#L3172-L3182), [Transformers](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/qwen2/modeling_qwen2.py#L293-L305)). В `task/task.yaml` поле не задано. |
| `past_key_values is not None` | Нет | Стандартный collator передаёт только `input_ids`, `labels`, `attention_mask`; cache не является ключом batch ([TRL](https://github.com/huggingface/trl/blob/v0.29.1/trl/trainer/sft_trainer.py#L184-L212)). |
| `kwargs` непустой (старый `or kwargs`) | **Да** | PEFT вызывает base model с `output_attentions=...`, `output_hidden_states=...`, `return_dict=...` ([PEFT](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/peft_model.py#L1918-L1929)); локальный head их пересылает через `**kwargs` ([head patch](../../solution/head_patch.py#L77-L92)). Это детерминированная причина прежнего раннего fallback. |
| `config.output_hidden_states` | Нет | Transformers документирует default `False`; модельный config не переопределяет поле ([config defaults](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/configuration_utils.py#L180-L185), [checkpoint config](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/raw/main/config.json)). |
| `config.output_attentions` | Нет | Тот же Transformers config default `False`; checkpoint config не задаёт переопределение ([источники](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/configuration_utils.py#L180-L185), [model config](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/raw/main/config.json)). |
| `use_cache` | Нет | Head patch записывает `inputs["use_cache"] = False` ([локальный код](../../solution/head_patch.py#L160-L169)); это переопределяет checkpoint `use_cache: true` ([config](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/raw/main/config.json)). |
| Тип `attention_mask` не `Tensor/dict/None` | Нет | В обычном режиме collator строит `attention_mask = pad(...)`, то есть tensor `[B,T]` ([TRL](https://github.com/huggingface/trl/blob/v0.29.1/trl/trainer/sft_trainer.py#L183-L201)); Qwen строит causal-mask mapping уже после входа в модель ([Qwen2Model](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/qwen2/modeling_qwen2.py#L333-L349)). |
| Тип `position_ids` не `Tensor/None` | Нет | `position_ids` выдаётся collator-ом только в padding-free ветке; в обычной ветке он выдаёт mask ([TRL](https://github.com/huggingface/trl/blob/v0.29.1/trl/trainer/sft_trainer.py#L158-L168)). Qwen2 вычисляет позиции при `None` ([Qwen2Model](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/qwen2/modeling_qwen2.py#L329-L332)). |
| `layers` пуст или длина не равна `num_hidden_layers` | Нет | Qwen2Model строит `ModuleList` по `range(config.num_hidden_layers)` ([источник](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/qwen2/modeling_qwen2.py#L293-L305)). |
| `len(config.layer_types) != num_hidden_layers` | Нет | Если `layer_types` отсутствует, Qwen2Config генерирует список той же длины, что число слоёв ([источник](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/qwen2/configuration_qwen2.py#L77-L89)); checkpoint задаёт 28 слоёв ([config](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/raw/main/config.json)). |
| `hidden_size > MAX_WIDTH` | Нет | В checkpoint `hidden_size=1536`; локальный kernel limit `MAX_WIDTH=16384` ([model config](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/raw/main/config.json), [локальный MAX_WIDTH](../../solution/add_rmsnorm.py#L24-L26)). |
| Финальный `norm` не точный `Qwen2RMSNorm` | Нет | Qwen2Model конструирует финальный norm именно этим классом ([источник](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/qwen2/modeling_qwen2.py#L298-L305)). |
| `type(layer) is not Qwen2DecoderLayer` | Нет | Qwen2Model создаёт каждый элемент списка как `Qwen2DecoderLayer` ([источник](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/qwen2/modeling_qwen2.py#L298-L301)). |
| `type(layer.self_attn) is not Qwen2Attention` | Нет | Конструктор Qwen2DecoderLayer присваивает `Qwen2Attention` ([источник](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/qwen2/modeling_qwen2.py#L234-L243)). |
| `type(layer.mlp) is not Qwen2MLP` | Нет | Тот же конструктор присваивает `Qwen2MLP` ([источник](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/qwen2/modeling_qwen2.py#L234-L243)). |
| `layer.gradient_checkpointing` | Нет | `GradientCheckpointingLayer.gradient_checkpointing = False` по умолчанию ([источник](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/modeling_layers.py#L47-L72)); Soup default тоже false ([schema](https://github.com/MakazhanAlpamys/Soup/blob/v0.77.0/src/soup_cli/config/schema.py#L3172-L3182)). |
| Тип `input_layernorm` / `post_attention_layernorm` не `Qwen2RMSNorm` | Нет | Qwen2DecoderLayer создаёт оба norm объекта как `Qwen2RMSNorm` ([источник](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/qwen2/modeling_qwen2.py#L239-L243)). |
| Layer norm weights, final norm weight или embedding weight не на CUDA | Неизвестно; ожидаемо нет | Soup передаёт `device_map` в загрузчик для своего `device="cuda"` ([источник](https://github.com/MakazhanAlpamys/Soup/blob/v0.77.0/src/soup_cli/trainer/sft.py#L1550-L1559)), но ни `train.log`, ни метрики не записывают `.is_cuda` каждого параметра. T4 в runtime известен из сообщения пользователя, placement каждого слоя — нет. |
| `input_ids` не Tensor/CUDA | Условие Tensor — нет; CUDA — ожидаемо нет | Trainer рекурсивно переносит batch tensors на `args.device` ([источник](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/trainer.py#L2015-L2034)); старый лог не сохранял саму проверку `.is_cuda`. |
| `inputs_embeds` не 3D/CUDA, ширина больше 16384 или dtype вне fp16/fp32 | Shape/width/dtype — нет; CUDA — ожидаемо нет, но не залогировано | Qwen embedding вычисляется из `input_ids` ([источник](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/qwen2/modeling_qwen2.py#L321-L325)); модельная ширина 1536 ([config](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/raw/main/config.json)), локальный fast path допускает fp16/fp32 ([guard](../../solution/layer_tail.py#L187-L194)). Точное устройство результата не записано. |

Источников для «сработает» (`kwargs`) достаточно без предположений о данных: PEFT передаёт kwargs,
а `head_patch` их сохраняет кроме `num_items_in_batch`. Для CUDA условий у нас нет исторического
runtime telemetry; там указано ожидание, а не GPU-проверка.

## Почему отсутствие строки не доказывает `packed_idx is None` или потерю stdout

`_varlen_why` и `_varlen_note` вызываются только внутри пути после всех ранних guards; оба печатают
только при `SOUP_X_WARM_LOG`. Старый ранний `fallback()` не печатал ничего. Поэтому при найденном
`kwargs` возврате выполнение не могло дойти ни до подготовки `packed_idx`, ни до ветки varlen.

Если guard пройден, collator даёт подходящий 2D mask, но `packed_idx` становится `None`, когда в
данном batch нет нулей mask (все строки одной длины). Длины конкретных строк/батчей того Kaggle run
в предоставленных логах не записаны, поэтому само это условие отдельно не установлено.

Потеря stdout в harness маловероятна по исходникам: `run_logged` открывает `train.log`, передаёт его
как `stdout` дочернему процессу и объединяет `stderr` с `stdout`; `print(..., flush=True)` пишет
напрямую туда. Мы не обнаружили в указанном пути Soup/TRL перенаправления `sys.stdout`. Значит,
если `_varlen_why` действительно вызван при заданном окружении, он должен быть виден в этом файле.

## Минимальная правка

В `solution/layer_tail.py` исходные kwargs копируются для неизменного `fallback()`; перед fast-path
guard отбрасываются только безопасные значения, передаваемые PEFT:

- `output_attentions` / `output_hidden_states`: `None` или `False`; если config сам требует эти
  outputs, отдельная проверка config всё равно оставляет fast path выключенным;
- `return_dict`: `True` либо `None` только при `config.use_return_dict=True`. Если config отключил
  словарный формат, явный `return_dict=True` остаётся в kwargs и ведёт в оригинальный forward.

`return_dict=False`, `cache_position` и любые неизвестные kwargs не игнорируются: они оставляют
`kwargs` непустым и идут в оригинальный forward. Для fallback добавлен счётчик по текстовой причине;
печать первого появления каждой причины включается только при `SOUP_X_WARM_LOG`.

Эта правка позволяет Qwen2 train forward продолжить после guard, но не доказывает, что каждый batch
попадёт в `packed_idx` или что все последующие varlen-specific проверки пройдут. `scripts/t4_exp9.sh`
теперь выводит в кратком логе и `[varlen]`, и `[layer_tail]`, чтобы показать причину fallback.

## Проверка

- Успешно выполнены `python3 -m py_compile solution/layer_tail.py`, `bash -n scripts/t4_exp9.sh` и `git diff --check`.
- Pytest недоступен локально: отсутствуют `pytest` и `torch`. T4 benchmark и новый `train.log` не получены: текущая сессия не может управлять открытыми Kaggle/Colab вкладками.
- В ветке изменены `solution/layer_tail.py` и `scripts/t4_exp9.sh`; отчёт и прочие c18 заметки остаются незакоммиченными. Коммит не создан.

## Источники

- [Soup v0.77.0 `trainer/sft.py`: SFT trainer kwargs и rewind wiring](https://github.com/MakazhanAlpamys/Soup/blob/v0.77.0/src/soup_cli/trainer/sft.py#L1007-L1014), [rewind wrapper selection](https://github.com/MakazhanAlpamys/Soup/blob/v0.77.0/src/soup_cli/trainer/sft.py#L1132-L1148), [text LoRA `TaskType.CAUSAL_LM`](https://github.com/MakazhanAlpamys/Soup/blob/v0.77.0/src/soup_cli/trainer/sft.py#L1816-L1832), [gradient-checkpointing default](https://github.com/MakazhanAlpamys/Soup/blob/v0.77.0/src/soup_cli/config/schema.py#L3172-L3182).
- [Soup v0.77.0 `rewind_hf.py`: recorder `compute_loss` delegates to `super()`](https://github.com/MakazhanAlpamys/Soup/blob/v0.77.0/src/soup_cli/trainer/rewind_hf.py#L278-L310).
- [TRL v0.29.1 `sft_trainer.py`: default collator selection](https://github.com/huggingface/trl/blob/v0.29.1/trl/trainer/sft_trainer.py#L771-L787), [collator outputs and mask construction](https://github.com/huggingface/trl/blob/v0.29.1/trl/trainer/sft_trainer.py#L151-L212).
- [TRL v0.29.1 `sft_config.py`: packing and padding-free defaults](https://github.com/huggingface/trl/blob/v0.29.1/trl/trainer/sft_config.py#L64-L75), [field defaults](https://github.com/huggingface/trl/blob/v0.29.1/trl/trainer/sft_config.py#L166-L190).
- [Transformers v5.16.1 `trainer.py`: model kwargs support and `num_items_in_batch`](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/trainer.py#L464-L470), [compute_loss model call](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/trainer.py#L1813-L1825), [batch tensors moved to device](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/trainer.py#L2015-L2034).
- [PEFT v0.20.0 `PeftModelForCausalLM.forward`](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/peft_model.py#L1888-L1929), [LoRA tuner base forward](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/tuners_utils.py#L300-L308).
- [Transformers v5.16.1 Qwen2Model construction and forward signature](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/qwen2/modeling_qwen2.py#L293-L320), [Qwen2ForCausalLM signature and `self.model(...)`](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/qwen2/modeling_qwen2.py#L384-L419), [decoder layer concrete submodules](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/qwen2/modeling_qwen2.py#L234-L243).
- [Transformers v5.16.1 output defaults](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/configuration_utils.py#L180-L185), [Qwen2 layer-types default construction](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/qwen2/configuration_qwen2.py#L77-L89), [checkpoint layer default](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/modeling_layers.py#L47-L72).
- [Qwen2.5-1.5B-Instruct config: hidden size, layers, and attention fields](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/raw/main/config.json).
- Local logging route: [`solution/layer_tail.py`](../../solution/layer_tail.py#L11), [`harness/common.py`](../../harness/common.py#L103), [`harness/run_task.py`](../../harness/run_task.py#L96).
