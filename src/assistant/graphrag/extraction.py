"""Извлечение сущностей и связей из чанка. Требование FR-1."""

import json
import re

from assistant.infra.config import vocabulary
from assistant.infra.grounding import (
    CODE_TYPES,
    canonical_name,
    find_codes,
)
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import (
    ENTITY_TYPES,
    EXTRACTION_SCHEMA,
    RELATION_SIGNATURES,
    Chunk,
    Entity,
    ExtractionResult,
    Relation,
)

EXTRACTION_PROMPT = """
Ты извлекаешь сущности и связи 
из фрагметнов текста.

Допустимые типы сущностей:
- ProductLine: линейка 
- Model: код модели 
- Component: название узла 
- Part: номер детали 
- ErrorCode: код ошибки
- Procedure: номер регламента 
- Supplier: название поставщика

Допустимые связи и их направления:
- BELONGS_TO: Model -> ProductLine
- HAS_PART: Model -> Part
- IS_A: Part -> Component
- INDICATES: ErrorCode -> Component
- FIXED_BY: ErrorCode -> Procedure
- REPLACED_BY: Part -> Part
- SUPPLIED_BY: Part -> Supplier
- FAILED: Model -> Component

Для REPLACED_BY направление всегда
от старой детали к новой.

Извлекай только факты, подтверждённые текстом.
Не придумывай сущности и связи.
Если фактов нет, верни пустые массивы.

Пример текста:
"Деталь P-1042 заменена на P-1042B."

Пример ответа:
{
  "entities": [
    {"type": "Part", "name": "P-1042"},
    {"type": "Part", "name": "P-1042B"}
  ],
  "relations": [
    {
      "subject_type": "Part",
      "subject": "P-1042",
      "type": "REPLACED_BY",
      "object_type": "Part",
      "object": "P-1042B"
    }
  ]
}
"""


MANUAL_PARTS_GUIDANCE = """
Для каждой строки «узел – деталь КОД, поставщик НАЗВАНИЕ» извлеки три связи.
Код модели возьми из заголовка или фразы перед списком. Копируй артикул целиком.
Пример для AR-999 и строки «контроллер – деталь P-8888, поставщик „АвтоматикаСевер“»:
{"entities":[],"relations":[
{"subject_type":"Model","subject":"AR-999","type":"HAS_PART","object_type":"Part","object":"P-8888"},
{"subject_type":"Part","subject":"P-8888","type":"IS_A","object_type":"Component","object":"контроллер"},
{"subject_type":"Part","subject":"P-8888","type":"SUPPLIED_BY","object_type":"Supplier","object":"АвтоматикаСевер"}
]}
Добавь BELONGS_TO, если модель прямо отнесена к линейке. Указывай в entities только
сущности, которых ещё нет в relations. Верни компактный JSON без пояснений.
"""

BULLETIN_GUIDANCE = """
В бюллетене копируй артикулы посимвольно, включая последнюю цифру и букву.
Если деталь снята, а вместо неё устанавливается другая, REPLACED_BY идёт от снятой
к установленной. Если рядом с артикулом указан узел, добавь IS_A.
Перед ответом проверь, что каждый артикул есть в тексте в точности.
"""

MANUAL_PART_LINE = re.compile(r"^- (.+?)\s+[–—-]\s+деталь\s+(P-\d+[A-Z]*)\b")
PROCEDURE_RULE = re.compile(r"^Регламент (.+?) применяется при ошибке ([^.\n]+)\.", re.M)
REPORT_MODEL = re.compile(r"^Оборудование:\s*([^.\n]+)", re.M)
REPORT_FAILURE = re.compile(r"^Диагностика показала отказ узла:\s*([^.\n]+)", re.M)


def parse_response(content: str) -> dict:

    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("Некорректный JSON") from exc


    if not isinstance(data, dict):
        raise ValueError("Ожидался JSON-объект")

    if not isinstance(data.get("entities"), list):
        raise ValueError("entities должен быть списком")

    if not isinstance(data.get("relations"), list):
        raise ValueError("relations должен быть списком")

    for item in data["entities"]:
        if not isinstance(item, dict):
            raise ValueError("Сущность должна быть объектом")

        if not all(
            isinstance(item.get(key), str)
            for key in ("type", "name")
        ):
            raise ValueError("Неверные поля сущности")

        if item["type"] not in ENTITY_TYPES:
            raise ValueError("Неизвестный тип сущности")


    relation_fields = (
        "subject_type",
        "subject",
        "type",
        "object_type",
        "object",
    )

    for item in data["relations"]:
        if not isinstance(item, dict):
            raise ValueError("Связь должна быть объектом")

        if not all(
            isinstance(item.get(key), str)
            for key in relation_fields
        ):
            raise ValueError("Неверные поля связи")

        if (
            item["subject_type"] not in ENTITY_TYPES
            or item["object_type"] not in ENTITY_TYPES
            or item["type"] not in RELATION_SIGNATURES
        ):
            raise ValueError("Неизвестный тип в связи")

    return data

def _validate_entities(
    raw_entities: list[dict],
    text: str,
    vocab: dict[str, list[str]],
) -> tuple[list[Entity], int]:

    codes_in_text = set(find_codes(text))

    valid_entities = []
    seen = set()
    dropped = 0

    for raw in raw_entities:
        entity_type = raw["type"]
        raw_name = raw["name"]
        name = canonical_name(
            entity_type,
            raw_name,
            vocab,
        )
        if name is None:
            dropped += 1
            continue
        if (
            entity_type in CODE_TYPES
            and name not in codes_in_text
        ):
            dropped += 1
            continue
        entity = Entity(
            type=entity_type,
            name=name,
        )

        if entity.node_id not in seen:
            valid_entities.append(entity)
            seen.add(entity.node_id)

    return valid_entities, dropped



def _manual_pairs(text: str, vocab: dict[str, list[str]]) -> set[tuple[str, str, str]]:
    title, _, body = text.partition("\n")
    assembly = body.partition("## Состав узлов")[2].split("\n## ", 1)[0]
    model_codes = find_codes(title)
    if not assembly or len(model_codes) != 1:
        return set()
    pairs = set()
    for line in assembly.splitlines():
        match = MANUAL_PART_LINE.match(line)
        if match:
            label, part = match.groups()
            pairs.add((model_codes[0], "HAS_PART", part))
            component = canonical_name("Component", label, vocab)
            if component:
                pairs.add((part, "IS_A", component))
    return pairs


def _explicit_relations(chunk: Chunk, vocab: dict[str, list[str]]) -> list[dict]:
    """Восстанавливает связи из явно размеченных полей документа."""
    pairs = set()
    if chunk.doc_id.startswith("manual_"):
        pairs = _manual_pairs(chunk.text, vocab)
    elif chunk.doc_id.startswith("procedure_"):
        for raw_procedure, raw_error in PROCEDURE_RULE.findall(chunk.text):
            procedures, errors = find_codes(raw_procedure), find_codes(raw_error)
            if len(procedures) == len(errors) == 1:
                pairs.add((errors[0], "FIXED_BY", procedures[0]))
    elif chunk.doc_id.startswith("report_"):
        equipment = REPORT_MODEL.search(chunk.text)
        diagnosis = REPORT_FAILURE.search(chunk.text)
        if equipment and diagnosis:
            models = find_codes(equipment[1])
            component = canonical_name("Component", diagnosis[1], vocab)
            if len(models) == 1 and component:
                pairs.add((models[0], "FAILED", component))

    return [
        {
            "subject_type": RELATION_SIGNATURES[kind][0],
            "subject": subject,
            "type": kind,
            "object_type": RELATION_SIGNATURES[kind][1],
            "object": obj,
        }
        for subject, kind, obj in sorted(pairs)
    ]


def _validate_relations(
    raw_relations: list[dict],
    entities: list[Entity],
    text: str,
    vocab: dict[str, list[str]],
) -> tuple[list[Entity], list[Relation], int]:

    codes_in_text = set(find_codes(text))

    valid_entities = list(entities)

    entities_by_key = {
        (entity.type, entity.name): entity
        for entity in valid_entities
    }

    valid_relations = []
    seen = set()
    dropped = 0
    manual_pairs = _manual_pairs(text, vocab)

    for raw in raw_relations:
        subject_type = raw["subject_type"]
        object_type = raw["object_type"]


        subject_name = canonical_name(
            subject_type,
            raw["subject"],
            vocab,
        )

        object_name = canonical_name(
            object_type,
            raw["object"],
            vocab,
        )


        if subject_name is None or object_name is None:
            dropped += 1
            continue

        if (
            (subject_type in CODE_TYPES
             and subject_name not in codes_in_text)
            or
            (object_type in CODE_TYPES
             and object_name not in codes_in_text)
        ):
            dropped += 1
            continue

        relation_type = raw["type"]
        # Исправляем перестановку концов только при явном подтверждении строкой комплектации.
        if (
            RELATION_SIGNATURES[relation_type] == (object_type, subject_type)
            and (object_name, relation_type, subject_name) in manual_pairs
        ):
            subject_type, object_type = object_type, subject_type
            subject_name, object_name = object_name, subject_name

        subject_key = (subject_type, subject_name)
        object_key = (object_type, object_name)

        subject = entities_by_key.get(subject_key)
        obj = entities_by_key.get(object_key)

        # Если сущности ещё нет, создаём её
        if subject is None:
            subject = Entity(subject_type, subject_name)

        if obj is None:
            obj = Entity(object_type, object_name)


        relation = Relation(
            subject=subject,
            type=relation_type,
            object=obj,
        )

        if not relation.matches_schema():
            dropped += 1
            continue

        for entity in (subject, obj):
            key = (entity.type, entity.name)

            if key not in entities_by_key:
                entities_by_key[key] = entity
                valid_entities.append(entity)

        key = (
            subject.node_id,
            relation.type,
            obj.node_id,
        )

        if key not in seen:
            valid_relations.append(relation)
            seen.add(key)

    return valid_entities, valid_relations, dropped



def extract_chunk(chunk: Chunk, llm: LLMClient) -> ExtractionResult:
    """Возвращает сущности и связи чанка строго по схеме графа.

    Модель вызывается через `llm.chat(..., json_schema=EXTRACTION_SCHEMA)`. Запрос описывает схему
    с направлением каждой связи и содержит хотя бы один пример. Всё, что не подтверждается текстом,
    отбрасывается и считается в `dropped`: код, которого нет в чанке, название не из словаря,
    связь с отброшенным концом. Пригодятся `find_codes` и `canonical_name` из `assistant.infra.grounding`.
    Функция не падает ни на каком ответе модели: одна повторная попытка, затем пустой результат.
    """
    # TODO: запрос к модели, разбор JSON, проверки по тексту чанка и словарю

    vocab = vocabulary()
    prompt = EXTRACTION_PROMPT
    max_tokens = 1024
    if "## Состав узлов" in chunk.text:
        prompt += MANUAL_PARTS_GUIDANCE
        max_tokens = 3072
    elif chunk.doc_id.startswith("bulletin_"):
        prompt += BULLETIN_GUIDANCE
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": chunk.text},
    ]

    for attempt in range(2):
        response = llm.chat(messages=messages, json_schema=EXTRACTION_SCHEMA, max_tokens=max_tokens)
        try:
            data = parse_response(response.content)
        except ValueError:
            if attempt == 0:
                messages = [
                    *messages,
                    {"role": "assistant", "content": str(response.content)},
                    {"role": "user", "content": (
                        "Предыдущий ответ не соответствует JSON-схеме. "
                        "Исправь формат и верни JSON с массивами entities и relations."
                    )},
                ]
            continue

        entities, dropped_entities = _validate_entities(data["entities"], chunk.text, vocab)
        entities, relations, dropped_relations = _validate_relations(
            [*data["relations"], *_explicit_relations(chunk, vocab)], entities, chunk.text, vocab
        )
        return ExtractionResult(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            entities=entities,
            relations=relations,
            dropped=dropped_entities + dropped_relations,
        )

    return ExtractionResult(chunk_id=chunk.chunk_id, doc_id=chunk.doc_id)



