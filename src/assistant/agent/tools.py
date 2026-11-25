"""Инструменты агента. Требование FR-6."""

from collections.abc import Callable

from assistant.graphrag.search import global_search, local_search
from assistant.infra.erp_client import ErpClient, ErpNotFound, ErpRejected
from assistant.infra.grounding import canonical_name
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import KnowledgeBase

WAREHOUSES = ("Москва", "Санкт-Петербург", "Казань", "Новосибирск")
_QUESTION = {"type": "string", "description": "Вопрос по документам сервисной службы"}
_PART_ID = {"type": "string", "description": "Номер детали, например P-1042B", "pattern": r"^P-[0-9]{4}[A-Z]?$"}
_WAREHOUSE = {"type": "string", "enum": list(WAREHOUSES)}


def _schema(name: str, description: str, **properties: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            },
        },
    }


TOOL_SCHEMAS = [
    _schema("search_knowledge", "Найти факты об ошибке, модели, детали и её заменах в документах.", question=_QUESTION),
    _schema("summarize_topic", "Ответить на обзорный вопрос по сводкам документов.", question=_QUESTION),
    _schema("check_stock", "Проверить остаток детали на выбранном складе ERP.", part_id=_PART_ID, warehouse=_WAREHOUSE),
    _schema("get_price", "Узнать актуальную цену детали в ERP.", part_id=_PART_ID),
    _schema(
        "create_service_order",
        "Создать заказ только по явной просьбе пользователя с указанным им количеством.",
        part_id=_PART_ID,
        warehouse=_WAREHOUSE,
        qty={"type": "integer", "minimum": 1},
        model={"type": "string", "description": "Код модели оборудования, например AR-210"},
    ),
]


def build_tools(kb: KnowledgeBase, llm: LLMClient, erp: ErpClient) -> dict[str, Callable[..., dict]]:
    """Связывает вызовы агента с графом знаний и ERP."""

    def search_knowledge(question: str) -> dict:
        if not isinstance(question, str) or not question.strip():
            return {"error": "Укажите текст вопроса."}
        answer = local_search(question, kb, llm)
        return {"answer": answer.text, "facts": answer.facts, "sources": answer.sources, "refused": answer.refused}

    def summarize_topic(question: str) -> dict:
        if not isinstance(question, str) or not question.strip():
            return {"error": "Укажите текст вопроса."}
        answer = global_search(question, kb, llm)
        return {"answer": answer.text, "sources": answer.sources, "refused": answer.refused}

    def check_stock(part_id: str, warehouse: str) -> dict:
        return _erp_call(erp.get_stock, part_id, warehouse=warehouse)

    def get_price(part_id: str) -> dict:
        return _erp_call(erp.get_price, part_id)

    def create_service_order(part_id: str, warehouse: str, qty: int, model: str) -> dict:
        if type(qty) is not int or qty < 1:
            return {"error": "Количество должно быть целым положительным числом."}
        code = canonical_name("Model", model, {}) if isinstance(model, str) else None
        if code is None:
            return {"error": "Укажите код модели, например AR-210."}
        return _erp_call(erp.create_order, part_id, warehouse=warehouse, qty=qty, model=code)

    return {
        "search_knowledge": search_knowledge,
        "summarize_topic": summarize_topic,
        "check_stock": check_stock,
        "get_price": get_price,
        "create_service_order": create_service_order,
    }


def _erp_call(method: Callable[..., dict], part_id: str, **args) -> dict:
    part = canonical_name("Part", part_id, {}) if isinstance(part_id, str) else None
    if part is None:
        return {"error": "Укажите номер детали, например P-1042B."}
    if "warehouse" in args and args["warehouse"] not in WAREHOUSES:
        return {"error": f"Допустимые склады: {', '.join(WAREHOUSES)}."}
    try:
        return method(part, **args)
    except ErpNotFound as exc:
        return {"error": f"Данных нет: {exc}"}
    except ErpRejected as exc:
        return {"error": f"ERP отклонила запрос: {exc}"}
