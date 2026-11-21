"""Инструменты агента. Требование FR-6."""

from collections.abc import Callable

from assistant.infra.erp_client import ErpClient
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
    """Словарь «имя инструмента -> функция». Каждая функция возвращает словарь.

    Функция проверяет свои аргументы до обращения к ERP: склад из списка, номер детали вида P-1042
    или P-1042B, количество больше нуля. При ошибке возвращает {"error": "..."} и ничего не выполняет.
    `ErpNotFound` и `ErpRejected` превращаются в {"error": "..."} с разным текстом, чтобы исходы
    оставались различимыми. `ErpUnavailable` передаётся дальше: повторы делает агент (FR-8).
    """
    # TODO: пять функций поверх local_search, global_search и клиента ERP
    raise NotImplementedError("Реализуйте build_tools")
