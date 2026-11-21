"""Цикл агента (FR-7–FR-10)."""

import json
import re
from collections.abc import Callable
from decimal import Decimal
from inspect import signature

from assistant.agent.tools import TOOL_SCHEMAS, WAREHOUSES
from assistant.infra.erp_client import ErpNotFound, ErpRejected, ErpUnavailable
from assistant.infra.grounding import code_type, find_codes, normalize_code
from assistant.infra.llm import LLMClient, ToolCall, tool_message
from assistant.infra.schemas import AgentResult, ToolStep

_SYSTEM = (
    "Ты ассистент сервисной службы. Найди актуальную деталь через search_knowledge, если номер неизвестен. "
    "Для обзоров используй summarize_topic, для цены и наличия — ERP. Различай нулевой остаток, "
    "отсутствие данных и недоступность ERP. Создавай заказ только по явной просьбе с указанным количеством. "
    "Отвечай по-русски по результатам инструментов. Если данных нет, скажи: В базе знаний не найдено."
)
_REFUSAL = "В базе знаний не найдено."
_UNAVAILABLE = "Данные ERP получить не удалось."
_ORDER = "create_service_order"
_ERP = {"check_stock", "get_price", _ORDER}

_NUMBER = r"(?<![\w-])[−-]?(?:\d{1,3}(?:[ \u00a0\u202f]\d{3})+|\d+)(?:[,.]\d+)?(?!\d|[.,]\d)"
_STANDALONE = rf"{_NUMBER}(?![\w-]|[ \u00a0\u202f]\d{{3}}\b)"
_RUBLES = r"(?:₽|руб(?:л(?:ь|я|ей))?\b)"
_PIECES = r"шт(?:ук(?:а|и)?)?\b"
_PRICE = re.compile(rf"({_NUMBER})\s*{_RUBLES}|\b(?:цен[ауы]|стоит)\s*[:=–—]?\s*({_STANDALONE})", re.I)
_STOCK = re.compile(
    rf"({_NUMBER})\s*{_PIECES}|\b(?:остаток|в наличии|на складе(?: в [\w-]+)?)\s*[:=–—]?\s*({_STANDALONE})",
    re.I,
)
_ORDER_ID = re.compile(r"\bSO-\d+\b", re.I)
_QUANTITY_WORDS = {
    word: count
    for count, word in enumerate("один два три четыре пять шесть семь восемь девять десять".split(), 1)
}
_QUANTITY_WORDS.update(одна=1, одну=1, две=2)
_WORD = rf"\b(?:{'|'.join(_QUANTITY_WORDS)})\b"
_QUANTITY = re.compile(
    rf"({_NUMBER}|{_WORD})\s*(?:{_PIECES}|детал(?:ь|и|ей)\b)|"
    rf"\b(?:закаж\w*|заказать|оформ\w*(?:\s+заказ)?|количество)\s*[:=–—]?\s*"
    rf"({_STANDALONE}|{_WORD})(?!\s*{_RUBLES})",
    re.I,
)


def run_agent(
    question: str,
    llm: LLMClient,
    tools: dict[str, Callable[..., dict]],
    *,
    max_steps: int = 6,
    confirm: bool = False,
) -> AgentResult:
    """Выполняет вызовы модели и возвращает ответ с журналом."""
    messages = [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": question}]
    result = AgentResult(answer="")
    codes = set(find_codes(question))
    cache: dict[tuple[str, str], dict] = {}
    answer = None

    while len(result.steps) < max_steps:
        response = llm.chat(messages, tools=TOOL_SCHEMAS)
        messages.append(response.as_message())
        if not response.tool_calls:
            answer = response.content.strip()
            break

        remaining = max_steps - len(result.steps)
        for call in response.tool_calls[:remaining]:
            args = call.arguments
            tool = tools.get(call.name)
            error = _argument_error(call, tool, codes)
            if error:
                data = {"error": error}
            else:
                assert tool is not None
                if call.name == _ORDER:
                    if args["qty"] not in _numbers(question, _QUANTITY):
                        result.answer = "Уточните, сколько штук детали нужно заказать."
                        result.steps.append(ToolStep(_ORDER, args, error=result.answer))
                        return result
                    if not confirm:
                        result.answer = f"Подтвердите заказ детали {args['part_id']}."
                        result.needs_confirmation = True
                        result.pending_action = {"tool": _ORDER, "args": args}
                        result.steps.append(ToolStep(_ORDER, args, error="Требуется подтверждение"))
                        return result
                key = (call.name, json.dumps(args, sort_keys=True))
                if key not in cache:
                    cache[key] = _call_tool(tool, args)
                data = cache[key]

            result.steps.append(ToolStep(call.name, args, data, data.get("error")))
            messages.append(tool_message(call, data))
            if "error" not in data and not data.get("refused"):
                codes.update(find_codes(json.dumps(data, ensure_ascii=False)))
                result.sources = list(dict.fromkeys([*result.sources, *data.get("sources", [])]))

    return _finish(result, answer, question, codes, messages, llm)


def _argument_error(call: ToolCall, tool: Callable[..., dict] | None, codes: set[str]) -> str | None:
    if tool is None:
        return "Неизвестный инструмент."
    args = call.arguments
    if not isinstance(args, dict) or "_raw" in args:
        return "Не удалось прочитать аргументы инструмента."
    try:
        signature(tool).bind(**args)
    except TypeError:
        return "Неверные аргументы инструмента."

    if "part_id" in args or call.name in _ERP:
        raw_part = args.get("part_id")
        part = normalize_code(raw_part) if isinstance(raw_part, str) else None
        if not part or not part.startswith("P-") or part not in codes:
            return "Укажите номер детали из вопроса или результатов инструментов. Вызовите search_knowledge."
    if call.name == _ORDER:
        if args.keys() != {"part_id", "warehouse", "qty", "model"} or type(args["qty"]) is not int or args["qty"] <= 0:
            return "Для заказа укажите деталь, склад, модель и целое положительное количество."
        raw_model = args["model"]
        model = normalize_code(raw_model) if isinstance(raw_model, str) else None
        if args["warehouse"] not in WAREHOUSES or not model or code_type(model) != "Model":
            return "Для заказа укажите допустимый склад и код модели."
    return None


def _call_tool(tool: Callable[..., dict], args: dict) -> dict:
    for _ in range(3):
        try:
            return tool(**args)
        except ErpUnavailable:
            continue
        except ErpNotFound as exc:
            return {"error": f"Данных нет: {exc}"}
        except ErpRejected as exc:
            return {"error": f"ERP отклонила запрос: {exc}"}
    return {"error": _UNAVAILABLE}


def _numbers(text: str, pattern: re.Pattern) -> set[Decimal]:
    values = set()
    for first, second in pattern.findall(text.lower()):
        term = first or second
        if term in _QUANTITY_WORDS:
            values.add(Decimal(_QUANTITY_WORDS[term]))
        else:
            values.add(Decimal("".join(term.split()).replace(",", ".").replace("−", "-")))
    return values


def _finish(
    result: AgentResult, answer: str | None, question: str, codes: set[str], messages: list[dict], llm: LLMClient
) -> AgentResult:
    prices, stock = _numbers(question, _PRICE), _numbers(question, _STOCK)
    orders = set()
    partial = []
    missing_data = False
    for step in result.steps:
        data = step.result or {}
        if data.get("refused") or (step.tool in _ERP and re.search(r"данных нет|не найд", step.error or "", re.I)):
            missing_data = True
        if step.error or data.get("refused"):
            continue
        if data.get("answer"):
            partial.append(data["answer"])
        if step.tool == "get_price" and "price_rub" in data:
            prices.add(data["price_rub"])
            partial.append(f"{step.args['part_id']}: {data['price_rub']} руб.")
        if step.tool == "check_stock" and "qty" in data:
            stock.add(data["qty"])
            partial.append(f"{step.args['part_id']}, склад {step.args['warehouse']}: {data['qty']} шт.")
        if step.tool == _ORDER and data.get("order_id"):
            orders.add(data["order_id"].upper())
    if orders:
        stock.update(_numbers(question, _QUANTITY))

    no_data = missing_data and not partial and not orders
    if answer is None:
        result.refused = no_data
        if no_data:
            partial.append(_REFUSAL)
        answer = " ".join([*dict.fromkeys(partial), "Достигнут лимит вызовов инструментов. Работа прервана."])
    else:
        if no_data:
            answer = _REFUSAL
        if not _grounded(answer, codes, prices, stock, orders):
            messages.append({"role": "user", "content": "Исправь ответ. Используй только подтверждённые коды и числа."})
            answer = llm.chat(messages).content.strip()
            if not _grounded(answer, codes, prices, stock, orders):
                answer = _REFUSAL
        result.refused = not answer or answer.rstrip(".! ").casefold() == _REFUSAL[:-1].casefold()
        if result.refused:
            answer = _REFUSAL

    if any(step.error == _UNAVAILABLE for step in result.steps) and _UNAVAILABLE.casefold() not in answer.casefold():
        answer += f" {_UNAVAILABLE}"
    for order in sorted(orders - {match.upper() for match in _ORDER_ID.findall(answer)}):
        answer += f" Заказ {order} создан."
    result.answer = answer
    return result


def _grounded(answer: str, codes: set[str], prices: set[Decimal], stock: set[Decimal], orders: set[str]) -> bool:
    return (
        set(find_codes(answer)) <= codes
        and _numbers(answer, _PRICE) <= prices
        and _numbers(answer, _STOCK) <= stock
        and set(_ORDER_ID.findall(answer.upper())) <= orders
    )
