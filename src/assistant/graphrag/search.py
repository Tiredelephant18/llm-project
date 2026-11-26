"""Локальный поиск по графу и обзорный поиск по сводкам."""

import json
from collections import Counter
from collections.abc import Iterable
from itertools import combinations, pairwise

import networkx as nx

from assistant.infra.grounding import code_type, find_codes, find_names
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import MAP_SCHEMA, Answer, KnowledgeBase

_NOT_FOUND = "В базе знаний не найдено."
_LOCAL_PROMPT = (
    "Ответь на вопрос только по приведённым фактам. Учитывай направление связей и модель из вопроса. "
    "HAS_PART указывает исходную деталь модели, IS_A — её вид, INDICATES связывает ошибку и узел, "
    "FIXED_BY — ошибку и регламент, SUPPLIED_BY — деталь и поставщика. "
    "REPLACED_BY направлена от старой детали к новой. Для вопросов о замене, выборе детали и текущем поставщике "
    "пройди всю цепочку замены нужной детали и назови конечную деталь. Для вопросов об исходной комплектации "
    "используй HAS_PART. Для сравнений по отказам используй количество, указанное для связи FAILED. "
    f"Если фактов для ответа недостаточно, напиши: {_NOT_FOUND}"
)


def local_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Находит связи около сущностей из вопроса и передаёт их модели."""
    anchors = _anchors(question, kb) or set()
    edges = _nearby_edges(kb.graph, anchors, question)
    facts, sources = _local_context(kb.graph, edges)
    if not facts:
        return Answer(_NOT_FOUND, refused=True)
    context = "\n".join(f"{i}. {fact}" for i, fact in enumerate(facts, 1))
    prompt = _LOCAL_PROMPT
    kinds = {node.split(":", 1)[0] for node in anchors}
    if {"Model", "ErrorCode"} <= kinds and "-REPLACED_BY->" in context:
        prompt += (
            " В ответе о детали, соответствующей ошибке, первым называй актуальный артикул. "
            "Используй исходный артикул только для объяснения цепочки замены или если запросили исходную деталь."
        )
    text = _ask(llm, prompt, question, context)
    if set(find_codes(text)) - set(find_codes(f"{question}\n{context}")):
        text = _NOT_FOUND
    answer = _answer(text, sources, facts)
    current = _current_part(kb.graph, anchors, question)
    if current and current in find_codes(context) and (answer.refused or current not in find_codes(text)):
        return Answer(f"Актуальная деталь — {current}.", sources=sources, facts=facts)
    return answer


def _current_part(graph: nx.DiGraph, anchors: set[str], question: str) -> str | None:
    """Выводит артикул, когда модель, узел и маршрут замены имеют однозначную связь."""
    query = question.casefold()
    if any(stem in query for stem in ("постав", "регламент", "исходн", "штатн", "изначальн", "старый", "старая")):
        return None

    parts = {node for node in anchors if node.startswith("Part:")}
    if parts:
        if len(anchors) != 1 or "замен" not in query:
            return None
    else:
        models = {node for node in anchors if node.startswith("Model:")}
        if len(models) != 1 or not any(stem in query for stem in ("детал", "артикул")):
            return None
        nodes = {node for node in anchors if node.startswith("Component:")}
        errors = {node for node in anchors if node.startswith("ErrorCode:")}
        if errors:
            indicated = {target for _, target, kind in graph.out_edges(errors, data="type") if kind == "INDICATES"}
            nodes = nodes & indicated if nodes else indicated
        if len(nodes) != 1:
            return None
        parts = {
            part for _, part, kind in graph.out_edges(models, data="type")
            if kind == "HAS_PART" and any(
                target in nodes and relation == "IS_A" for _, target, relation in graph.out_edges(part, data="type")
            )
        }

    if len(parts) != 1:
        return None
    part = start = next(iter(parts))
    seen = set()
    while part not in seen:
        seen.add(part)
        replacements = [target for _, target, kind in graph.out_edges(part, data="type") if kind == "REPLACED_BY"]
        if not replacements:
            if part == start and start in anchors:
                return None
            return graph.nodes[part]["name"]
        if len(replacements) != 1:
            return None
        part = replacements[0]
    return None


def _anchors(question: str, kb: KnowledgeBase) -> set[str] | None:
    anchors = {f"{code_type(code)}:{code}" for code in find_codes(question)}
    if anchors.difference(kb.graph):
        return None
    anchors.update(f"{kind}:{name}" for kind, name in find_names(question, kb.vocabulary))
    return anchors.intersection(kb.graph.nodes)


def _nearby_edges(graph: nx.DiGraph, anchors: set[str], question: str) -> set[tuple[str, str]]:
    if not anchors:
        return set()

    nearby = graph.to_undirected(as_view=True)
    if len(anchors) == 1:
        distances = nx.single_source_shortest_path_length(nearby, next(iter(anchors)), cutoff=2)
        edges = set(graph.subgraph(distances).edges)
    else:
        edges = set()
        for start, end in combinations(sorted(anchors), 2):
            for path in nx.all_simple_paths(nearby, start, end, cutoff=3):
                for left, right in pairwise(path):
                    if graph.has_edge(left, right):
                        edges.add((left, right))
                    if graph.has_edge(right, left):
                        edges.add((right, left))
    query = question.casefold()
    supplier = any(stem in query for stem in ("поставщик", "поставля", "поставк"))
    if len(anchors) > 1 or supplier or any(stem in query for stem in ("замен", "снят", "снял", "актуальн")):
        _add_replacements(graph, edges, supplier)
    return edges


def _add_replacements(graph: nx.DiGraph, edges: set[tuple[str, str]], supplier: bool) -> None:
    pending = [node for node in {node for edge in edges for node in edge} if node.startswith("Part:")]
    seen = set(pending)
    while pending:
        part = pending.pop()
        for _, target, data in graph.out_edges(part, data=True):
            if data["type"] == "REPLACED_BY":
                edges.add((part, target))
                if target not in seen:
                    seen.add(target)
                    pending.append(target)
            elif supplier and data["type"] == "SUPPLIED_BY":
                edges.add((part, target))


def _local_context(graph: nx.DiGraph, edges: set[tuple[str, str]]) -> tuple[list[str], list[str]]:
    facts, chunks = [], []
    size = 0
    # Замены идут первыми, чтобы их не обрезал лимит контекста.
    for source, target in sorted(edges, key=lambda edge: (graph.edges[edge]["type"] != "REPLACED_BY", edge)):
        data = graph.edges[source, target]
        kind = data["type"]
        fact = f"{graph.nodes[source]['name']} -{kind}-> {graph.nodes[target]['name']}"
        if kind == "FAILED":
            fact += f" (случаев: {data['weight']})"
        line_size = len(f"{len(facts) + 1}. {fact}") + (1 if facts else 0)
        if size + line_size > 8000:
            break
        size += line_size
        facts.append(fact)
        chunks.extend(data.get("sources", []))
    return facts, _documents(chunks)


def _line_context(graph: nx.DiGraph, lines: set[str]) -> tuple[list[str], list[str]]:
    """Собирает данные по линейке, включая рёбра между сообществами."""
    facts, links = [], set()
    for line in sorted(lines):
        by_model, by_component = Counter(), Counter()
        for model, _, membership in graph.in_edges(line, data=True):
            if membership["type"] != "BELONGS_TO":
                continue
            for _, target, relation in graph.out_edges(model, data=True):
                if relation["type"] == "FAILED":
                    weight = relation["weight"]
                    by_model[graph.nodes[model]["name"]] += weight
                    by_component[graph.nodes[target]["name"]] += weight
                    links.add((model, target))
                    links.add((model, line))
                elif relation["type"] == "HAS_PART":
                    for _, new, replacement in graph.out_edges(target, data=True):
                        if replacement["type"] == "REPLACED_BY":
                            links.update(((model, line), (model, target), (target, new)))
        for label, counts in (("по узлам", by_component), ("по моделям", by_model)):
            if counts:
                totals = ", ".join(f"{name}: {count}" for name, count in sorted(counts.items()))
                highest = max(counts.values())
                leaders = ", ".join(sorted(name for name, weight in counts.items() if weight == highest))
                facts.append(
                    f"Линейка {graph.nodes[line]['name']}, суммы весов FAILED {label}: {totals}. "
                    f"Наибольший вес {label}: {leaders} ({highest})."
                )
    relations, sources = _local_context(graph, links)
    return facts + relations, sources


def global_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Получает частичные ответы по сводкам и объединяет полезные."""
    anchors = _anchors(question, kb)
    if anchors is None:
        return Answer(_NOT_FOUND, refused=True)
    lines = {node for node in anchors if node.startswith("ProductLine:")}
    if lines and lines == anchors:
        facts, fact_sources = _line_context(kb.graph, lines)
    else:
        facts, fact_sources = _local_context(kb.graph, _nearby_edges(kb.graph, anchors, question))
    evidence = "\n".join(facts)
    communities = [c for c in kb.communities if c.summary.strip()]
    focus = max(communities, key=lambda c: len(anchors.intersection(c.nodes)), default=None)
    partials = []
    chunks = []
    for community in communities:
        summary = community.summary
        if community is focus and evidence:
            summary += f"\n\nПроверенные данные графа для вопроса:\n{evidence}"
        reply = _ask(
            llm,
            "Ответь на вопрос по одной сводке. Сохрани важные факты и числа. Оцени полезность от 0 до 10, "
            "где 0 означает, что данных для ответа нет. Запиши частичный ответ в answer, а оценку в score.",
            question,
            summary,
            json_schema=MAP_SCHEMA,
        )
        try:
            data = json.loads(reply)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        score, text = data.get("score"), data.get("answer")
        if type(score) is not int or not 0 < score <= 10 or not isinstance(text, str) or not text.strip():
            continue

        partials.append((score, text.strip()))
        subgraph = kb.graph.subgraph(community.nodes)
        for _, attrs in subgraph.nodes(data=True):
            chunks.extend(attrs.get("sources", []))
        for _, _, attrs in subgraph.edges(data=True):
            chunks.extend(attrs.get("sources", []))

    if not partials:
        return Answer(_NOT_FOUND, refused=True)
    context = "\n".join(f"- {text}" for _, text in sorted(partials, key=lambda item: -item[0]))
    system = (
        "Объедини частичные ответы, относящиеся к вопросу. Используй только их факты и числа. "
        "Учитывай область каждого частичного ответа и не считай повторяющиеся факты дважды. "
        f"Если итоговый ответ из них получить нельзя, напиши: {_NOT_FOUND}"
    )
    if evidence:
        system = (
            "Ответь на вопрос по проверенным фактам графа и частичным ответам. "
            "При расхождениях опирайся на факты графа. В обзорных вопросах используй "
            "готовые суммы весов по нужной линейке, не пересчитывай их по частичным ответам. "
            "REPLACED_BY — связь от старой детали к актуальной; для замены пройди цепочку до конца. "
            f"Если данных о нужном факте нет, ответь: {_NOT_FOUND}"
        )
        if lines and lines == anchors:
            system += (
                " Если в готовой таблице есть запрошенный показатель, назови указанный в ней максимум и его значение. "
                "Отсутствие данных в отдельных сводках не опровергает таблицу по всей линейке. Отвечай кратко."
            )
        context = f"Проверенные факты графа:\n{evidence}\n\nЧастичные ответы:\n{context}"
        chunks.extend(fact_sources)
    text = _ask(
        llm,
        system,
        question,
        context,
    )
    return _answer(text, _documents(chunks))


def _ask(llm: LLMClient, system: str, question: str, context: str, *, json_schema: dict | None = None) -> str:
    reply = llm.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": f"{context}\n\nВопрос: {question}"}],
        json_schema=json_schema,
    )
    return reply.content.strip()


def _answer(text: str, sources: list[str], facts: list[str] | None = None) -> Answer:
    if not text or _NOT_FOUND[:-1].casefold() in text.casefold():
        return Answer(_NOT_FOUND, refused=True)
    return Answer(text, sources=sources, facts=facts or [])


def _documents(chunks: Iterable[str]) -> list[str]:
    return sorted({chunk.split("#", 1)[0] for chunk in chunks})
