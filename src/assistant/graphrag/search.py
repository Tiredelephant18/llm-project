"""Локальный поиск по графу и обзорный поиск по сводкам."""

import json
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
    graph = kb.graph
    anchors = {f"{code_type(code)}:{code}" for code in find_codes(question)}
    if anchors.difference(graph):
        return Answer(_NOT_FOUND, refused=True)
    anchors.update(f"{kind}:{name}" for kind, name in find_names(question, kb.vocabulary))
    anchors.intersection_update(graph.nodes)
    if not anchors:
        return Answer(_NOT_FOUND, refused=True)

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
        supplier = any(stem in question.casefold() for stem in ("поставщик", "поставля", "поставк"))
        _add_replacements(graph, edges, supplier)

    facts, sources = _local_context(graph, edges)
    if not facts:
        return Answer(_NOT_FOUND, refused=True)
    context = "\n".join(f"{i}. {fact}" for i, fact in enumerate(facts, 1))
    text = _ask(llm, _LOCAL_PROMPT, question, context)
    if set(find_codes(text)) - set(find_codes(f"{question}\n{context}")):
        return Answer(_NOT_FOUND, refused=True)
    return _answer(text, sources, facts)


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


def global_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Получает частичные ответы по сводкам и объединяет полезные."""
    partials = []
    chunks = []
    for community in kb.communities:
        if not community.summary.strip():
            continue
        reply = _ask(
            llm,
            "Ответь на вопрос по одной сводке. Сохрани важные факты и числа. Оцени полезность от 0 до 10, "
            "где 0 означает, что данных для ответа нет. Запиши частичный ответ в answer, а оценку в score.",
            question,
            community.summary,
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
    text = _ask(
        llm,
        "Объедини частичные ответы, относящиеся к вопросу. Используй только их факты и числа. "
        "Учитывай область каждого частичного ответа и не считай повторяющиеся факты дважды. "
        f"Если итоговый ответ из них получить нельзя, напиши: {_NOT_FOUND}",
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
