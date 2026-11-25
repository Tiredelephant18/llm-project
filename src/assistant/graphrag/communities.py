"""Сообщества графа и их сводки. Требование FR-3."""

from collections import Counter

import networkx as nx

from assistant.infra.llm import LLMClient
from assistant.infra.schemas import Community


def detect_communities(graph: nx.DiGraph) -> list[Community]:
    """Разбивает узлы на сообщества алгоритмом Лувена с фиксированным `seed`.

    Каждый узел входит ровно в одно сообщество. Поле `summary` здесь остаётся пустым.
    """
    undirected = nx.Graph()
    undirected.add_nodes_from(sorted(graph.nodes))
    for source, target, weight in sorted(graph.edges(data="weight", default=1)):
        # Встречные рёбра дают общий вес; исходный граф сохраняет оба направления.
        previous = undirected.get_edge_data(source, target, default={})
        undirected.add_edge(source, target, weight=previous.get("weight", 0) + weight)

    if undirected.size(weight="weight") == 0:
        groups = [{node} for node in undirected]
    else:
        groups = nx.community.louvain_communities(undirected, weight="weight", seed=42)

    return [Community(index, nodes) for index, nodes in enumerate(sorted(sorted(group) for group in groups))]


def summarize_community(community: Community, graph: nx.DiGraph, llm: LLMClient) -> str:
    """Сводка сообщества не длиннее 150 слов: какие сущности входят, как связаны, что встречается чаще.

    Все числа считает код по весам рёбер и передаёт модели готовыми. Модель их только формулирует.
    """
    subgraph = graph.subgraph(community.nodes)
    entities = [f"{attrs['name']} ({attrs['type']})" for _, attrs in sorted(subgraph.nodes(data=True))]
    facts = []
    relation_weights = Counter()
    failure_weights = Counter()
    for source, target, attrs in sorted(subgraph.edges(data=True)):
        source_name = subgraph.nodes[source]["name"]
        target_name = subgraph.nodes[target]["name"]
        relation_type = attrs["type"]
        weight = attrs.get("weight", 1)
        facts.append(f"{source_name} -{relation_type}-> {target_name} (вес {weight})")
        relation_weights[relation_type] += weight
        if relation_type == "FAILED":
            failure_weights[target_name] += weight

    relation_totals = [f"{kind}: {weight}" for kind, weight in relation_weights.most_common()]
    failure_totals = [f"{name}: {weight}" for name, weight in failure_weights.most_common()]
    sections = {
        "Сущности": entities,
        "Связи внутри сообщества": facts,
        "Суммы весов по типам связей": relation_totals,
        "Суммы весов связей FAILED по узлам отказа в сообществе": failure_totals,
    }
    context = "\n\n".join(f"{title}:\n" + ("\n".join(rows) or "Нет данных") for title, rows in sections.items())
    response = llm.chat(
        [
            {
                "role": "system",
                "content": (
                    "Напиши по-русски сводку сообщества графа знаний объёмом до 150 слов. "
                    "Назови основные сущности, их связи и наиболее частые отказы, если есть данные. "
                    "Сохраняй названия линеек, моделей и коды деталей. "
                    "Используй только приведённые факты и готовые подсчёты. "
                    "Вес связи — число чанков, где она найдена. Сумма весов не равна числу уникальных "
                    "чанков или реальных отказов."
                ),
            },
            {"role": "user", "content": context},
        ]
    )
    return " ".join(response.content.split()[:150])
