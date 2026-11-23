"""Сборка графа знаний из результатов извлечения. Требование FR-2."""

import networkx as nx

from assistant.infra.schemas import Entity, ExtractionResult


def _add_entity(graph: nx.DiGraph, entity: Entity, chunk_id: str) -> None:
    node_id = entity.node_id
    if node_id not in graph:
        graph.add_node(node_id, type=entity.type, name=entity.name, sources=set())
    graph.nodes[node_id]["sources"].add(chunk_id)


def build_graph(results: list[ExtractionResult]) -> nx.DiGraph:
    """Объединяет результаты по всем чанкам в один направленный граф.

    Узел: идентификатор `Entity.node_id`, атрибуты `type`, `name`, `sources` (список chunk_id).
    Ребро: атрибуты `type`, `weight` (число разных чанков, где связь найдена), `sources`.
    Связи, не соответствующие схеме (`Relation.matches_schema`), отбрасываются. Если между двумя
    деталями есть `REPLACED_BY` в обе стороны, остаётся связь с большим весом, при равенстве
    отбрасываются обе. Число таких конфликтов записывается в `graph.graph["conflicts"]`.
    """
    graph = nx.DiGraph(conflicts=0)

    for result in results:
        for entity in result.entities:
            _add_entity(graph, entity, result.chunk_id)

        for relation in result.relations:
            if not relation.matches_schema():
                continue

            _add_entity(graph, relation.subject, result.chunk_id)
            _add_entity(graph, relation.object, result.chunk_id)
            source = relation.subject.node_id
            target = relation.object.node_id
            if not graph.has_edge(source, target):
                graph.add_edge(source, target, type=relation.type, sources=set())
            graph.edges[source, target]["sources"].add(result.chunk_id)

    for _, attrs in graph.nodes(data=True):
        attrs["sources"] = sorted(attrs["sources"])

    for _, _, attrs in graph.edges(data=True):
        attrs["weight"] = len(attrs["sources"])
        attrs["sources"] = sorted(attrs["sources"])

    edges_to_remove = []
    for source, target, attrs in graph.edges(data=True):
        # Проверяем каждую встречную пару один раз.
        if source >= target or attrs["type"] != "REPLACED_BY":
            continue
        reverse = graph.get_edge_data(target, source)
        if reverse is None or reverse["type"] != "REPLACED_BY":
            continue

        graph.graph["conflicts"] += 1
        if attrs["weight"] <= reverse["weight"]:
            edges_to_remove.append((source, target))
        if reverse["weight"] <= attrs["weight"]:
            edges_to_remove.append((target, source))

    graph.remove_edges_from(edges_to_remove)
    return graph
