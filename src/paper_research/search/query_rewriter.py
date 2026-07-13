import re


class QueryRewriter:
    def rewrite(self, query: str) -> list[str]:
        normalized = " ".join(re.sub(r"[^\w\s-]", " ", query).split())
        without_hyphens = normalized.replace("-", " ")
        terms = [term for term in without_hyphens.split() if len(term) > 2]
        compact = " ".join(terms[:8])
        variants = [query.strip(), without_hyphens, compact]
        return list(dict.fromkeys(variant for variant in variants if variant))
