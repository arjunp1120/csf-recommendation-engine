"""Product Resolver — canonical instrument_name -> product mapping (plan §7.2, §0.9).

The mapping table itself is owned by ``infra/db/instrument_products.py``;
this package wraps the rows in a fast in-memory index (``ProductResolver``)
used everywhere a callsite asks "what product is this instrument?".
"""

from csf_recommendation_engine.domain.products.resolver import (
    ProductMatch,
    ProductResolver,
)

__all__ = ["ProductMatch", "ProductResolver"]
