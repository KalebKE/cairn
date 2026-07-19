# Entity Registry

## Overview

The Cairn entity registry manages corporate entities with types, jurisdictions, relationships, and hierarchies. Entities serve as scoping boundaries --- you can attach memories, encrypted profiles, and knowledge base documents to a specific entity, keeping different organizations' data cleanly separated.

Install: `pip install cairn[entity]` (includes encryption support for entity-scoped profiles).

Entities are soft-deleted (status set to `dissolved`) rather than hard-deleted, preserving historical data and relationships.

## Quick Example

```
# Create a holding company and subsidiary
cairn_entity_create(entity_id="holdco", name="Holding Company Inc", entity_type="c_corp", jurisdiction="US-DE")
cairn_entity_create(entity_id="acme", name="Acme LLC", entity_type="llc", jurisdiction="US-WY")

# Define the relationship
cairn_entity_add_relationship(source_entity_id="holdco", target_entity_id="acme", relationship_type="parent_of",
    metadata={"ownership_pct": 100, "year": 2024})

# View the hierarchy
cairn_entity_tree(entity_id="holdco")

# Store entity-scoped data
cairn_store(content="Acme uses a hosted payment provider", event_type="decision", entity_id="acme")
```

## Entity Types

| Type | Description |
|------|-------------|
| `company` | Generic company |
| `llc` | Limited liability company |
| `s_corp` | S corporation |
| `c_corp` | C corporation |
| `foundation` | Foundation |
| `startup` | Startup |
| `trust` | Trust |
| `partnership` | Partnership |
| `sole_proprietorship` | Sole proprietorship |
| `nonprofit` | Nonprofit organization |
| `other` | Other entity type |

## Relationship Types

| Type | Direction | Example |
|------|-----------|---------|
| `parent_of` | Source is parent of target | HoldCo `parent_of` Acme |
| `subsidiary_of` | Source is subsidiary of target | Acme `subsidiary_of` HoldCo |
| `owned_by` | Source is owned by target | Acme `owned_by` Founder |
| `acquired_by` | Source was acquired by target | StartupX `acquired_by` BigCorp |
| `partner_of` | Source partners with target | Acme `partner_of` PartnerCo |
| `investor_in` | Source invests in target | VCFund `investor_in` Acme |
| `operated_by` | Source is operated by target | Brand `operated_by` Acme |

All relationships are directed (source to target). To express a bidirectional relationship, create two entries.

## Tools Reference

| Tool | Purpose |
|------|---------|
| `cairn_entity_create` | Create an entity with ID (slug), display name, type, jurisdiction, and optional metadata |
| `cairn_entity_get` | Get detailed information about a specific entity by ID |
| `cairn_entity_list` | List all entities with optional type and status filters |
| `cairn_entity_update` | Update an entity's name, status, jurisdiction, or metadata (only provided fields change) |
| `cairn_entity_delete` | Soft-delete an entity (sets status to `dissolved`). Does not remove data. |
| `cairn_entity_add_relationship` | Add a directed relationship between two entities with optional metadata |
| `cairn_entity_relationships` | Get all relationships for an entity. Filter by direction (`outgoing`/`incoming`) and type. |
| `cairn_entity_tree` | Recursive hierarchy view from a root entity (follows `parent_of` relationships, with cycle detection) |

## Common Workflows

### Create and Organize Entities

```
# Create entities
cairn_entity_create(entity_id="holdco", name="Holding Company Inc", entity_type="c_corp", jurisdiction="US-DE",
    metadata={"ein": "12-3456789", "founded": "2020"})

cairn_entity_create(entity_id="acme-llc", name="Acme Operations LLC", entity_type="llc", jurisdiction="US-WY",
    metadata={"ein": "98-7654321", "founded": "2022"})

cairn_entity_create(entity_id="acme-brand", name="Acme Brand Co", entity_type="llc", jurisdiction="US-DE")

# Define relationships
cairn_entity_add_relationship(source_entity_id="holdco", target_entity_id="acme-llc", relationship_type="parent_of",
    metadata={"ownership_pct": 100})
cairn_entity_add_relationship(source_entity_id="holdco", target_entity_id="acme-brand", relationship_type="parent_of",
    metadata={"ownership_pct": 100})

# View the tree
cairn_entity_tree(entity_id="holdco")
# holdco (Holding Company Inc)
#   acme-llc (Acme Operations LLC)
#   acme-brand (Acme Brand Co)
```

### Query Relationships

```
# All relationships for an entity
cairn_entity_relationships(entity_id="acme-llc")

# Only incoming relationships
cairn_entity_relationships(entity_id="acme-llc", direction="incoming")

# Only parent_of relationships
cairn_entity_relationships(entity_id="holdco", relationship_type="parent_of", direction="outgoing")
```

### Update and Manage Entities

```
# Update metadata
cairn_entity_update(entity_id="acme-llc", metadata={"registered_agent": "Northwest"})

# Change status
cairn_entity_update(entity_id="acme-brand", status="dormant")

# Soft-delete (dissolve)
cairn_entity_delete(entity_id="acme-brand")
# Status is now "dissolved" --- data and relationships are preserved
```

### Entity-Scoped Data

Memories, profiles, and documents can all be scoped to an entity:

**Memories**:
```
cairn_store(content="Acme uses us-east-1 for all AWS services", event_type="decision", entity_id="acme-llc")
cairn_query(query="AWS region", entity_id="acme-llc")
```

**Encrypted profiles**:
```
cairn_profile_set(category="financial", field_name="ein", value="98-7654321", entity_id="acme-llc")
cairn_profile_get(category="financial", entity_id="acme-llc")
cairn_profile_search(query="ein", entity_id="acme-llc")
```

**Knowledge base documents**:
```
cairn_ingest_document(path_or_url="/docs/acme-operating-agreement.pdf", entity_id="acme-llc")
cairn_search_documents(query="operating agreement terms", entity_id="acme-llc")
```

### List and Filter Entities

```
# All entities
cairn_entity_list()

# Only LLCs
cairn_entity_list(entity_type="llc")

# Only active entities
cairn_entity_list(status="active")
```

## Tips

- **Entity IDs are slugs.** Use lowercase with hyphens (e.g., `acme-llc`, `holding-co`). These are permanent identifiers.
- **Soft-delete preserves history.** `cairn_entity_delete` sets status to `dissolved` but keeps all data. This is intentional --- corporate records should not vanish.
- **Use metadata freely.** The `metadata` field on entities and relationships is a flexible JSON object. Store EINs, founding dates, ownership percentages, registered agents, or anything else you need.
- **Relationships are directed.** `parent_of` goes from parent to child. If you need to query "who is my parent?", filter by `direction="incoming"` and `relationship_type="parent_of"`.
- **Tree traversal has cycle detection.** `cairn_entity_tree` follows `parent_of` edges recursively but detects and breaks cycles, so circular relationships will not cause infinite loops.
- **Entity scoping is optional.** You do not have to use entities. Memories, profiles, and documents without an `entity_id` are unscoped and globally accessible. Add entity scoping only when you need organizational separation.
- **Set null to delete metadata keys.** In `cairn_entity_update`, setting a metadata key to `null` removes it: `metadata={"old_key": null}`.
