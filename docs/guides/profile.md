# Secure Profile

## Overview

The Cairn secure profile stores sensitive personal data with AES-256 encryption using your macOS Keychain for key management. Data is encrypted before it touches SQLite and decrypted only on-device when you request it. Zero cloud exposure by default.

Install: `pip install cairn[encrypt]`

Profiles support 7 categories and can be scoped to entities (e.g., store credentials for different organizations separately).

## Quick Example

```
# Store sensitive data
cairn_profile_set(category="identity", field_name="passport_number", value="X12345678",
    metadata={"issuing_country": "US", "expires": "2032-06"})

# Retrieve and decrypt
cairn_profile_get(category="identity", field_name="passport_number")
# Returns: "X12345678" (decrypted on-device)

# See what's stored without decrypting
cairn_profile_list()
# Returns: identity (3 fields), financial (2 fields), contacts (5 fields)

# Search metadata (not encrypted values)
cairn_profile_search(query="passport")
```

## Categories

| Category | Example Fields |
|----------|---------------|
| `identity` | Full name, passport number, date of birth, national ID |
| `medical` | Blood type, allergies, medications, emergency contact |
| `financial` | Bank account numbers, tax IDs, crypto wallet addresses |
| `personal` | Home address, phone numbers, personal email |
| `professional` | Work email, employee ID, office address, credentials |
| `contacts` | Key contacts with phone/email/relationship |
| `legal` | Attorney info, power of attorney, will location |

## Tools Reference

| Tool | Purpose |
|------|---------|
| `cairn_profile_set` | Store an encrypted field in a category. Metadata (unencrypted) can be attached for search. |
| `cairn_profile_get` | Decrypt and retrieve fields. Omit `field_name` to get all fields in a category. |
| `cairn_profile_search` | Search metadata and field names (not encrypted values). Use to find fields, then `get` to decrypt. |
| `cairn_profile_list` | List all categories with field counts. Shows what is stored without decrypting anything. |

## Common Workflows

### Store Sensitive Data

```
cairn_profile_set(
    category="financial",
    field_name="bank_routing",
    value="021000021",
    metadata={"bank": "Chase", "account_type": "checking"}
)
```

With entity scoping:
```
cairn_profile_set(
    category="financial",
    field_name="ein",
    value="12-3456789",
    entity_id="acme",
    metadata={"type": "federal_ein"}
)
```

### Retrieve Everything in a Category

```
cairn_profile_get(category="identity")
# Returns all identity fields, decrypted
```

### Search Before Retrieving

```
cairn_profile_search(query="bank")
# Returns: financial/bank_routing (metadata: bank=Chase)

cairn_profile_get(category="financial", field_name="bank_routing")
# Returns: "021000021"
```

### Entity-Scoped Profiles

Keep different organizations' data separate:

```
# Personal profile
cairn_profile_set(category="identity", field_name="ssn", value="123-45-6789")

# Company profile
cairn_profile_set(category="financial", field_name="ein", value="98-7654321", entity_id="acme")
cairn_profile_set(category="professional", field_name="registered_agent", value="Northwest", entity_id="acme")

# Retrieve only Acme's data
cairn_profile_get(category="financial", entity_id="acme")
```

### Inventory Check

```
cairn_profile_list()
# identity: 4 fields
# financial: 2 fields
# contacts: 7 fields

cairn_profile_list(entity_id="acme")
# financial: 1 field
# professional: 1 field
```

## Security Model

- **Encryption**: AES-256-GCM with a unique key per Cairn installation
- **Key storage**: macOS Keychain (hardware-backed on Apple Silicon)
- **At rest**: Values are encrypted in SQLite. Only metadata and field names are searchable.
- **In transit**: Data never leaves your machine unless you explicitly enable cloud sync
- **Cloud sync**: Optional. If enabled, encrypted values are synced as-is (still encrypted). Decryption only happens on devices with the Keychain key.

## Tips

- **Metadata is not encrypted.** Use it for search-friendly labels (e.g., `{"bank": "Chase"}`) but do not put sensitive data in metadata. The actual value goes in `value`.
- **Use `profile_list` before `profile_get`.** It shows you what categories and field counts exist without decrypting anything, so you know what to request.
- **Entity scoping is optional.** Personal data (your SSN, passport) does not need an entity ID. Use entity scoping only for organizational data.
- **Fields are upserted.** Calling `cairn_profile_set` with the same category + field_name overwrites the previous value. No versioning.
- **Search is metadata-only.** `cairn_profile_search` searches field names and metadata, never encrypted values. This is by design.
