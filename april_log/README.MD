# Discord → Grafana Loki Logging

This project logs Discord events (messages, edits, deletes, reactions) to a [Grafana Loki](https://grafana.com/oss/loki/) backend using structured, queryable JSON. This README documents the API data structure and gives quick-start instructions for both sending and querying data.

---

## 📡 Loki Ingestion Endpoint

**HTTP POST**  
`<LOKI_URL>/loki/api/v1/push`

**Required Headers:**
- `Content-Type: application/json`
- `Authorization: Bearer <TOKEN>` _(optional, only if Loki auth is enabled)_

**Payload Format:**
```json
{
  "streams": [
    {
      "stream": { "<label_key>": "<label_value>", ... },
      "values": [
        [ "<timestamp_ns>", "<json_log_line>" ]
      ]
    }
  ]
}
```
stream: Loki labels, used for querying (see below)

values: Each is [timestamp_ns, json-string-of-event]


🏷️ Loki Stream Labels
Every log includes at least:

app: always "discord-bot"

event_type: "message", "edit", "delete", or "reaction"

channel_id: Discord channel ID (string)

guild_id: Discord guild/server ID (string) or "DM" for private messages

Example:
```
"stream": {
  "app": "discord-bot",
  "event_type": "message",
  "channel_id": "123456789012345678",
  "guild_id": "987654321098765432"
}
```
📦 Event Data Schemas
All values entries are JSON strings.
See below for each event’s data structure.
```
New Message (event_type=message)
{
  "content": "Text of the message",
  "message_id": "123...",
  "author": {
    "id": "456...",
    "name": "Username",
    "discriminator": "1234",
    "bot": false,
    "avatar": "https://..."  // null if not set
  },
  "channel": {
    "id": "789...",
    "name": "general",
    "category": "Category Name"  // null if unavailable
  },
  "guild": {
    "id": "654...",
    "name": "Guild Name"
  }, // null for DMs
  "created_at": "2024-01-01T12:34:56.789123+00:00",
  "attachments": [
    {
      "url": "https://...",
      "filename": "file.png",
      "size": 123456,
      "content_type": "image/png"
    }
  ],
  "embeds": [
    {
      "type": "rich",
      "title": "Title",
      "description": "Embed description"
    }
  ]
}
```
{
  "content": "Text of the message",
  "message_id": "123...",
  "author": {
    "id": "456...",
    "name": "Username",
    "discriminator": "1234",
    "bot": false,
    "avatar": "https://..."  // null if not set
  },
  "channel": {
    "id": "789...",
    "name": "general",
    "category": "Category Name"  // null if unavailable
  },
  "guild": {
    "id": "654...",
    "name": "Guild Name"
  }, // null for DMs
  "created_at": "2024-01-01T12:34:56.789123+00:00",
  "attachments": [
    {
      "url": "https://...",
      "filename": "file.png",
      "size": 123456,
      "content_type": "image/png"
    }
  ],
  "embeds": [
    {
      "type": "rich",
      "title": "Title",
      "description": "Embed description"
    }
  ]
}
Message Edit (event_type=edit)
json
Copy
Edit
{
  "event_type": "edit",
  "message_id": "123...",
  "channel_id": "789...",
  "guild_id": "654...",
  "author_id": "456...",
  "old_content": "Previous content",
  "new_content": "Edited content",
  "edited_at": "2024-01-01T12:40:00.000000+00:00"
}
Message Delete (event_type=delete)
json
Copy
Edit
{
  "event_type": "delete",
  "message_id": "123...",
  "channel_id": "789...",
  "guild_id": "654...",
  "author_id": "456...",
  "content": "Deleted message content",
  "created_at": "2024-01-01T12:34:56.789123+00:00",
  "deleted_at": "2024-01-01T12:50:00.000000+00:00"
}
Reaction Add/Remove (event_type=reaction)
json
Copy
Edit
{
  "event_type": "reaction_add", // or "reaction_remove"
  "message_id": "123...",
  "channel_id": "789...",
  "guild_id": "654...",
  "user": {
    "id": "456...",
    "name": "Username"
  },
  "emoji": {
    "name": "smile",
    "id": "789012345678901234", // null for unicode emoji
    "custom": true // true if custom, false if unicode
  }
}
🔎 Querying With Loki (LogQL)
Find all messages in a channel:

arduino
Copy
Edit
{app="discord-bot", event_type="message", channel_id="123..."}
Find all edits in a guild:

arduino
Copy
Edit
{app="discord-bot", event_type="edit", guild_id="654..."}
Search for content or users (with parsed JSON):

python
Copy
Edit
{app="discord-bot", event_type="message"} | json | author.id = "456..."
Search message content for a string (case-insensitive):

pgsql
Copy
Edit
{app="discord-bot", event_type="message"} | json | content=~"(?i)keyword"
(Adjust LogQL for your Loki version and requirements.)

⚠️ Edge Cases & Notes
DMs: guild_id label is "DM", and the guild field in the log value is null.

Timestamps: Always nanoseconds since Unix epoch, as string (Loki requirement).

Deleted messages: Only metadata available; attachment URLs may expire.

Auth: Include bearer token if required.

🛠️ Integration Notes
To produce compatible logs:

Send POSTs to /loki/api/v1/push using above label/value structures.

To consume logs:

Query using LogQL and parse log lines as JSON (| json).

Extending:

Add new event types or labels as needed, following these conventions.

Example Payload
```json
Copy
Edit
{
  "streams": [
    {
      "stream": {
        "app": "discord-bot",
        "event_type": "message",
        "channel_id": "112233445566",
        "guild_id": "998877665544"
      },
      "values": [
        [
          "1719068573895120000",
          "{\"content\": \"Hello world!\", \"message_id\": \"777888999\", ... }"
        ]
      ]
    }
  ]
}
```
License
MIT (or your project’s license of choice).

