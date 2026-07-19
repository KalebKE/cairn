#!/bin/bash
# Cairn -- Set up conversational instructions for Claude Desktop
# Copies instructions to clipboard, then guides the user to paste them.

INSTRUCTIONS='You have Cairn persistent memory tools available. Use them automatically:

- At the start of every conversation, call cairn_welcome() to check for context
- When the user shares preferences, decisions, or important information, call cairn_store() to save it
- When the user asks about past conversations or context, call cairn_query() to search memory
- When ending a conversation with important context, call cairn_checkpoint() to save state

Be conversational about memory. You do not need to be asked to remember things.
If something seems worth remembering, store it. If context might help, query for it.'

echo "$INSTRUCTIONS" | pbcopy

echo ""
echo "=== Cairn Instructions Copied to Clipboard ==="
echo ""
echo "Now paste them into Claude Desktop:"
echo ""
echo "  1. Open Claude Desktop"
echo "  2. Click your profile icon (bottom-left corner)"
echo "  3. Click 'Settings'"
echo "  4. Under 'Profile', find 'Custom Instructions'"
echo "  5. Paste (Cmd+V) into the text box"
echo "  6. Click 'Save'"
echo ""
echo "Done! Claude will now use Cairn memory automatically."
echo ""
