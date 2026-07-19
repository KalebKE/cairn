# Cairn Demo — Terminal Recording
# Install VHS: brew install charmbracelet/tap/vhs
# Run: vhs docs/demo.tape

Output docs/demo.gif

Set FontSize 16
Set Width 800
Set Height 400
Set Theme "Catppuccin Mocha"
Set Padding 20

Type "# Start a new Claude Code session..."
Sleep 500ms
Enter
Sleep 1s

Type 'claude "Remember: always use early returns, never nest more than 2 levels"'
Sleep 500ms
Enter
Sleep 2s

Type "# Cairn auto-captures this as a user preference"
Sleep 500ms
Enter
Sleep 1s

Type "# Later, in a completely new session..."
Sleep 500ms
Enter
Sleep 1s

Type 'claude "What are my code style preferences?"'
Sleep 500ms
Enter
Sleep 2s

Type "# Cairn surfaces the memory instantly:"
Sleep 500ms
Enter
Type '# "Use early returns, never nest more than 2 levels"'
Sleep 500ms
Enter
Sleep 1s

Type "cairn stats --card"
Sleep 500ms
Enter
Sleep 3s
