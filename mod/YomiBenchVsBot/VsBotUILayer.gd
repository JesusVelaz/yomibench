extends "res://ui/UILayer.gd"
# The main menu gets "Multiplayer (vs bot)" after the game's multiplayer entries, the way other mods add theirs
# (a Button in their container). Pressing it starts a local multiplayer session against a YomiBench bot (VsBot.gd):
# the bot server and a headless copy of this game on this PC, joined over 127.0.0.1 only. It is refused while
# an online session or a Steam lobby is open. "Back to main menu" stops both and reloads the menu.

const VsBot = preload("res://YomiBenchVsBot/VsBot.gd")
const AFTER = ["MultiplayerButton", "DirectConnectButton", "SteamMultiplayerButton"]


func _ready():
	var anchor = null
	for n in AFTER:  # the last of the game's multiplayer entries in the menu
		var b = get_node_or_null("%" + n)
		if b != null and (anchor == null or (b.get_parent() == anchor.get_parent() and b.get_index() > anchor.get_index())):
			anchor = b
	if anchor == null:
		return
	var b = Button.new()
	b.name = "VsBotButton"
	b.text = "Multiplayer (vs bot)"
	b.flat = anchor.flat
	b.mouse_default_cursor_shape = anchor.mouse_default_cursor_shape
	b.rect_min_size = anchor.rect_min_size
	anchor.get_parent().add_child_below_node(anchor, b)
	b.connect("pressed", self, "_on_vsbot_pressed")  # the game wires menu sounds to every button itself
	if VsBot.autoplay_schedule() != "" and VsBot.instance(get_tree()) == null:
		call_deferred("_autoplay")  # headless test hook (--yomibench-vsbot-autoplay=...)


func _autoplay():
	while not Global.mods_loaded:
		yield(get_tree(), "idle_frame")
	_on_vsbot_pressed()


func _on_vsbot_pressed():
	# Not while the game still loads its mods in the background (the character select's thread).
	if VsBot.instance(get_tree()) != null or not Global.mods_loaded:
		return
	Global.frame_advance = false
	var v = VsBot.new()
	get_tree().root.add_child(v)  # outlives the menu scene's reloads
	if v.launch():  # refuses (and says why) while an online session or a Steam lobby is open
		hide_main_menu()
