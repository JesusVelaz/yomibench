extends Node
# YomiBench "Multiplayer (vs bot)" (docs/HUMAN_PLAY.md, "In the game's menu"). Loaded by the game's own
# mod loader from <game>/mods/YomiBenchVsBot.zip or a Workshop item. It only adds a main-menu entry
# (VsBotUILayer.gd); everything else happens after the player picks it. client_side: it never enters
# the online version hash, and it never acts in a match it didn't start.


func _init(modLoader = ModLoader):
	modLoader.installScriptExtension("res://YomiBenchVsBot/VsBotUILayer.gd")
