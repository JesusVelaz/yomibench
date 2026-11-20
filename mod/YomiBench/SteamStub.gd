extends Node
# Stands in for the game's SteamHustle autoload (override.cfg replaces it). The real one
# calls Steam init, which crashes without a Steam app id; the server needs none of it.

signal started

var IS_ONLINE = false
var IS_OWNED = false
var STEAM_ID = 0
var STEAM_NAME = ""
var APP_ID = 0
var STARTED = false
var WORKSHOP_ENABLED = false

func record_winner(_w): pass
func incr_stat(_s): pass
func set_stat(_s, _v): pass
func unlock_achievement(_a): pass
func has_supporter_pack(_id): return false
func print_all_achievements(): pass
