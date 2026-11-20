extends Node
# YomiBench human-vs-bot play (docs/HUMAN_PLAY.md): a person plays P1 in the game's own UI, a
# yomibench bot plays P2. Loaded as an autoload through override.cfg (tools/human_play.py builds the
# run dir) and active only when the game is started with --yomi-human=PORT; otherwise it does
# nothing. It never touches YomiServer or the engine protocol: it is a TCP client of the bot server
# (src/yomibench/human_play.py) on 127.0.0.1:PORT, one JSON line per message, replies in order.
#
# - Matches are started here, not from the character select: the standard online settings (as
#   YomiServer._standard_settings) with the panel's settings preset applied (the presets come from
#   the server's hello: standard, ranked), the characters picked on this script's panel, a random
#   seed. The match runs through Main._on_match_ready like a singleplayer match from the character
#   select, and match_start reports exactly the settings it was given.
# - At every decision (both turn branches opened, game paused) the mod sends the inputs both
#   fighters were given last turn and a compact snapshot of the real state, and gets the bot's pick
#   for P2, which it submits with P2's own ActionButtons.on_action_submitted: the call the panel's
#   Select button makes, so the game treats it as a normal selection. P2's panel stays hidden and
#   takes no input, and both Undo buttons are hidden (an undo would rewind the real match only), as
#   are the in-match replay keys (see _input).
# - Inputs are read from each fighter right after on_action_selected (its queued action, data and
#   extra, as the replay records them), so P1's are exactly what the UI sent.
# - At the end it sends the final snapshot and the winner; the server writes the result.
# - "Bot thinks until you lock in" (panel; default off): when P1 locks in (P1's action_selected) while
#   the bot's pick is still pending, the mod sends a one-way {"type": "locked_in", "turn": N} (no
#   reply, no callback queued): THAT P1 locked in, never what. P1's input still reaches the server
#   only with the next decision. The bottom bar shows how deep the bot got (the reply's `depth`).
#
# Options: --yomi-human=PORT (required), --yomi-human-host=ADDR (default 127.0.0.1; tests in Docker),
# --yomi-human-p1=CHAR / --yomi-human-p2=CHAR (the panel's defaults), --yomi-human-autoplay (headless
# test: P1 plays random legal moves through its own action panel, games come from the server),
# --yomi-human-speed=K (physics ticks per second x K; autoplay only), --yomi-human-autoplay-wait=S
# (autoplay P1 waits S seconds before Select: a turn timer test).
#
# Sides. "single" (above): one game drives both fighters. "Multiplayer (vs bot)" (the YomiBenchVsBot mod)
# splits it over two copies of the game joined by the game's own direct-connect multiplayer on 127.0.0.1:
# - "host", the player's game: the panel, the match start and P1. It never talks to the bot server and
#   never learns the bot's pick before the game's own protocol delivers it (after both locked in).
# - "client", a headless copy (MpClient.gd): P2 and the bot server connection. It submits each pick with its
#   own P2 panel (Network.submit_action, then the game's send_action once both are ready), as a player's
#   game does, and relays to the host only what the host's panel shows (the hello, refusals, last turn's
#   estimate, results). P1's lock-in reaches it as the game's own "turn ready" message (no move in it).
# `link` is the node that carries messages between the two (VsBot.gd / MpClient.gd).

const PROTOCOL = 1
const VERSION = "6"  # 2: characters from the server's hello (all five), pick style, win estimate; 3: settings presets; 4: think until lock-in; 5: deadline in every mode; 6: multiplayer sides
# The characters the panel offers until the server's hello names the ones its sim plays.
var characters = ["Ninja", "Cowboy", "Wizard", "Mutant", "Robot"]

var port = 0
var host = "127.0.0.1"
var autoplay = false
var peer = null
var connected = false
var retry_at = 0
var inbuf = ""
var callbacks = []       # reply handlers, in request order
var server = {}          # the server's hello
var settings = {}

var main = null
var game = null
var p1_panel = null
var p2_panel = null
var phase = "boot"       # boot, connecting, idle, starting, playing, ended
var match_ = {}
var turn = 0
var decision_tick = -1000
var inputs_at = {}       # tick of the decision -> [P1 input, P2 input]
var pick = null          # the bot's pick for this decision, until submitted
var pick_pending = false
var desynced = false
var p2_fallback = false
var auto_tick = -1000
var games_played = 0
var rng = RandomNumberGenerator.new()

var layer = null
var status = null
var panel = null
var title = null
var info = null
var p1_opt = null
var p2_opt = null
var pick_opt = null
var settings_opt = null
var presets = {"standard": {}}  # the server's settings presets (name -> overrides of the standard settings)
var preset_names = ["standard"]
var estimate_box = null
var start_btn = null
var pick_styles = ["mixed", "strongest"]
var estimate = null      # the bot's win estimate from this decision's pick, shown once the turn is played
var shown_estimate = null
# Think until lock-in: the bot deepens its search until P1 locks in. The server only ever hears
# THAT P1 locked in (one-way `locked_in`, type and turn only), never what P1 picked.
var think_box = null
var think_on = false
var pending_turn = -1    # the decision whose pick is pending
var decision_msec = 0
var deadline_sent = false
var turn_ms = 0          # a per-turn ("default") timer's length; the server follows chess/increment banks itself
var depth_text = ""


signal leave_requested  # hosted: the player pressed "Back to main menu"
signal quitting         # hosted: autoplay is about to quit the game (the host stops its server first)

var hosted = false       # started by the YomiBenchVsBot menu mod (not by --yomi-human): adds "Back to main menu"
var side = "single"      # single, host or client (see the top)
var link = null          # host and client: the VsBot.gd / MpClient.gd node between the two games
var remote_status = {}   # host: what the client last reported (turn, desynced, last turn's estimate, depth)
var p1_locked_tick = -1000  # client: the decision tick P1 locked in at (the game's "turn ready", no move)
var p1_lock_msec = 0     # host: when P1 locked in this turn
var autoplay_wait = 0.0


func _ready():
	var p = _option("human", "")
	if p == "":
		set_process(false)
		set_process_input(false)
		return
	start(int(p), _option("human-host", host), "--yomi-human-autoplay" in OS.get_cmdline_args())


# Connect to the bot server on host:port and show the panel. --yomi-human=PORT calls this at start-up.
# Multiplayer (vs bot): VsBot.gd starts the host side (no server connection: port unused) and MpClient.gd
# the client side (the server connection, no panel), each with itself as `link_`.
func start(port_, host_ = "127.0.0.1", autoplay_ = false, hosted_ = false, side_ = "single", link_ = null):
	port = int(port_)
	host = host_
	autoplay = autoplay_
	hosted = hosted_
	side = side_
	link = link_
	autoplay_wait = float(_option("human-autoplay-wait", "0"))
	pause_mode = PAUSE_MODE_PROCESS
	rng.randomize()
	set_process(false)
	set_process_input(side != "client")
	call_deferred("_boot")


func _option(name, default):
	var flag = "--yomi-%s=" % name
	for a in OS.get_cmdline_args():
		if a.begins_with(flag):
			return a.substr(flag.length())
	return default


func _boot():
	# The game's own main scene stays (unlike YomiServer): wait for it and its character loading.
	while true:
		yield(get_tree(), "idle_frame")
		var s = get_tree().current_scene
		if s != null and s.has_method("_on_match_ready") and Global.mods_loaded:
			break
	settings = _standard_settings()
	var speed = float(_option("human-speed", "1"))
	if autoplay and speed > 1 and side == "single":
		Engine.iterations_per_second = int(60 * speed)
	if side != "client":
		_make_overlay()
	else:
		Network.connect("player_turn_ready", self, "_on_net_turn_ready")
	phase = "connecting"
	set_process(true)


# The same settings YomiServer plays (the settings panel's scene defaults, Sadness on as online
# play forces, no turn timer, no dummy): the settings the bots were trained under.
func _standard_settings():
	var scene = load("res://ui/GameSettingsPanelContainer.tscn")
	if scene == null:
		return {}
	var p = scene.instance()
	add_child(p)
	var s = {}
	if typeof(p.get("scene_defaults")) == TYPE_DICTIONARY:
		s = p.scene_defaults.duplicate(true)
	p.queue_free()
	if s.empty():
		return s
	s["sadness_enabled"] = true
	s["timer_mode"] = "none"
	s["p2_dummy"] = false
	return s


# --- connection --------------------------------------------------------------------------------

func _process(_delta):
	if side != "host":
		_net()
	elif phase == "playing" or phase == "ended":
		var post = main.get_node_or_null("%PostGameButtons") if is_instance_valid(main) else null
		if post and post.visible:  # the game's rematch: this panel starts the next match instead
			post.hide()
	match phase:
		"starting":
			_attach()
		"playing":
			_play_frame()
	_update_status()


func _net():
	if peer == null or (not connected and peer.get_status() in [StreamPeerTCP.STATUS_NONE, StreamPeerTCP.STATUS_ERROR]):
		if OS.get_ticks_msec() < retry_at:
			return
		retry_at = OS.get_ticks_msec() + 1000
		peer = StreamPeerTCP.new()
		peer.connect_to_host(host, port)
		return
	var st = peer.get_status()
	if st == StreamPeerTCP.STATUS_CONNECTED and not connected:
		connected = true
		peer.set_no_delay(true)
		_send({"type": "hello", "protocol": PROTOCOL, "mod": VERSION}, "_on_hello")
	elif st != StreamPeerTCP.STATUS_CONNECTED and connected:
		connected = false
		callbacks.clear()
		inbuf = ""
		if side == "client":
			link.bot_lost()  # the server stopped (the game quit): this copy quits too
			return
		_say("Lost the bot server. Restart tools/human_play.py." if not hosted else "Lost the bot server.")
		if phase == "playing" or phase == "starting":
			phase = "idle"
		return
	if not connected:
		return
	var n = peer.get_available_bytes()
	if n > 0:
		var r = peer.get_partial_data(n)
		if r[0] == OK:
			inbuf += r[1].get_string_from_utf8()
	var i = inbuf.find("\n")
	while i >= 0:
		var line = inbuf.substr(0, i)
		inbuf = inbuf.substr(i + 1)
		var res = JSON.parse(line)
		var cb = callbacks.pop_front() if not callbacks.empty() else ""
		var reply = res.result if res.error == OK and typeof(res.result) == TYPE_DICTIONARY else {"type": "error", "message": "bad reply"}
		if cb != "":
			call(cb, reply)
		i = inbuf.find("\n")


func _send(msg, cb = ""):
	if not connected:
		return false
	peer.put_data((JSON.print(msg) + "\n").to_utf8())
	callbacks.append(cb if cb != "" else "_on_ignored")
	return true


func _on_ignored(_reply):
	pass


# One-way message: the server sends no reply, so no callback is queued and replies stay in order.
func _notify(msg):
	if connected:
		peer.put_data((JSON.print(msg) + "\n").to_utf8())


func _on_hello(r):
	if r.get("type") != "hello":
		_say("The bot server refused: %s" % str(r.get("message")))
		return
	server = r
	if side == "client":  # the panel is the host's
		phase = "idle"
		link.to_host("hello", r)
		return
	phase = "idle"
	if typeof(r.get("characters")) == TYPE_ARRAY and not r.characters.empty():
		_fill_characters(r.characters)
	var opts = r.get("options", {})
	if typeof(r.get("settings_presets")) == TYPE_DICTIONARY and not r.settings_presets.empty():
		presets = r.settings_presets
		preset_names = presets.keys()
		settings_opt.clear()
		for n in preset_names:
			settings_opt.add_item(str(n))
	if typeof(opts) == TYPE_DICTIONARY:
		settings_opt.select(max(0, preset_names.find(str(opts.get("settings", "standard")))))
		pick_opt.select(max(0, pick_styles.find(str(opts.get("pick", "mixed")))))
		estimate_box.pressed = bool(opts.get("show_estimate", false))
		think_box.pressed = bool(opts.get("think_until_lockin", false))
	if autoplay:
		_next_or_quit(r.get("next"))
	else:
		_show_panel(_panel_title(), "")


func _panel_title():
	if side == "host":
		return "YomiBench multiplayer: you (P1) against %s (P2), over this PC only" % str(server.get("bot", "the bot"))
	return "YomiBench: play P1 against %s" % str(server.get("bot", "the bot"))


func _next_or_quit(nxt):
	if typeof(nxt) == TYPE_DICTIONARY:
		_start_match(str(nxt.p1), str(nxt.p2), int(nxt.seed))
	else:
		emit_signal("quitting")
		get_tree().quit()


# --- matches -----------------------------------------------------------------------------------

func _on_start_pressed():
	var ready = connected if side == "single" else not server.empty()
	if phase == "playing" or phase == "starting" or not ready:
		return
	randomize()
	_start_match(characters[p1_opt.selected], characters[p2_opt.selected], randi())


func _start_match(p1, p2, seed_):
	main = get_tree().current_scene
	if main == null or not main.has_method("_on_match_ready"):
		_say("The game's main scene isn't ready yet.")
		return
	if phase == "playing" and game != null:
		_send({"type": "abort", "reason": "a new match was started"})
	var options = {}
	var preset = str(server.get("options", {}).get("settings", "standard"))
	if not autoplay:  # the panel's choices; autoplay keeps the launcher's
		preset = str(preset_names[max(0, settings_opt.selected)])
		options = {"pick": pick_styles[max(0, pick_opt.selected)], "show_estimate": estimate_box.pressed,
			"settings": preset, "think_until_lockin": think_box.pressed}
	var think = think_box.pressed if not autoplay else bool(server.get("options", {}).get("think_until_lockin", false))
	var msg = {"p1": p1, "p2": p2, "seed": seed_, "settings": _with_preset(preset), "options": options, "think": think}
	if side == "host":
		# The client tells the server and starts the same match; both games then run it in lockstep.
		link.to_client("match", msg)
		remote_status = {}
		_launch(msg)
	else:
		start_bot_match(msg)


# Single and client: the server hears of the match, then the game starts it.
func start_bot_match(msg):
	main = get_tree().current_scene
	if main == null or not main.has_method("_on_match_ready"):
		_say("The game's main scene isn't ready yet.")
		return
	_send({"type": "match_start", "p1": msg.p1, "p2": msg.p2, "seed": msg.seed, "settings": msg.settings,
		"options": msg.options, "mod": VERSION}, "_on_match_started")
	_launch(msg)


# Start the match in this game. Host and client build the same match data from the same message.
func _launch(msg):
	match_ = {"p1": msg.p1, "p2": msg.p2, "seed": msg.seed}
	think_on = bool(msg.think)
	var match_settings = msg.settings
	turn_ms = int(float(match_settings.get("turn_time", 0)) * 1000) if str(match_settings.get("timer_mode", "none")) == "default" else 0
	var data = match_settings.duplicate(true)
	data["singleplayer"] = side == "single"
	data["selected_characters"] = {1: {"name": msg.p1}, 2: {"name": msg.p2}}
	data["seed"] = int(msg.seed)
	if side != "single":  # as the game's rematch resets them (Network.send_rematch_request)
		Network.forfeiter = 0
		Network.rematch_menu = false
	# As Main._on_game_started(true) then the character select's match_ready, minus the select.
	ReplayManager.playback = false
	ReplayManager.resimulating = false
	ReplayManager.replaying_ingame = false
	Global.frame_advance = false
	main.ui_layer.reset_ui()
	if is_instance_valid(main.game):
		main.game.free()
		main.game = null
		main.stop_ghost()
	main.singleplayer = side == "single"
	main.hide_main_menu(true)
	var css = main.get_node_or_null("%CharacterSelect")
	if css:
		css.hide()
	game = null
	turn = 0
	decision_tick = -1000
	auto_tick = -1000
	inputs_at = {}
	pick = null
	pick_pending = false
	desynced = false
	p2_fallback = false
	estimate = null
	shown_estimate = null
	pending_turn = -1
	depth_text = ""
	p1_locked_tick = -1000
	phase = "starting"
	if panel:
		panel.hide()
	main._on_match_ready(data)


# The standard settings with a preset's overrides, each in the type the game's own setting has
# (JSON numbers arrive as floats; the game's integer settings must stay integers).
func _with_preset(name):
	var s = settings.duplicate(true)
	var over = presets.get(name, {})
	for k in over:
		var v = over[k]
		match typeof(s.get(k)):
			TYPE_INT:
				v = int(v)
			TYPE_REAL:
				v = float(v)
			TYPE_STRING:
				v = str(v)
			TYPE_BOOL:
				v = bool(v)
		s[k] = v
	return s


func _on_match_started(r):
	if not r.get("ok", false):
		desynced = true
		_say("The bot can't play this match (it won't count): %s" % str(r.get("message", "the server refused it")))


func _attach():
	if not is_instance_valid(main) or not is_instance_valid(main.game) or not main.game.game_started:
		return
	game = main.game
	p1_panel = main.get_node("%P1ActionButtons")
	p2_panel = main.get_node("%P2ActionButtons")
	if side == "host":  # P2's moves come from the other game, through the game's own protocol
		game.connect("game_ended", self, "_on_host_game_ended")
		game.p1.connect("action_selected", self, "_on_host_move", [1])
		game.p2.connect("action_selected", self, "_on_host_move", [2])
		phase = "playing"
		return
	game.connect("game_ended", self, "_on_game_ended")
	game.p1.connect("action_selected", self, "_on_input", [1])
	game.p2.connect("action_selected", self, "_on_input", [2])
	if side == "single":
		# P2 is the bot's: no keyboard lock-in, no auto-continue, never shown.
		p2_panel.set_process_input(false)
		p2_panel.get_node("%AutoButton").set_pressed_no_signal(false)
	phase = "playing"


func _play_frame():
	if not is_instance_valid(game) or not is_instance_valid(main) or main.game != game:
		_abort("the match was closed")
		return
	if side == "single":
		if p2_panel.visible:
			p2_panel.hide()
		for p in [p1_panel, p2_panel]:
			var u = p.get_node("%UndoButton")
			if u.visible:
				u.hide()
	if game.game_finished:
		return
	if side == "host":
		_host_frame()
		return
	if game.game_paused and game.is_waiting_on_player() and game.p1_turn and game.p2_turn and game.current_tick != decision_tick:
		_decision()
	if pick != null and p2_panel.active and game.p2.state_interruptable:
		_submit_pick()
	# A per-turn timer: tell the server to answer now, 1.5 s before it runs out (any search mode). The
	# server stops itself earlier (its deadline margin); this is the game's own clock, as a backstop.
	if pick_pending and not deadline_sent and turn_ms > 0 and OS.get_ticks_msec() - decision_msec >= turn_ms - 1500:
		deadline_sent = true
		_notify({"type": "deadline", "turn": pending_turn})
	if autoplay and auto_tick != decision_tick and decision_tick == game.current_tick and p1_panel.active and game.p1.state_interruptable:
		auto_tick = decision_tick
		_autoplay_p1()


func _decision():
	var prev = decision_tick
	decision_tick = game.current_tick
	var msg = {"type": "decision", "turn": turn, "snapshot": _snapshot()}
	if turn > 0:
		msg["inputs"] = _inputs_for(prev)
		msg["p2_fallback"] = p2_fallback
	p2_fallback = false
	shown_estimate = estimate  # last turn's estimate: that turn has now been played
	estimate = null
	pick = null
	pick_pending = true
	pending_turn = turn
	decision_msec = OS.get_ticks_msec()
	deadline_sent = false
	depth_text = ""
	_send(msg, "_on_pick")
	var locked = p1_locked_tick == decision_tick if side == "client" else inputs_at.has(decision_tick) and inputs_at[decision_tick][0] != null
	if think_on and locked:
		_notify({"type": "locked_in", "turn": pending_turn})  # P1 locked in before this was sent
	turn += 1
	_report()


# Client: what the host's bottom bar shows. Last turn's estimate only (that turn has been played).
func _report():
	if side == "client":
		link.to_host("status", {"turn": turn, "desynced": desynced, "estimate": shown_estimate, "depth": depth_text})


# Client: the game's own "turn ready" (Network.multiplayer_turn_ready): THAT a player locked in. P1's
# input itself arrives only once both are ready (Network.send_action).
func _on_net_turn_ready(id):
	if id != 1 or not is_instance_valid(game):
		return
	p1_locked_tick = game.current_tick
	if think_on and pick_pending and phase == "playing" and decision_tick == game.current_tick:
		_notify({"type": "locked_in", "turn": pending_turn})


func _on_pick(r):
	pick_pending = false
	if phase != "playing":
		return
	if r.get("type") != "pick":
		_say("Bot server error: %s" % str(r.get("message")))
		r = {"action": "Continue", "data": null, "extra": null, "fallback": true}
	desynced = r.get("desynced", false) or desynced
	if r.get("fallback", false):
		p2_fallback = true
	estimate = r.get("estimate", null)
	var d = r.get("depth")
	depth_text = str(d.get("label", "")) if typeof(d) == TYPE_DICTIONARY else ""
	pick = r
	_report()


func _submit_pick():
	var a = str(pick.get("action", "Continue"))
	var d = _from_json(pick.get("data"))
	var e = _from_json(pick.get("extra"))
	pick = null
	# In sync the pick is legal by construction. Desynced, a move the real P2 can't use would turn
	# into a forfeit (BaseChar.on_action_selected), so P2 holds instead.
	if desynced and a != "Continue":
		var st = game.p2.state_machine.get_state(a)
		if st == null or not st.is_usable():
			a = "Continue"
			d = null
			e = null
			p2_fallback = true
	# Single: P2's panel's own Select call. Client: this game's player is P2, so the same call also does
	# Network.submit_action, exactly as a player's Select does online.
	p2_panel.on_action_submitted(a, d, e)
	if side == "single":
		p2_panel.hide()
	else:
		link.trace("turn %d: P2 locked in %d ms after the turn opened" % [pending_turn, OS.get_ticks_msec() - decision_msec])


func _on_input(_action, _data, _extra, id):
	# The input exactly as the fighter got it (before an unusable move becomes Forfeit), keyed by the
	# decision it answers. Copied now: the fighter's states may change the dicts during the turn.
	if not is_instance_valid(game):
		return
	if id == 1 and think_on and pick_pending and phase == "playing":
		_notify({"type": "locked_in", "turn": pending_turn})  # that P1 locked in, never what
	var f = game.p1 if id == 1 else game.p2
	var t = game.current_tick
	if not inputs_at.has(t):
		inputs_at[t] = [null, null]
	inputs_at[t][id - 1] = {"action": f.queued_action, "data": _dup(f.queued_data), "extra": _dup(f.queued_extra)}


func _inputs_for(tick):
	var out = inputs_at.get(tick, [null, null]).duplicate()
	inputs_at.erase(tick)
	for i in range(2):
		if out[i] == null:  # fall back on the replay's record of the input (the tick that used it)
			var rec = ReplayManager.frames[i + 1].get(tick + 1)
			if rec != null:
				out[i] = {"action": rec.action, "data": _dup(rec.data), "extra": _dup(rec.extra)}
	return out


func _on_game_ended(_winner = null):
	if phase != "playing":
		return
	phase = "ended"
	var w = null
	if game.p1.hp > game.p2.hp:
		w = 0
	elif game.p2.hp > game.p1.hp:
		w = 1
	_send({"type": "end", "turn": turn, "inputs": _inputs_for(decision_tick), "snapshot": _snapshot(), "winner": w},
		"_on_result")


func _on_result(r):
	if side == "client":
		phase = "idle"
		link.to_host("result", r)
		return
	games_played += 1
	if autoplay:
		print("YOMIHUMAN_RESULT ", JSON.print(r))
		yield(get_tree().create_timer(0.5), "timeout")
		_next_or_quit(r.get("next"))
		return
	phase = "idle"
	if r.get("type") != "result":
		_show_panel("Game over", "The bot server couldn't record this game: %s" % str(r.get("message")))
		return
	var res = {"human": "You win!", "bot": "The bot wins.", "draw": "Draw."}.get(r.get("result"), str(r.get("result")))
	var sync = "In sync every turn: the game counts." if r.get("valid", false) else "The bot's simulation lost sync: this game is marked invalid."
	_show_panel("Game over. %s (%d turns)" % [res, int(r.get("turns", 0))],
		sync + ("\nPlay again, or go back to the main menu." if hosted else "\nPlay again, or close the window to finish."))


func _abort(reason):
	if phase == "playing" or phase == "starting":
		if side == "host":
			link.to_client("abort", reason)
		else:
			_send({"type": "abort", "reason": reason})
	phase = "idle"
	game = null
	if not autoplay:
		_show_panel(_panel_title(), "The last match was closed before it ended.")


# --- multiplayer, host side ----------------------------------------------------------------------

# What the client relays (MpClient.gd -> VsBot.gd): the server's hello, the
# bottom bar's status, a message, the result.
func on_bot_message(kind, r):
	match kind:
		"hello":
			_on_hello(r)
		"status":
			remote_status = r if typeof(r) == TYPE_DICTIONARY else {}
		"say":
			_say(str(r))
		"result":
			_on_bot_result(r)


func _host_frame():
	if game.game_paused and game.is_waiting_on_player() and game.p1_turn and game.p2_turn and game.current_tick != decision_tick:
		decision_tick = game.current_tick
		turn += 1
		# The check that the bot's move stays hidden: everything of P2's this game holds as the turn opens.
		link.trace("turn %d opened (tick %d): P2 here: queued %s, network input %s, ghost %s, P2 locked in %s" % [
			turn, decision_tick, str(game.p2.queued_action), str(Network.action_inputs[2].action),
			str(main.p2_ghost_action), str(Network.turns_ready.get(2, false))])
	if autoplay and auto_tick != decision_tick and decision_tick == game.current_tick and p1_panel.active and game.p1.state_interruptable:
		auto_tick = decision_tick
		_autoplay_p1()


func _on_host_move(_action, _data, _extra, id):
	if not is_instance_valid(game):
		return
	if id == 1:  # P1's own panel: the fighter has it at once; the other game gets it once both are ready
		p1_lock_msec = OS.get_ticks_msec()
		link.trace("turn %d: P1 locked in (tick %d); P2 here: queued %s, network input %s, ghost %s, P2 locked in %s" % [
			turn, game.current_tick, str(game.p2.queued_action), str(Network.action_inputs[2].action),
			str(main.p2_ghost_action), str(Network.turns_ready.get(2, false))])
	else:  # Network.send_action: the bot's move, delivered by the game's own protocol
		link.trace("turn %d: P2's move arrived (tick %d) %d ms after P1 locked in: %s" % [
			turn, game.current_tick, OS.get_ticks_msec() - p1_lock_msec, str(game.p2.queued_action)])


func _on_host_game_ended(_winner = null):
	if phase != "playing":
		return
	phase = "ended"
	if autoplay:
		return
	var res = "You win!" if game.p1.hp > game.p2.hp else ("The bot wins." if game.p2.hp > game.p1.hp else "Draw.")
	_show_panel("Game over. %s" % res, "Waiting for the bot's record of the game...")


func _on_bot_result(r):
	games_played += 1
	if autoplay:
		print("YOMIHUMAN_RESULT ", JSON.print(r))
		link.trace("result " + JSON.print(r))
		yield(get_tree().create_timer(0.5), "timeout")
		_next_or_quit(r.get("next"))
		return
	phase = "idle"
	if r.get("type") != "result":
		_show_panel("Game over", "The bot server couldn't record this game: %s" % str(r.get("message")))
		return
	var res = {"human": "You win!", "bot": "The bot wins.", "draw": "Draw."}.get(r.get("result"), str(r.get("result")))
	var sync = "In sync every turn: the game counts." if r.get("valid", false) else "The bot's simulation lost sync: this game is marked invalid."
	_show_panel("Game over. %s (%d turns)" % [res, int(r.get("turns", 0))], sync + "\nPlay again, or go back to the main menu.")


# The game's in-match replay ("playback", R) rebuilds the match from its replay, and "edit_replay"
# cuts the match at a replayed tick and plays on from there: the real match would leave the sim
# behind. Both are swallowed while one of our matches is on (autoloads see input first).
func _input(event):
	if (phase == "playing" or phase == "ended") and (event.is_action_pressed("playback") or event.is_action_pressed("edit_replay")):
		get_tree().set_input_as_handled()


func _notification(what):
	if what == MainLoop.NOTIFICATION_WM_QUIT_REQUEST and connected and phase == "playing" and side != "host":
		_send({"type": "abort", "reason": "the game was closed"})


# --- state -------------------------------------------------------------------------------------

# Compact real state for the sync check: the fields of YomiMatch._player / _detail the server
# compares (human_play.PLAYER_FIELDS, OBJECT_FIELDS), computed the same way.
func _snapshot():
	var ps = []
	for p in [game.p1, game.p2]:
		var pos = p.get_pos()
		var vel = p.get_vel()
		var st = p.current_state()
		ps.append({"hp": p.hp, "x": pos.x, "y": pos.y, "vx": str(vel.x), "vy": str(vel.y),
			"facing": p.get_facing_int(), "state": st.state_name, "state_tick": st.current_tick,
			"hitlag": p.hitlag_ticks, "busy_interrupt": p.busy_interrupt, "grounded": p.is_grounded(),
			"meter": p.supers_available, "meter_partial": p.super_meter, "burst": p.bursts_available,
			"burst_partial": p.burst_meter, "combo": p.combo_count, "feints": p.feints, "air_options": p.air_option_bar})
	var objs = []
	for o in game.objects:
		if is_instance_valid(o) and not o.disabled:
			var opos = o.get_pos()
			var ost = o.current_state()
			objs.append({"kind": o.filename.get_file().get_basename(), "owner": o.id, "x": opos.x, "y": opos.y,
				"state": ost.state_name if ost else ""})
	return {"tick": game.current_tick, "players": ps, "objects": objs}


func _dup(v):
	if typeof(v) == TYPE_DICTIONARY or typeof(v) == TYPE_ARRAY:
		return v.duplicate(true)
	return v


# JSON numbers arrive as floats; the game's inputs are integers (ReplayManager.force_ints does the
# same for loaded replays). "[Object:null]" is how JSON prints a null Object (an option's icon).
func _from_json(v):
	match typeof(v):
		TYPE_REAL:
			return int(v) if v == floor(v) else v
		TYPE_STRING:
			return null if v == "[Object:null]" else v
		TYPE_DICTIONARY:
			var d = {}
			for k in v:
				d[k] = _from_json(v[k])
			return d
		TYPE_ARRAY:
			var a = []
			for x in v:
				a.append(_from_json(x))
			return a
	return v


# --- headless autoplay: P1 through its own panel -----------------------------------------------

func _autoplay_p1():
	var p = p1_panel
	for _i in range(3):
		yield(get_tree(), "idle_frame")
	if phase != "playing" or not p.active:
		return
	var legal = []
	for b in p.buttons:
		if is_instance_valid(b) and b.visible and not b.get_disabled():
			legal.append(b)
	if not legal.empty():
		var b = legal[rng.randi() % legal.size()]
		if not b.is_pressed():
			b.set_pressed(true)  # the button's toggled -> was_pressed -> on_action_selected, as a click
		for _i in range(3):
			yield(get_tree(), "idle_frame")
		if b.data_node and is_instance_valid(b.data_node):
			_randomize(b.data_node)
	var di = p.get_node("%DI")
	if di.is_visible_in_tree() and rng.randf() < 0.6:
		_random_xy(di)
	for name in ["%FeintButton", "%ReverseButton"]:
		var t = p.get_node(name)
		if t.is_visible_in_tree() and not t.disabled and rng.randf() < 0.25:
			t.pressed = not t.pressed
			p.send_ui_action()
	for _i in range(3):
		yield(get_tree(), "idle_frame")
	if autoplay_wait > 0:  # a slow player: the game's own turn timer may lock P1 in first
		yield(get_tree().create_timer(autoplay_wait), "timeout")
	if phase == "playing" and p.active:
		p._on_submit_pressed()  # the Select button


func _random_xy(n):
	var a = rng.randf() * TAU
	var r = sqrt(rng.randf()) * n.panel_radius
	n.update_value(Vector2(cos(a), sin(a)) * r, false, false)
	n.emit_signal("data_changed")


# Random values for a move's widgets, set through the widgets as a player's clicks would.
func _randomize(node):
	for c in node.get_children():
		if not (c is Control) or not c.is_visible_in_tree():
			continue
		if c.has_method("update_value") and c.get("panel_radius") != null:
			_random_xy(c)
		elif c is Range:
			c.value = c.min_value + rng.randf() * (c.max_value - c.min_value)
		elif (c is CheckBox or c is CheckButton) and rng.randf() < 0.3:
			c.pressed = not c.pressed
		else:
			_randomize(c)


# --- overlay -----------------------------------------------------------------------------------

func _make_overlay():
	layer = CanvasLayer.new()
	layer.layer = 120
	add_child(layer)
	var theme = load("res://theme.tres")
	status = Label.new()
	status.rect_position = Vector2(4, 346)
	if theme:
		status.theme = theme
	layer.add_child(status)
	panel = PanelContainer.new()
	if theme:
		panel.theme = theme
	panel.rect_position = Vector2(170, 40)
	panel.rect_min_size = Vector2(300, 0)
	layer.add_child(panel)
	var box = VBoxContainer.new()
	panel.add_child(box)
	title = Label.new()
	title.autowrap = true
	box.add_child(title)
	var row = HBoxContainer.new()
	box.add_child(row)
	var l1 = Label.new()
	l1.text = "You (P1):"
	row.add_child(l1)
	p1_opt = OptionButton.new()
	row.add_child(p1_opt)
	var l2 = Label.new()
	l2.text = "  Bot (P2):"
	row.add_child(l2)
	p2_opt = OptionButton.new()
	row.add_child(p2_opt)
	_fill_characters(characters)
	var row3 = HBoxContainer.new()
	box.add_child(row3)
	var l4 = Label.new()
	l4.text = "Settings:"
	row3.add_child(l4)
	settings_opt = OptionButton.new()
	for n in preset_names:
		settings_opt.add_item(n)
	row3.add_child(settings_opt)
	var row2 = HBoxContainer.new()
	box.add_child(row2)
	var l3 = Label.new()
	l3.text = "Bot picks:"
	row2.add_child(l3)
	pick_opt = OptionButton.new()
	pick_opt.add_item("mixed (samples its strategy)")
	pick_opt.add_item("strongest (its most likely move)")
	row2.add_child(pick_opt)
	estimate_box = CheckBox.new()
	estimate_box.text = "Show the bot's win estimate"
	box.add_child(estimate_box)
	think_box = CheckBox.new()
	think_box.text = "Bot thinks until you lock in"
	box.add_child(think_box)
	start_btn = Button.new()
	start_btn.text = "Start game"
	start_btn.connect("pressed", self, "_on_start_pressed")
	box.add_child(start_btn)
	if hosted:
		var back_btn = Button.new()
		back_btn.text = "Back to main menu"
		back_btn.connect("pressed", self, "_on_back_pressed")
		box.add_child(back_btn)
	info = Label.new()
	info.autowrap = true
	box.add_child(info)
	_show_panel("YomiBench: connecting to the bot server...", "")
	if autoplay:
		panel.hide()


func _on_back_pressed():
	if (phase == "playing" or phase == "starting") and side == "single":
		_send({"type": "abort", "reason": "back to the main menu"})  # host: the client goes with the session
	phase = "ended"
	emit_signal("leave_requested")


# The character pickers: `names` in order, keeping each picker's choice if it's still offered (else
# the launcher's --yomi-human-p1/p2, else the first).
func _fill_characters(names):
	var keep = [_option("human-p1", "Ninja"), _option("human-p2", "Ninja")]
	if p1_opt.get_item_count() > 0 and p1_opt.selected >= 0 and p2_opt.selected >= 0:
		keep = [characters[p1_opt.selected], characters[p2_opt.selected]]
	characters = []
	for c in names:
		characters.append(str(c))
	for k in range(2):
		var opt = [p1_opt, p2_opt][k]
		opt.clear()
		for c in characters:
			opt.add_item(c)
		opt.select(max(0, characters.find(keep[k])))


func _show_panel(t, i):
	if autoplay or panel == null:
		return
	title.text = t
	info.text = i
	panel.show()


func _say(t):
	print("YomiHuman: ", t)
	if side == "client":
		link.to_host("say", t)
		return
	if info:
		info.text = t
		if not autoplay and phase != "playing":
			panel.show()


func _update_status():
	if status == null:
		return
	if side == "host":
		_update_host_status()
		return
	var s = "YomiBench vs %s" % str(server.get("bot", "?")) if connected else "YomiBench: waiting for the bot server on port %d" % port
	if phase == "playing":
		s += "  |  game %d, turn %d" % [games_played + 1, turn]
		if desynced:
			s += "  |  OUT OF SYNC (game won't count)"
		if shown_estimate != null:
			s += "  |  bot's estimate: %d%% for the bot" % int(shown_estimate)
		if depth_text != "":
			s += "  |  bot thought: %s" % depth_text
		if pick_pending and is_instance_valid(game) and not game.p1.state_interruptable:
			s += "  |  bot is thinking..."
	status.text = s


# Host: no "bot is thinking": as online, the game's own timer bar and sound say when the other side locked in.
func _update_host_status():
	var s = "YomiBench multiplayer vs %s" % str(server.get("bot", "?")) if not server.empty() else "YomiBench: starting the bot's game..."
	if phase == "playing":
		s += "  |  game %d, turn %d" % [games_played + 1, turn]
		if remote_status.get("desynced", false) or desynced:
			s += "  |  OUT OF SYNC (game won't count)"
		if remote_status.get("estimate") != null:
			s += "  |  bot's estimate: %d%% for the bot" % int(remote_status.estimate)
		if str(remote_status.get("depth", "")) != "":
			s += "  |  bot thought: %s" % str(remote_status.depth)
	status.text = s
