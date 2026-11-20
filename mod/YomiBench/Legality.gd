extends Node
# Which action-menu moves a fighter may pick now, for one game (the real match or the preview game).
# The candidate generators (Candidates*.gd) ask this node through legal_states(gen, extra).
#
# The game's own ActionButtons panel answers: a hidden pair of real P1/P2 panels
# (ui/ActionSelector/ActionButtons.tscn), wired to this game the way the game's UILayer does it
# (ActionButtons.init(game, id), each panel the other's opponent). Each question runs the panel's own
# update_buttons() and reads which buttons it left visible, so legality is always the game's own.

const PANEL_SCENE = "res://ui/ActionSelector/ActionButtons.tscn"

var mode = "game"
var game = null
var panels = []          # [P1 panel, P2 panel]
var shim = null          # the PlayerExtra the panels read the asked extra dict from
var cache = {}           # gen instance id -> [gen.legal_epoch, {extra JSON: [states]}] (game mode)


# The extra panel a game's ActionButtons reads in get_extra(): here it returns the extra dict the
# generator asks about (its own PlayerExtra panel's get_extra(), with an extra set applied), so the
# real panel sees exactly the extras the candidate will carry.
class ExtraShim extends "res://ui/ActionSelector/PlayerExtra.gd":
	var extra = {}

	func get_extra():
		return extra


func setup(game_, _mode = "game", _check_log = "", _counts = null):
	game = game_
	_make_panels()


func _make_panels():
	var scene = load(PANEL_SCENE)
	# Both panels go under a holder that enters the tree in one piece, so each one's onready
	# opponent_action_buttons path finds the other.
	var holder = Control.new()
	holder.name = "Panels"
	holder.visible = false
	holder.rect_position = Vector2(-10000, -10000)
	for id in [1, 2]:
		var p = scene.instance()
		p.name = "P%dActionButtons" % id
		p.player_id = id
		p.opponent_action_buttons_path = NodePath("../P%dActionButtons" % (3 - id))
		holder.add_child(p)
		panels.append(p)
	add_child(holder)
	# init() registers each panel in Network.action_button_panels; keep the game's own entries.
	var registered = Network.action_button_panels.duplicate()
	for id in [1, 2]:
		var p = panels[id - 1]
		p.opposite_buttons = panels[2 - id]   # BottomBar._ready
		p.init(game, id)                      # UILayer.setup_action_buttons
		# Nothing renders or reads input here, and a panel's per-frame code is for a shown menu.
		p.set_process(false)
		p.set_physics_process(false)
		p.set_process_input(false)
	Network.action_button_panels = registered
	shim = ExtraShim.new()


func _exit_tree():
	if shim != null:
		shim.free()
		shim = null


func legal_states(gen, extra):
	return _game(gen, extra)


func _names(states):
	var out = []
	for s in states:
		out.append(s.name)
	return out


# The game's panel for this fighter, asked with this extra: update_buttons() leaves exactly the
# legal moves' buttons visible (it hides every button, then shows each one a cancel allows).
# Within one generator epoch (one generate / expand / default_candidate call) the game state is
# fixed, so each distinct extra is asked once per epoch.
func _game(gen, extra):
	var key = JSON.print(extra)
	var id = gen.get_instance_id()
	var c = cache.get(id)
	if c == null or c[0] != gen.legal_epoch:
		c = [gen.legal_epoch, {}]
		cache[id] = c
	if c[1].has(key):
		return c[1][key].duplicate()
	var panel = panels[0] if gen.fighter == game.p1 else panels[1]
	var own = panel.fighter_extra
	shim.extra = extra
	panel.fighter_extra = shim
	panel.update_buttons(false)
	panel.fighter_extra = own
	var out = []
	for b in panel.buttons:
		if b != panel.continue_button and b.visible:
			out.append(b.state)
	c[1][key] = out
	return out.duplicate()
