extends Control
# FROZEN: candidate set 2 (main before set 3), kept only so an arena can give one seat set 2 and
# the other set 4 on the same engine (new_match "candidate_sets"). Do not change.
#
# Legal moves for one fighter, each with a small fixed set of parameter choices.
#
# Legality comes from Legality.gd (the game's own ActionButtons panel answers). Move data
# is never built by hand: the move's own data_ui_scene is instanced and its widgets are set to
# a few values, and the scene's get_data() produces the dict. So every data dict the server
# applies is one the game's UI could have sent (bad data panics the native fixed-point lib).
#
# Discretization (variant 0 is always the widget's default):
#   XY plot   default, then forward, forward-up 45, up, back, forward-down 45, back-up 45, down
#             at full length, then the same 7 at half length (landing spot and travel distance
#             depend on length for jumps, air dashes, teleports and some specials); the plot's own
#             limits (angle range, min length, always-max) apply, and duplicates collapse
#   8-way     default, then forward, forward-up, forward-down, up, down, back, back-up,
#             back-down, neutral (enabled directions only)
#   slider    default, max, min, midpoint     count  default, max, min, midpoint
#   checkbox  default, toggled                 dropdown  each item
#   block height  high, low
# Several widgets on one move vary one at a time (others at default), not as a product.
# Continue (always candidate 0, the protocol's default move) gets DI variants (neutral, toward,
# away, up, down) while the fighter is in a hurt state. A seat with no other option gets exactly
# one candidate, Continue.
# The per-seat budget keeps Continue and every legal move's default, then adds variants
# round-robin across moves, attacks and specials first, until max_candidates (0 = no limit: every
# variant). Move-coverage testing found the budget, not the angle set, loses most outcomes, and
# mostly on attack aims (docs/testing/move-coverage.md). Reverse, feint and prediction stay off.

const XY_ANGLES = [0, 45, 90, 180, -45, 135, -90]
const WAY_DIRS = ["E", "NE", "SE", "N", "S", "W", "NW", "SW", "Neutral"]
const DI_FULL = 100
const CATEGORY_ORDER = ["Continue", "Attack", "Special", "Super", "Movement", "Defense", "Other"]
const XY_LENGTHS = [1.0, 0.5]

var fighter = null
var max_candidates = 30
var button_states = []   # states shown in the action menu, in the UI's creation order
var data_nodes = {}      # state name -> instanced data_ui_scene
var extra_node = null    # the character's PlayerExtra scene (Wizard hover, Robot fly, ...)


class ButtonsStub:
	extends Reference
	# Data scenes peek at the opponent's selected button (Parry pre-fills its timing).
	var opponent_action_buttons = self
	var current_button = null


func setup(fighter_, max_candidates_):
	fighter = fighter_
	max_candidates = max_candidates_ if max_candidates_ > 0 else 1 << 30
	# Data scenes read is_visible_in_tree(), so this container stays "visible" (nothing renders).
	visible = true
	rect_position = Vector2(-10000, -10000)
	var stub = ButtonsStub.new()
	for category in fighter.action_cancels:
		for state in fighter.action_cancels[category]:
			if state.show_in_menu and not state in button_states:
				button_states.append(state)
				if state.data_ui_scene:
					var node = state.data_ui_scene.instance()
					node.action_buttons = stub
					node.state = state
					node.fighter = fighter
					add_child(node)
					node.init()
					data_nodes[state.name] = node
	if fighter.player_extra_params_scene:
		extra_node = fighter.player_extra_params_scene.instance()
		extra_node.set_fighter(fighter)
		add_child(extra_node)


# Returns this decision's candidates: [{id, action, data, extra}, ...], Continue first.
func generate():
	legal_epoch += 1
	_activate_extra()
	var moves = []   # [{action, state, variants: [{data, di}]}]
	moves.append({"action": "Continue", "state": null, "variants": _continue_variants()})
	for state in legal_states():
		moves.append({"action": state.name, "state": state, "variants": _data_variants(state)})
	# Budget: every move's default first, then extra variants round-robin in CATEGORY_ORDER.
	var picked = []
	for m in moves:
		picked.append(1)
	var order = []
	for cat in CATEGORY_ORDER:
		for i in range(moves.size()):
			if _category(moves[i].state) == cat:
				order.append(i)
	var total = moves.size()
	var round_ = 1
	var grew = true
	while total < max_candidates and grew:
		grew = false
		for i in order:
			if total >= max_candidates:
				break
			if moves[i].variants.size() > round_:
				picked[i] += 1
				total += 1
				grew = true
		round_ += 1
	var out = []
	for i in range(moves.size()):
		var m = moves[i]
		var extra_cache = {}
		for v in range(picked[i]):
			var variant = m.variants[v]
			var key = JSON.print(variant.di)
			if not extra_cache.has(key):
				extra_cache[key] = _extra_for(m.state, variant.di)
			out.append({"id": out.size(), "action": m.action, "data": variant.data, "extra": extra_cache[key]})
	return out


func _category(state):
	if state == null:
		return "Continue"
	var c = state.get_ui_category()
	return c if c in CATEGORY_ORDER else "Other"


# Candidate 0 of generate(): Continue with neutral DI and the default extras. Used for the
# protocol's default continuation without paying for a full generate().
func default_candidate():
	legal_epoch += 1
	_activate_extra()
	return {"id": 0, "action": "Continue", "data": null, "extra": _extra_for(null, {"x": 0, "y": 0})}


# --- legality (Legality.gd) -------------------------------------------------------------------

var legality = null      # this game's Legality node (YomiMatch._make_gens)
var legal_epoch = 0      # bumped by every new epoch: Legality caches its answers within one


# The menu moves (button_states, in menu order) legal now with the extra panel as it stands.
# fresh: the game state may have changed since this epoch began (YomiMatch._open_turn).
func legal_states(fresh = false):
	if fresh:
		legal_epoch += 1
	return legality.legal_states(self, _current_extra())


# --- extra (ActionButtons.get_extra / activate) ----------------------------------------------

# The extra panel (DI, feint, reverse, the move's own widgets) is reused for every candidate, so before
# each one it's brought back to a known baseline: visible, with its options, only while the fighter can act
# (the player sees no extras otherwise), and every widget at its default. The panel's own methods do the work.
func _activate_extra():
	if extra_node == null:
		return
	var can_act = not fighter.busy_interrupt
	extra_node.visible = can_act
	if can_act:
		extra_node.show_options()
	extra_node.reset()


func _current_extra():
	return extra_node.get_extra() if extra_node else {}


func _extra_for(state, di):
	var extra = {"DI": di, "reverse": false, "feint": false, "prediction": -1}
	if extra_node:
		extra_node.selected_move_will_restart = true
		extra_node.update_selected_move(state)
		extra.merge(extra_node.get_extra())
	return extra


func _continue_variants():
	var neutral = {"x": 0, "y": 0}
	var out = [{"data": null, "di": neutral}]
	if fighter.di_enabled and fighter.current_state().is_hurt_state:
		var toward = fighter.get_opponent_dir()
		for di in [{"x": DI_FULL * toward, "y": 0}, {"x": -DI_FULL * toward, "y": 0}, {"x": 0, "y": -DI_FULL}, {"x": 0, "y": DI_FULL}]:
			out.append({"data": null, "di": di})
	return out


# --- move data from the move's own UI scene ---------------------------------------------------

func _data_variants(state):
	var neutral = {"x": 0, "y": 0}
	if not data_nodes.has(state.name):
		return [{"data": null, "di": neutral}]
	var node = data_nodes[state.name]
	# Same sequence the action menu runs: update_buttons, then show the selected button's data.
	node.show()
	node.set_facing(fighter.get_facing_int())
	node.fighter_update()
	node.set_facing(fighter.get_opponent_dir())
	node.on_button_selected()
	var widgets = []
	_find_widgets(node, widgets)
	var out = []
	var seen = {}
	_add_variant(out, seen, node.get_data())
	# One widget at a time, interleaved so the budget spreads across widgets.
	var value_lists = []
	for w in widgets:
		value_lists.append(_widget_values(w) if w.node.is_visible_in_tree() else [])
	var i = 0
	var any = true
	while any:
		any = false
		for wi in range(widgets.size()):
			if i < value_lists[wi].size():
				any = true
				var w = widgets[wi]
				var saved = _save(w)
				_apply(w, value_lists[wi][i])
				_add_variant(out, seen, node.get_data())
				_restore(w, saved)
		i += 1
	var res = []
	for d in out:
		res.append({"data": d, "di": neutral})
	return res


func _add_variant(out, seen, data):
	var key = JSON.print(data)
	if seen.has(key):
		return
	seen[key] = true
	if typeof(data) == TYPE_DICTIONARY or typeof(data) == TYPE_ARRAY:
		data = data.duplicate(true)
	out.append(data)


func _find_widgets(node, out):
	for c in node.get_children():
		var kind = _kind(c)
		if kind != "":
			out.append({"kind": kind, "node": c})
		else:
			_find_widgets(c, out)


func _kind(n):
	var s = n.get_script()
	var file = s.resource_path.get_file().get_basename() if s else ""
	if file == "XYPlot" or file == "DIWheel":
		return "xy"
	if file == "8Way":
		return "8way"
	if file == "HorizSlider":
		return "slider"
	if file == "CountOption":
		return "count"
	if file == "BlockHeightSelector":
		return "height"
	if n is CheckButton and n.has_method("get_data"):
		return "check"
	if n is OptionButton and n.has_method("get_data"):
		return "option"
	return ""


func _range_node(w):
	if w.kind == "slider":
		return w.node.get_node("Direction")
	return w.node.get_node("HSlider")


# Non-default values to try for one widget, in priority order.
func _widget_values(w):
	var n = w.node
	match w.kind:
		"xy":
			var r = n.panel_radius
			var f = fighter.get_opponent_dir()
			var vals = []
			for length in XY_LENGTHS:
				for a in XY_ANGLES:
					var rad = deg2rad(a)
					vals.append(Vector2(cos(rad) * f, -sin(rad)) * r * length)
			return vals
		"8way":
			var vals = []
			for d in WAY_DIRS:
				var b = n.get_node("%" + n.dir_to_facing(d)) if n.consider_facing else n.get_node("%" + d)
				if not b.disabled:
					vals.append(d)
			return vals
		"slider", "count":
			var rn = _range_node(w)
			var lo = rn.min_value
			var hi = rn.max_value
			var mid = lo + (hi - lo) / 2.0
			if rn.step > 0:
				mid = lo + round((mid - lo) / rn.step) * rn.step
			return [hi, lo, mid]
		"check":
			return [not n.pressed]
		"option":
			var vals = []
			for i in range(n.get_item_count()):
				if i != n.selected and not n.is_item_disabled(i):
					vals.append(i)
			return vals
		"height":
			return [true, false]
	return []


func _save(w):
	var n = w.node
	match w.kind:
		"xy":
			return n.value_float
		"8way":
			return n.pressed_button.name if n.pressed_button else null
		"slider", "count":
			return _range_node(w).value
		"check":
			return n.pressed
		"option":
			return n.selected
		"height":
			return n.high.pressed
	return null


func _apply(w, v):
	var n = w.node
	match w.kind:
		"xy":
			n.update_value(v, false, true)
		"8way":
			n.set_dir(v)
		"slider", "count":
			_range_node(w).value = v
		"check":
			n.pressed = v
		"option":
			n.select(v)
		"height":
			n.set_height(v)


func _restore(w, saved):
	var n = w.node
	match w.kind:
		"xy":
			n.update_value(saved, false, true)
		"8way":
			if saved != null:
				n.set_dir(saved, true)
		_:
			_apply(w, saved)


# --- debug: every move and widget, and probes at chosen widget values ---------------------------
#
# `describe()` lists every move in the action menu with its legality now, and every widget its
# data scene, the PlayerExtra scene and the DI wheel show. `probe(spec)` sets those widgets the way
# a player would (the same widget methods mouse and keyboard input end up in) and returns the
# {action, data, extra} the menu would submit. Data still only ever comes from the widgets'
# get_data(); a probe only chooses the widget values. Widget and extra state is restored afterwards
# so probes never change what generate() returns.

const XYPlotScene = preload("res://ui/XYPlot/XYPlot.tscn")
var di_node = null       # a stand-in DI wheel (ActionButtons' DI is a default XYPlot)


func _ensure_di():
	if di_node == null:
		di_node = XYPlotScene.instance()
		add_child(di_node)
	return di_node


func describe():
	_activate_extra()
	var legal = {}
	for s in legal_states():
		legal[s.name] = true
	var moves = []
	for state in button_states:
		var m = {
			"name": state.name, "title": state.title if state.title else state.name,
			"type": _category(state), "grab": state.get("is_grab") == true,
			"air_type": int(state.air_type), "legal": legal.has(state.name), "widgets": [],
			"usable": state.is_usable(), "in_stance": state.allowed_in_stance(),
		}
		if legal.has(state.name):
			m["reversible"] = state.flip_allowed() and state.reversible
			_extra_select(state)
			m["feintable"] = state.can_feint() and (extra_node == null or extra_node.can_feint)
			if data_nodes.has(state.name):
				var node = _open_data_node(state)
				var ws = []
				_find_widgets(node, ws)
				for w in ws:
					m.widgets.append(_describe_widget(w))
		moves.append(m)
	var extra_ws = []
	if extra_node:
		_activate_extra()
		_extra_select(null)
		for w in _extra_widgets():
			extra_ws.append(_describe_widget(w))
	var preds = [-1]
	if fighter.opponent.state_interruptable and not fighter.opponent.busy_interrupt:
		for state in button_states:
			if not int(state.type) in preds:
				preds.append(int(state.type))
	var out = {"moves": moves, "extra_widgets": extra_ws, "predictions": preds,
		"busy": fighter.busy_interrupt, "hurt": fighter.current_state().is_hurt_state,
		"grounded": fighter.is_grounded(), "facing": fighter.get_facing_int(),
		"opponent_dir": fighter.get_opponent_dir(), "di_enabled": fighter.di_enabled}
	_activate_extra()
	return out


# The menu's sequence when a move button is selected (update_buttons, then its data scene).
func _open_data_node(state):
	var node = data_nodes[state.name]
	node.show()
	node.set_facing(fighter.get_facing_int())
	node.fighter_update()
	node.set_facing(fighter.get_opponent_dir())
	node.on_button_selected()
	return node


# ActionButtons.on_action_selected / extra_updated: the extra panel learns the selected move.
func _extra_select(state):
	if extra_node:
		extra_node.selected_move_will_restart = true
		extra_node.update_selected_move(state)


func _extra_widgets():
	var out = []
	_find_extra_widgets(extra_node, out)
	return out


func _find_extra_widgets(node, out):
	for c in node.get_children():
		var kind = _kind(c)
		if kind == "" and c is BaseButton and c.toggle_mode:
			kind = "toggle"
		if kind != "":
			out.append({"kind": kind, "node": c})
		else:
			_find_extra_widgets(c, out)


func _describe_widget(w):
	var n = w.node
	var d = {"kind": w.kind, "name": str(n.name), "visible": n.is_visible_in_tree()}
	match w.kind:
		"xy":
			d["radius"] = n.panel_radius
			d["panel_type"] = int(n.panel_type)
			d["bottom_half"] = n.bottom_half
			d["always_max"] = n.always_max
			d["min_length"] = n.min_length
			d["limit_angle"] = n.limit_angle
			d["limit_center_degrees"] = n.limit_center_degrees
			d["limit_range_degrees"] = n.limit_range_degrees
			d["limit_symmetrical"] = n.limit_symmetrical
			d["snap_angles"] = n.snap_angles if n.snap else 0
			d["facing"] = n.facing
			d["value"] = n.get_data()
		"8way":
			var dirs = []
			for dir in WAY_DIRS:
				var b = n.get_node("%" + n.dir_to_facing(dir)) if n.consider_facing else n.get_node("%" + dir)
				if not b.disabled:
					dirs.append(dir)
			d["enabled"] = dirs
			d["value"] = n.get_dir()
		"slider", "count":
			var rn = _range_node(w)
			d["min"] = rn.min_value
			d["max"] = rn.max_value
			d["step"] = rn.step
			d["value"] = rn.value
		"check", "toggle":
			d["pressed"] = n.pressed
			d["disabled"] = n.disabled
		"option":
			var items = []
			for i in range(n.get_item_count()):
				items.append({"text": n.get_item_text(i), "disabled": n.is_item_disabled(i)})
			d["items"] = items
			d["selected"] = n.selected
		"height":
			d["value"] = n.high.pressed
	return d


# spec: {"action": name or "Continue", "widgets": {"<index>": value}, "extra": {"<index>": value},
#        "di": {"x": fx, "y": fy, "snap": bool} (fractions of the wheel radius, screen axes),
#        "feint": bool, "reverse": bool, "prediction": int}
# Widget values: xy {"x", "y", "snap"} (fractions of the radius, screen axes; snap = mouse input,
# no snap = keyboard nudges), 8way a direction name relative to facing (E = toward the opponent),
# slider/count a number, check/toggle/height a bool, option an item index.
# Returns {"candidate": {action, data, extra}} or {"error": message}.
func probe(spec):
	_activate_extra()
	var saved_extra = _save_extra()
	var res = _probe(spec)
	_restore_extra(saved_extra)
	_activate_extra()
	return res


func _probe(spec):
	var action = spec.get("action", "Continue")
	var state = null
	if action != "Continue":
		for s in button_states:
			if s.name == action:
				state = s
		if state == null:
			return {"error": "no move %s" % action}
	_extra_select(state)
	# Extra widgets first (as a player toggles them after picking the move); each click re-runs
	# the menu's extra_updated.
	var ewidgets = _extra_widgets() if extra_node else []
	var espec = spec.get("extra", {})
	for k in espec:
		var i = int(k)
		if i < 0 or i >= ewidgets.size():
			return {"error": "no extra widget %s" % k}
		var err = _set_widget(ewidgets[i], espec[k], true)
		if err != "":
			return {"error": "extra widget %s: %s" % [k, err]}
		_extra_select(state)
	if state != null:
		var ok = false
		for s in legal_states():
			if s == state:
				ok = true
		if not ok:
			return {"error": "illegal: %s is not available now" % action}
	var reverse = spec.get("reverse", false) == true
	var feint = spec.get("feint", false) == true
	if reverse and (state == null or not (state.flip_allowed() and state.reversible)):
		return {"error": "reverse is disabled for %s" % action}
	if feint and (state == null or not state.can_feint() or (extra_node != null and not extra_node.can_feint)):
		return {"error": "feint is disabled for %s" % action}
	var data = null
	if state != null and data_nodes.has(state.name):
		var node = _open_data_node(state)
		var ws = []
		_find_widgets(node, ws)
		var saved = []
		for w in ws:
			saved.append(_save(w))
		var wspec = spec.get("widgets", {})
		var err = ""
		for k in wspec:
			var i = int(k)
			if i < 0 or i >= ws.size():
				err = "no widget %s" % k
				break
			err = _set_widget(ws[i], wspec[k], false)
			if err != "":
				err = "widget %s: %s" % [k, err]
				break
		if err == "":
			# The menu's reverse button flips the data scene's facing (ActionButtons._send_ui_action).
			if reverse:
				node.set_facing(-fighter.get_opponent_dir())
			data = node.get_data()
			if typeof(data) == TYPE_DICTIONARY or typeof(data) == TYPE_ARRAY:
				data = data.duplicate(true)
			if reverse:
				node.set_facing(fighter.get_opponent_dir())
		for i in range(ws.size()):
			_restore(ws[i], saved[i])
		if err != "":
			return {"error": err}
	var di = {"x": 0, "y": 0}
	if spec.has("di"):
		var dn = _ensure_di()
		dn.reset()
		var v = spec.di
		var p = Vector2(float(v.get("x", 0)), float(v.get("y", 0))).limit_length(1.0) * dn.panel_radius
		dn.update_value(p, false, v.get("snap", false) != true)
		di = dn.get_data()
	var extra = {"DI": di, "reverse": reverse, "feint": feint, "prediction": int(spec.get("prediction", -1))}
	if extra_node:
		extra.merge(extra_node.get_extra())
	return {"candidate": {"action": action, "data": data, "extra": extra}}


# Sets a widget as the player would; returns "" or why the player couldn't.
func _set_widget(w, v, click):
	var n = w.node
	if not n.is_visible_in_tree():
		return "not visible"
	match w.kind:
		"xy":
			if typeof(v) != TYPE_DICTIONARY:
				return "xy value must be {x, y}"
			var p = Vector2(float(v.get("x", 0)), float(v.get("y", 0))).limit_length(1.0) * n.panel_radius
			n.update_value(p, false, v.get("snap", false) != true)
			if click:
				n.emit_signal("data_changed")
		"8way":
			if not v in WAY_DIRS:
				return "unknown direction %s" % str(v)
			var b = n.get_node("%" + n.dir_to_facing(v)) if n.consider_facing else n.get_node("%" + v)
			if b.disabled:
				return "direction %s is disabled" % v
			n.set_dir(v)
		"slider", "count":
			_range_node(w).value = float(v)
		"check", "height":
			if n.get("disabled") == true:
				return "disabled"
			_apply(w, v == true)
		"toggle":
			if n.disabled:
				return "disabled"
			if n.pressed != (v == true):
				n.pressed = v == true    # emits toggled, as a click does
				n.emit_signal("pressed")
		"option":
			var i = int(v)
			if i < 0 or i >= n.get_item_count() or n.is_item_disabled(i):
				return "no enabled item %s" % str(v)
			n.select(i)
			n.emit_signal("item_selected", i)
	return ""


func _save_extra():
	if extra_node == null:
		return null
	var ws = _extra_widgets()
	var vals = []
	for w in ws:
		match w.kind:
			"toggle":
				vals.append(w.node.pressed)
			_:
				vals.append(_save(w))
	var vars = {}
	for p in extra_node.get_script().get_script_property_list():
		var v = extra_node.get(p.name)
		if typeof(v) != TYPE_OBJECT:
			vars[p.name] = v.duplicate(true) if typeof(v) == TYPE_DICTIONARY or typeof(v) == TYPE_ARRAY else v
	return {"ws": ws, "vals": vals, "vars": vars}


func _restore_extra(saved):
	if saved == null:
		return
	for i in range(saved.ws.size()):
		var w = saved.ws[i]
		var v = saved.vals[i]
		match w.kind:
			"toggle":
				w.node.set_pressed_no_signal(v)
			"8way":
				if v != null and (w.node.pressed_button == null or w.node.pressed_button.name != v):
					if w.node.pressed_button:
						w.node.pressed_button.set_pressed_no_signal(false)
					w.node._on_button_pressed_no_signal(w.node.get_node("%" + v))
			_:
				_restore(w, v)
	for k in saved.vars:
		extra_node.set(k, saved.vars[k])
