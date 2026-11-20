extends Node
# A read-only description of everything each fighter can choose right now (`action_space` command),
# read off the game's own action menu at runtime: no game logic is copied.
#
# Like Legality.gd (game mode), this node holds a pair of real P1/P2 ActionButtons panels
# (res://ui/ActionSelector/ActionButtons.tscn) wired to one game as the game's UILayer wires them.
# Its own pair, kept shown (off screen): data scenes and extra panels read is_visible_in_tree(), and
# the legality panels stay hidden. For each seat it replays what the menu does when a turn opens
# (the extra panel's hide/show_options/reset, update_buttons) and when a player clicks a move
# (on_action_selected, the data scene shown and facing the opponent), and reads:
#   - legal moves: buttons update_buttons left visible and enabled
#   - each move's data scene: its widgets and their exported parameters (XYPlot angle limits,
#     panel shape, length limits; slider / count ranges; 8-way enabled directions; check, option
#     and block-height choices), and how get_data() assembles them
#   - reverse / feint: the menu's own ReverseButton / FeintButton enabled state with the move picked
#   - DI: the panel's DI wheel
#   - the PlayerExtra panel: every control, type, range, which move shows it, and which toggle
#     reveals it (found by clicking each toggle and watching what appears)
# Values are in the units get_data() / get_extra() send: XY and DI integer percents of the radius
# (x right, y down on screen), sliders and counts integers.

const PANEL_SCENE = "res://ui/ActionSelector/ActionButtons.tscn"
const GENERIC_DATA = "res://ui/ActionSelector/ActionUIData/ActionUIData.gd"

var game = null
var panels = []


func setup(game_):
	game = game_
	var scene = load(PANEL_SCENE)
	var holder = Control.new()
	holder.name = "ActionSpacePanels"
	holder.rect_position = Vector2(-20000, -20000)
	for id in [1, 2]:
		var p = scene.instance()
		p.name = "P%dActionButtons" % id
		p.player_id = id
		p.opponent_action_buttons_path = NodePath("../P%dActionButtons" % (3 - id))
		holder.add_child(p)
		panels.append(p)
	add_child(holder)
	var registered = Network.action_button_panels.duplicate()
	for id in [1, 2]:
		var p = panels[id - 1]
		p.opposite_buttons = panels[2 - id]
		p.init(game, id)
		p.set_process(false)
		p.set_physics_process(false)
		p.set_process_input(false)
	Network.action_button_panels = registered


# Both seats: [{"moves": [...], "extra": {...}, "di": widget or null}, ...].
func describe():
	return [_seat(panels[0]), _seat(panels[1])]


func _seat(p):
	var f = p.fighter
	var fx = p.fighter_extra
	p.show()
	p.current_button = null
	p.current_action = null
	# ActionButtons.activate: the extra panel around the legality pass.
	fx.hide()
	p.update_buttons(false)
	if not f.busy_interrupt:
		fx.show()
		fx.show_options()
	fx.reset()
	var out = {"busy_interrupt": f.busy_interrupt, "moves": [], "di": null}
	if p.get_node("%DI").visible:
		out.di = _widget(p.get_node("%DI"), "xy")
	var legal = []
	for b in p.buttons:
		if b != p.continue_button and b.visible and not b.get_disabled():
			legal.append(b)
	var extra = {"widgets": {}, "requires": {}, "shown_by": {}, "key_widget": {}}
	_extra_pass(p, null, extra)
	for b in legal:
		out.moves.append(_move(p, b, extra))
	# Moves an extra toggle makes legal (Cowboy's shift / drift set input_grounded / input_aerial):
	# each visible toggle clicked with Continue selected, the legality pass rerun.
	var more = {}
	_extra_select(p, null)
	var toggles = _extra_visible(p.fighter_extra)
	for name in toggles:
		var t = toggles[name]
		if t[0] != "toggle" or t[1].disabled:
			continue
		_click(p, t[1], null)
		p.update_buttons(false)
		for b in p.buttons:
			if b != p.continue_button and b.visible and not b.get_disabled() and not b in legal and not more.has(b):
				more[b] = name
		_extra_select(p, null)
	p.update_buttons(false)
	for b in more:
		# Described with the toggle on, as a player sees the move after clicking it.
		_extra_select(p, null)
		_click(p, toggles[more[b]][1], null)
		p.update_buttons(false)
		var m = _move(p, b, extra)
		m["requires_extra"] = more[b]
		out.moves.append(m)
	_extra_select(p, null)
	p.update_buttons(false)
	_unselect(p)
	_extra_pass(p, null, null)  # leave Continue selected, as the menu opens
	extra["keys"] = fx.get_extra().duplicate(true)
	for k in extra.keys:
		if not extra.key_widget.has(k):
			extra.key_widget[k] = null   # no visible control changed it at this decision
	out["extra"] = extra
	p.hide()
	return out


# The menu with move button b picked (ActionButtons.on_action_selected), then read.
func _move(p, b, extra):
	var f = p.fighter
	var s = b.state
	p.on_action_selected(b.action_name, b)
	p.buffered_ui_actions = []   # its send_ui_action only queues; nothing here flushes it
	var m = {"name": b.action_name, "title": b.action_title, "category": s.get_ui_category() if s else "",
		"reversible": not p.get_node("%ReverseButton").disabled,
		"feint": not p.get_node("%FeintButton").disabled, "data": null}
	var dn = b.data_node
	if dn != null:
		# show_button_data_node (deferred in the menu): shown, facing the opponent (or away, reversed).
		dn.show()
		dn.set_facing(f.get_opponent_dir())
		m.data = _data_spec(dn)
	m["extra_visible"] = _extra_pass(p, s, extra)
	if dn != null:
		dn.hide()
	_unselect(p)
	return m


func _unselect(p):
	for b in p.buttons:
		if is_instance_valid(b):
			b.set_pressed_no_signal(false)
	p.current_button = null
	p.current_action = null
	p.button_pressed = false
	p.buffered_ui_actions = []


# How a data scene's get_data() is built: one widget's data, a dict of its children's data by
# name (ActionUIData.get_data), or the scene's own get_data() (custom: widgets listed, plus the
# value it sends now).
func _data_spec(dn):
	var script = dn.get_script().resource_path if dn.get_script() else ""
	var kids = dn.get_children()
	var spec = {"scene_script": script.get_file(), "default": _plain(dn.get_data())}
	if script != GENERIC_DATA:
		spec["shape"] = "custom"
		var ws = []
		_find(dn, ws, false)
		spec["widgets"] = []
		for w in ws:
			spec.widgets.append(_widget(w[1], w[0]))
		return spec
	if kids.size() == 1:
		spec["shape"] = "single"
		var k = _kind(kids[0])
		spec["widget"] = _widget(kids[0], k) if k != "" else {"kind": "unknown", "name": str(kids[0].name)}
	else:
		spec["shape"] = "fields"
		spec["fields"] = {}
		for c in kids:
			var k = _kind(c)
			spec.fields[str(c.name)] = _widget(c, k) if k != "" else {"kind": "unknown", "name": str(c.name)}
	return spec


# The extra panel with this move selected (ActionButtons.on_action_selected / extra_updated):
# which controls show. With `extra`, records each control (first sighting), which moves show it
# (shown_by), and which toggle reveals it: each visible toggle is clicked, the controls that appear
# require it, and the panel is reset. Returns the names of the controls visible.
func _extra_pass(p, state, extra):
	var fx = p.fighter_extra
	_extra_select(p, state)
	var visible = _extra_visible(fx)
	if extra == null:
		return visible.keys()
	for name in visible:
		var w = visible[name]
		if not extra.widgets.has(name):
			extra.widgets[name] = _widget(w[1], w[0])
		var by = extra.shown_by.get(name, [])
		var who = "Continue" if state == null else state.name
		if not who in by:
			by.append(who)
		extra.shown_by[name] = by
	# Which get_extra() key each control drives: move it, see which keys change.
	for name in visible:
		var before = fx.get_extra().duplicate(true)
		if _nudge(visible[name]):
			fx.selected_move_will_restart = true
			fx.update_selected_move(state)   # extra_updated, as for a click
			var after_extra = fx.get_extra()
			for k in after_extra:
				if not extra.key_widget.has(k) and str(after_extra[k]) != str(before.get(k)):
					extra.key_widget[k] = name
			_extra_select(p, state)
	for name in visible:
		var w = visible[name]
		if w[0] != "toggle" or w[1].disabled:
			continue
		_click(p, w[1], state)
		var after = _extra_visible(fx)
		for n2 in after:
			if not visible.has(n2):
				if not extra.requires.has(n2):
					extra.requires[n2] = {"toggle": name, "pressed": w[1].pressed}
				if not extra.widgets.has(n2):
					extra.widgets[n2] = _widget(after[n2][1], after[n2][0])
		_extra_select(p, state)
	return visible.keys()


# A click on an extra toggle: its toggled handlers run, then the panel hears of it as the menu's
# extra_updated() tells it (update_selected_move with the move picked), which is where e.g. Cowboy's
# shift / drift set input_aerial / input_grounded.
func _click(p, button, state):
	button.pressed = not button.pressed
	p.fighter_extra.selected_move_will_restart = true
	p.fighter_extra.update_selected_move(state)


func _extra_select(p, state):
	var fx = p.fighter_extra
	fx.hide()
	if not p.fighter.busy_interrupt:
		fx.show()
	fx.reset()
	fx.selected_move_will_restart = true
	fx.update_selected_move(state)
	if not p.fighter.busy_interrupt:
		fx.show_options()
	fx.update_selected_move(state)


func _extra_visible(fx):
	var ws = []
	_find(fx, ws, true)
	var out = {}
	for w in ws:
		if w[1].is_visible_in_tree():
			out[str(w[1].name)] = w
	return out


# Moves one control off its value the way a click would (its own signals run); false if it can't.
func _nudge(w):
	var n = w[1]
	match w[0]:
		"toggle", "check":
			if n.disabled:
				return false
			n.pressed = not n.pressed
		"xy":
			var r = n.panel_radius
			var v = n.get_value_float()
			var to = Vector2(0, -r) if v.distance_to(Vector2(0, -r)) > 1 else Vector2(r, 0)
			n.update_value(to)
			n.emit_signal("data_changed")
		"8way":
			for b in n.get_buttons():
				if b != n.pressed_button and not b.disabled and b.visible:
					n._on_button_pressed(b)
					return true
			return false
		"slider", "count":
			var rn = n.get_node("Direction") if w[0] == "slider" else n.get_node("HSlider")
			rn.value = rn.min_value if rn.value != rn.min_value else rn.max_value
		"option":
			if n.get_item_count() < 2:
				return false
			n.select((n.selected + 1) % n.get_item_count())
			n.emit_signal("item_selected", n.selected)
		_:
			return false
	return true


func _find(node, out, toggles):
	for c in node.get_children():
		var k = _kind(c)
		if k == "" and toggles and c is BaseButton and c.toggle_mode:
			k = "toggle"
		if k != "":
			out.append([k, c])
		else:
			_find(c, out, toggles)


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


# A widget's parameters, read from the node (exported values and the widget's own getters).
func _widget(n, kind):
	var d = {"kind": kind, "name": str(n.name), "visible": n.is_visible_in_tree()}
	match kind:
		"xy":
			var r = n.panel_radius
			var mid = n.midpoint()
			var size = n.plot_panel_size
			var semi = null
			for c in n.get_children():
				if c.get("semicircle") != null:
					semi = c.semicircle
			d["script"] = n.get_script().resource_path.get_file()
			d["panel_type"] = ["full", "half", "quarter"][int(n.panel_type)]
			d["semicircle"] = semi
			d["facing"] = n.facing
			d["radius_px"] = r
			# The reachable box in percents: the plot panel around its origin (the mouse can't
			# leave the panel), and the circle of radius 100.
			d["box"] = {"x_min": _pct(-mid.x, r), "x_max": _pct(size.x - mid.x, r),
				"y_min": _pct(-mid.y, r), "y_max": _pct(size.y - mid.y, r)}
			d["bottom_half"] = n.bottom_half
			d["left_quarter"] = n.left_quarter
			d["always_max"] = n.always_max
			d["min_length"] = n.min_length
			d["limit_angle"] = n.limit_angle
			d["limit_center_degrees"] = n.limit_center_degrees
			d["limit_range_degrees"] = n.limit_range_degrees
			d["limit_symmetrical"] = n.limit_symmetrical
			if n.limit_angle:
				var v = n.get_limit_vec()   # the limit's centre on screen, facing applied
				d["limit_center_screen"] = {"x": v.x, "y": v.y, "degrees": rad2deg(v.angle())}
			d["snap_angles"] = n.snap_angles if n.snap else 0
			d["default"] = n.get_data()
		"8way":
			var en = []
			for b in n.get_buttons():
				if not b.disabled and b.visible:
					en.append(str(b.name))
			d["enabled_screen"] = en   # button names on screen; get_data() sends their unit vectors
			d["values"] = []
			for name in en:
				d.values.append(n.get_value(name))
			d["consider_facing"] = n.consider_facing
			d["facing"] = n.facing
			d["default"] = n.get_data()
		"slider", "count":
			var rn = n.get_node("Direction") if kind == "slider" else n.get_node("HSlider")
			d["min"] = rn.min_value
			d["max"] = rn.max_value
			d["step"] = rn.step
			d["exported_min"] = n.min_value
			d["exported_max"] = n.max_value
			if kind == "slider":
				d["centered"] = n.centered
			d["key"] = "x" if kind == "slider" else "count"
			d["default"] = n.get_data()
		"check", "toggle":
			d["pressed"] = n.pressed
			d["disabled"] = n.disabled
		"option":
			var items = []
			for i in range(n.get_item_count()):
				items.append({"id": i, "text": n.get_item_text(i), "disabled": n.is_item_disabled(i)})
			d["items"] = items
			d["selected"] = n.selected
		"height":
			d["default"] = n.get_data()
			var was_high = n.high.pressed
			n.set_height(true)
			var hi = n.get_data()
			n.set_height(false)
			var lo = n.get_data()
			n.set_height(was_high)
			d["values"] = [hi, lo]   # high, low
	return d


func _pct(px, r):
	return px / float(r) * 100.0


func _plain(v):
	if typeof(v) == TYPE_DICTIONARY or typeof(v) == TYPE_ARRAY:
		return v.duplicate(true)
	if typeof(v) == TYPE_OBJECT:
		return str(v)
	return v
