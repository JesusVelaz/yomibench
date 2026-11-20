extends Node
# Video renderer: plays one replay file in the real game and saves every drawn frame as a PNG,
# then quits a little after the match ends. Loaded instead of YomiServer by `yomibench render`.
# Args: --yomi-replay=PATH --yomi-frames=DIR [--yomi-tail=FRAMES] [--yomi-states=PATH]
#
# --yomi-states=PATH: a state log instead of a video: no frames or sounds are saved; every game tick
# the replay plays is logged to PATH as one JSON line (tick, both fighters, live objects), for checking
# a simulator against the real game on recorded human games (replay sync).
#
# Sound: the game's audio can't be recorded live, because a render runs slower than real time
# (every frame is stamped 1/60 s by --fixed-fps) while the mixer runs in real time. So every sound
# the game starts is logged instead, with the video frame it starts on, and each sound file is
# saved once; `yomibench render` mixes them into the video's audio track afterwards.
#   DIR.sounds.jsonl   one JSON object per line: buses, stream (a saved file), start, stop
#   DIR.sounds/        the saved sound files (game assets: never commit or publish them)

const MAX_FRAMES = 60 * 60 * 20  # a 20-minute cap, for a replay that never finishes

var replay_path = ""
var out_dir = ""
var tail = 150
var started = false
var frame = 0
var finished_at = -1
var audio_nodes = []   # every AudioStreamPlayer(2D) in the tree, tracked as they're added
var audio_state = {}   # instance id -> [playing, playback position, stream]
var saved_streams = {} # stream key -> saved file name
var sound_log = null
var states_path = ""
var states_log = null
var last_tick = -1

func _ready():
	for a in OS.get_cmdline_args():
		if a.begins_with("--yomi-replay="):
			replay_path = a.substr(14)
		elif a.begins_with("--yomi-frames="):
			out_dir = a.substr(14)
		elif a.begins_with("--yomi-tail="):
			tail = int(a.substr(12))
		elif a.begins_with("--yomi-states="):
			states_path = a.substr(14)
	if replay_path == "":
		return
	pause_mode = PAUSE_MODE_PROCESS
	call_deferred("_start")

func _start():
	for _i in range(60):  # let the title scene finish setting up
		yield(get_tree(), "idle_frame")
	var main = get_tree().current_scene
	print("render: loading ", replay_path)
	# The same path as picking the file in the game's replay browser.
	main.ui_layer.emit_signal("loaded_replay", ReplayManager.load_replay(replay_path))
	started = true
	if states_path != "":
		states_log = File.new()
		if states_log.open(states_path, File.WRITE) != OK:
			print("render: cannot write ", states_path)
			get_tree().quit()
			return
		tail = 1
	else:
		_start_sound_log()
	VisualServer.connect("frame_post_draw", self, "_capture")

func _start_sound_log():
	Directory.new().make_dir_recursive(out_dir + ".sounds")
	sound_log = File.new()
	if sound_log.open(out_dir + ".sounds.jsonl", File.WRITE) != OK:
		sound_log = null
		return
	var buses = {}
	for i in range(AudioServer.bus_count):
		buses[AudioServer.get_bus_name(i)] = {"volume_db": AudioServer.get_bus_volume_db(i),
			"mute": AudioServer.is_bus_mute(i), "send": AudioServer.get_bus_send(i)}
	sound_log.store_line(to_json({"ev": "buses", "buses": buses}))
	get_tree().connect("node_added", self, "_on_node_added")
	_track_tree(get_tree().root)

func _track_tree(n):
	_on_node_added(n)
	for c in n.get_children():
		_track_tree(c)

func _on_node_added(n):
	if n is AudioStreamPlayer or n is AudioStreamPlayer2D:
		audio_nodes.append(n)

func _stream_file(st):
	# Save each stream's data once; returns its file name in DIR.sounds/.
	var key = st.resource_path if st.resource_path != "" else str(st.get_instance_id())
	if key in saved_streams:
		return saved_streams[key]
	var fname = "s%03d" % saved_streams.size()
	var info = {"ev": "stream", "key": key, "loop": false}
	if st is AudioStreamSample:
		fname += ".wav"
		st.save_to_wav(out_dir + ".sounds/" + fname)
		info["loop"] = st.loop_mode != AudioStreamSample.LOOP_DISABLED
		info["loop_begin"] = st.loop_begin
		info["loop_end"] = st.loop_end
		info["mix_rate"] = st.mix_rate
	elif st is AudioStreamOGGVorbis or st is AudioStreamMP3:
		fname += ".ogg" if st is AudioStreamOGGVorbis else ".mp3"
		var f = File.new()
		if f.open(out_dir + ".sounds/" + fname, File.WRITE) == OK:
			f.store_buffer(st.data)
			f.close()
		info["loop"] = st.loop
		info["loop_offset"] = st.loop_offset
	else:
		fname = ""
		info["unsupported"] = st.get_class()
	info["file"] = fname
	saved_streams[key] = fname
	sound_log.store_line(to_json(info))
	return fname

func _log_sounds(f):
	# A sound started if its player turned on, jumped back (play() again) or got a new stream.
	if sound_log == null:
		return
	var vp = get_viewport()
	var listener = vp.get_canvas_transform().affine_inverse().xform(vp.size / 2)
	var keep = []
	for n in audio_nodes:
		if not is_instance_valid(n) or not n.is_inside_tree():
			continue
		keep.append(n)
		var id = n.get_instance_id()
		var playing = n.playing and n.stream != null
		var pos = n.get_playback_position() if playing else 0.0
		var was = audio_state.get(id, [false, 0.0, null])
		if playing and (not was[0] or pos + 0.001 < was[1] or n.stream != was[2]):
			var ev = {"ev": "start", "f": f, "id": id, "file": _stream_file(n.stream), "from": pos,
				"volume_db": n.volume_db, "pitch": n.pitch_scale, "bus": n.bus, "node": str(n.name)}
			if n is AudioStreamPlayer2D:
				var d = n.global_position - listener
				ev["dx"] = d.x
				ev["dy"] = d.y
				ev["max_distance"] = n.max_distance
				ev["attenuation"] = n.attenuation
			sound_log.store_line(to_json(ev))
		elif was[0] and not playing:
			sound_log.store_line(to_json({"ev": "stop", "f": f, "id": id}))
		audio_state[id] = [playing, pos, n.stream]
	audio_nodes = keep

func _game():
	var main = get_tree().current_scene
	if main and "game" in main and is_instance_valid(main.game):
		return main.game
	return null

func _process(_delta):
	var main = get_tree().current_scene
	if not started or not main or not "ui_layer" in main:
		return
	# Hide what is meant for a person at the keyboard: the replay hotkey hint ("R: Watch
	# replay - ..."), the one-time "large update" notice a fresh game profile shows, and the nag window.
	var hint = main.ui_layer.find_node("TopInfoReplay", true, false)
	if hint:
		hint.modulate.a = 0.0
	var notice = main.ui_layer.find_node("ModWarningWindow", true, false)
	if notice and notice.visible:
		notice.hide()
	# The game's "urgent message from Ivy Sly" nag: shown at random (1 in 50 launches) when Steam
	# isn't running, which is always true here. It bounces over the match for the whole replay.
	var nag = main.ui_layer.find_node("NagWindow", true, false)
	if nag and nag.visible:
		nag.hide()

func _capture():
	var g = _game()
	if g == null:
		return
	if states_log:
		_log_state(g)
	else:
		var img = get_viewport().get_texture().get_data()
		img.flip_y()
		img.save_png("%s/f%06d.png" % [out_dir, frame])
		_log_sounds(frame)
	frame += 1
	if finished_at < 0 and g.game_finished:
		finished_at = frame
		print("render: match over at frame ", frame, ", tick ", g.current_tick)
		var st = File.new()  # timing next to the frames, for checking playback speed against the game
		if st.open(out_dir + ".stats.json", File.WRITE) == OK:
			st.store_string(to_json({"end_frame": frame, "end_tick": g.current_tick,
				"replay_extra_freeze_frames": Global.replay_extra_freeze_frames,
				"playback_speed_mod": Global.playback_speed_mod}))
			st.close()
	if (finished_at >= 0 and frame >= finished_at + tail) or frame >= MAX_FRAMES:
		print("render: done, ", frame, " frames")
		if sound_log:
			sound_log.store_line(to_json({"ev": "end", "f": frame}))
			sound_log.close()
		if states_log:
			states_log.store_line(to_json({"end": true, "frame": frame, "tick": g.current_tick}))
			states_log.close()
		VisualServer.disconnect("frame_post_draw", self, "_capture")
		get_tree().quit()


# --yomi-states: one line per game tick the playback reaches (a frame can skip ticks; the line
# then carries how many). Positions and velocities as the game holds them (fixed-point strings for
# velocities would need the engine's detail path; floats are enough to find where a sim diverges).
func _log_state(g):
	if g.current_tick == last_tick:
		return
	var players = []
	for p in [g.p1, g.p2]:
		var pos = p.get_pos()
		var vel = p.get_vel()
		var st = p.current_state()
		players.append({"state": st.state_name if st else "", "state_tick": st.current_tick if st else -1,
			"x": pos.x, "y": pos.y, "vx": float(vel.x), "vy": float(vel.y), "hp": p.hp,
			"hitlag": p.hitlag_ticks, "busy_interrupt": p.busy_interrupt, "grounded": p.is_grounded(),
			"meter": p.supers_available, "meter_partial": p.super_meter, "combo": p.combo_count,
			"facing": p.get_facing_int()})
	var objects = []
	for o in g.objects:
		if is_instance_valid(o) and not o.disabled:
			var pos = o.get_pos()
			var st = o.current_state()
			objects.append({"name": str(o.obj_name), "kind": o.filename.get_file().get_basename(), "owner": o.id,
				"x": pos.x, "y": pos.y, "state": st.state_name if st else "",
				"state_tick": st.current_tick if st else -1})
	var line = {"tick": g.current_tick, "frame": frame, "players": players, "objects": objects}
	if last_tick >= 0 and g.current_tick > last_tick + 1:
		line["skipped"] = g.current_tick - last_tick - 1
	states_log.store_line(to_json(line))
	last_tick = g.current_tick
