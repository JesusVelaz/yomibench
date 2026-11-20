extends Node
# Our RPC endpoint in both games of a "Multiplayer (vs bot)" session, at /root/YomiBenchVsBot/MpLink in each. The
# host's VsBot.gd and the client's MpClient.gd each add one. It is one script on both sides because Godot 3.5
# addresses an RPC by the method's index in the script's list of remote methods: both ends must have the same list.
# Each call goes to the owner (VsBot.gd or MpClient.gd) with the sender's network id; the owner checks the sender.

var target = null


remote func mp_hello(token):  # client -> host
	if target and target.has_method("on_mp_hello"):
		target.on_mp_hello(get_tree().get_rpc_sender_id(), token)


remote func mp_from_bot(kind, payload):  # client -> host
	if target and target.has_method("on_mp_from_bot"):
		target.on_mp_from_bot(get_tree().get_rpc_sender_id(), kind, payload)


remote func mp_to_bot(kind, payload):  # host -> client
	if target and target.has_method("on_mp_to_bot"):
		target.on_mp_to_bot(get_tree().get_rpc_sender_id(), kind, payload)
