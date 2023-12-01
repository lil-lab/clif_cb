"""Interface for the web agent."""
from __future__ import annotations

import logging
import sys
import socketio

logging.getLogger('apscheduler').setLevel(logging.WARNING)
logging.getLogger('engineio').setLevel(logging.WARNING)
logging.getLogger('socketio').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

from web_agent import follower_agent
from protobuf import CerealBarProto_pb2
from environment.action import Action
from environment.card import load_cards_from_proto
from environment.position import Position
from environment.rotation import degree_to_rotation

SOCKET = socketio.Client()

if sys.argv[1] == 'ensemble':
    print('Running with ensemble!')
    AGENT = follower_agent.NeuralNetworkFollower(SOCKET,
                                                 sys.argv[4],
                                                 sys.argv[2],
                                                 spec_file=sys.argv[3])
else:
    print('Running with a single model only.')
    AGENT = follower_agent.NeuralNetworkFollower(SOCKET,
                                                 sys.argv[4],
                                                 sys.argv[1],
                                                 experiment_name=sys.argv[2],
                                                 model_save_name=sys.argv[3],
                                                 do_sampling=sys.argv[5])
AGENT.connect()


def set_game_info(obj):
    obj.gameinfo.seed = ""
    obj.gameinfo.workerid = AGENT.get_id()
    obj.gameinfo.assignmentid = AGENT.get_id()
    obj.gameinfo.character = 'Follower'
    obj.gameinfo.gameid = AGENT.get_game_id()


def basic_game_info():
    gameinfo = CerealBarProto_pb2.StaticGameInfo()
    gameinfo.seed = ""
    gameinfo.workerid = AGENT.get_id()
    gameinfo.assignmentid = AGENT.get_id()
    gameinfo.character = 'Follower'
    gameinfo.gameid = AGENT.get_game_id()
    return gameinfo


@SOCKET.on('recognizedAgent')
def on_recognized(jsn=None):
    logging.info('Authorized, now joining lobby.')
    join_lobby = CerealBarProto_pb2.JoinLobby()
    join_lobby.gameinfo.seed = ""
    join_lobby.gameinfo.workerid = AGENT.get_id()
    join_lobby.gameinfo.assignmentid = AGENT.get_id()
    join_lobby.gameinfo.character = 'Follower'

    SOCKET.emit('joinLobby', join_lobby.SerializeToString())


@SOCKET.on('gameEnded')
def on_game_end(jsn):

    game_ended = CerealBarProto_pb2.GameEnded()
    game_ended.ParseFromString(jsn)
    logging.info('Game ended with %s points' % game_ended.finalscore)

    AGENT.end_game()


@SOCKET.on('initGame')
def on_game_init(jsn):
    # Just save the basic information. The js client doesn't do anything besides save the information here.
    logging.info('Got the initial game info')
    game_info = CerealBarProto_pb2.StaticGameInfo()
    game_info.ParseFromString(jsn)
    AGENT.set_game_id(game_info.gameid)
    AGENT.set_game_seed(game_info.seed)
    AGENT.set_num_cards(game_info.numcards)


@SOCKET.on('startGame')
def start_game(jsn):
    logging.info('Starting game.')
    SOCKET.emit('readyToStartGamePlay', basic_game_info().SerializeToString())

    map_info = CerealBarProto_pb2.MapInfo()
    map_info.ParseFromString(jsn)
    AGENT.process_map_info(map_info)


@SOCKET.on('lobbyReady')
def on_lobby_ready(jsn=None):
    logging.info('Lobby is ready.')
    SOCKET.emit('readyPressed', basic_game_info().SerializeToString())


@SOCKET.on('receiveMoreCards')
def receive_more_cards(jsn=None):
    card_list = CerealBarProto_pb2.ScoreSetCard()
    card_list.ParseFromString(jsn)
    new_cards = load_cards_from_proto(card_list.newcards)

    while not AGENT.has_set_removed():
        pass
    logging.info('Got new cards: ')
    for c in new_cards:
        logging.info(c)

    AGENT.add_new_cards(new_cards)


@SOCKET.on('reboot')
def reboot(data):
    AGENT.reboot()


@SOCKET.on('movement')
def on_movement(jsn):
    movement: CerealBarProto_pb2.Movement = CerealBarProto_pb2.Movement()
    movement.ParseFromString(jsn)
    if movement.character == 'Leader':
        logging.info('Leader moved: %s' % movement.type)
        move_id = int(movement.actionid.split('_')[-1])
        pos = Position(movement.nextposition.hexX, movement.nextposition.hexZ)
        rot = degree_to_rotation(movement.nextrotation)

        AGENT.move_leader(Action(movement.type), move_id, pos, rot)


@SOCKET.on('yourTurn')
def on_your_turn(jsn=None):
    logging.info('Received yourTurn message from server.')
    AGENT.set_turn(leader=False)


@SOCKET.on('requestAction')
def predict_action(jsn=None):
    AGENT.request_action()


@SOCKET.on('endTurn')
def end_turn(jsn=None):
    logging.info('Received endTurn message from server.')
    AGENT.set_turn(leader=True)


@SOCKET.on('instruction')
def on_instruction(jsn):
    instruction: CerealBarProto_pb2.Instruction = CerealBarProto_pb2.Instruction(
    )
    instruction.ParseFromString(jsn)
    logging.info('Leader made a new instruction: %s' %
                 instruction.instructiontext)
    instr_id = instruction.gameinfo.instructionid
    AGENT.add_instruction(instruction.instructiontext, instr_id)


@SOCKET.on('kill')
def on_kill(jsn):
    AGENT.process_end_of_game()
